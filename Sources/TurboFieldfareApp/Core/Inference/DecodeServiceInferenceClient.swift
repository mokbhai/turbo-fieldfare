import Darwin
import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareDecodeProtocol

public final class DecodeServiceInferenceClient: AppModelLifecycleClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, @unchecked Sendable {
    /// Where a generation stands relative to the socket, which decides what a
    /// cancel can do about it. Only `.streaming` has a request the service can
    /// be asked to stop.
    private enum GenerationPhase {
        /// No generation. `expectingGeneration` is set by `expectGeneration()`
        /// and cleared when `generate` claims the phase: it marks the only
        /// stretch of time in which a cancel has a run to belong to but no
        /// request on the wire to name.
        case idle(expectingGeneration: Bool)
        /// A cancel landed with no generation registered. It belongs to the run
        /// the app has already started but has not framed yet, so it waits here
        /// for that run's `generate` to claim it.
        case cancelPending
        /// `generate` is framing its request; the service has not seen it.
        case starting(id: UUID, cancelRequested: Bool)
        /// The request is on the wire; only the service can end it now.
        case streaming(id: UUID)

        /// Which generation owns the phase, so a caller that speaks for one
        /// generation cannot act on another's. Every `generate` gets a stream
        /// and a termination handler, including the ones refused outright, and
        /// those handlers all run against this single shared phase.
        var owner: UUID? {
            switch self {
            case .idle, .cancelPending: nil
            case .starting(let id, _), .streaming(let id): id
            }
        }
    }

    /// What a `generate` call may do with the phase it tried to claim.
    private enum GenerationClaim {
        case claimed
        case cancelled
        case busy
    }

    private struct Connection {
        var input: FileHandle?
        var output: FileHandle?
        var loadedDirectory: URL?
        var launchLabel: String?
        var socketPath: String?
        var generation: GenerationPhase = .idle(expectingGeneration: false)
    }

    private let connection = Mutex(Connection())
    private let serviceURL: URL
    private let inferenceMemory = Mutex<UInt64?>(nil)
    public let generationTranscriptMailbox = GenerationTranscriptMailbox()

    public var currentInferenceMemoryBytes: UInt64? {
        inferenceMemory.withLock { $0 }
    }

    public init(serviceURL: URL? = nil) {
        self.serviceURL = serviceURL ?? Self.defaultServiceURL()
    }

    /// Test seam: adopts an already-connected pair of handles so the frames this
    /// client puts on the socket can be observed without launching the service.
    init(input: FileHandle, output: FileHandle,
         serviceURL: URL = URL(fileURLWithPath: "/nonexistent-decode-service")) {
        self.serviceURL = serviceURL
        connection.withLock {
            $0.input = input
            $0.output = output
        }
    }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.loading(.validatingDirectory))
        let handles = try await Task.detached(priority: .userInitiated) { [self] in
            try ensureProcess()
        }.value
        let request = DecodeLoadRequest(
            modelPath: modelDirectory.path, maxContextTokens: maxContextTokens,
            runtimeOptions: Self.decodeRuntimeOptions(options),
            forceLogitsHead: forceLogitsHead)
        try handles.input.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.load(request)))
        let event = try await readEvent(from: handles.output)
        guard event.generationID == request.requestID, event.kind == .ready else {
            throw AppInferenceError.modelLoadFailed(
                event.error ?? "decode service load failed")
        }
        inferenceMemory.withLock { $0 = event.currentMemoryBytes }
        connection.withLock { $0.loadedDirectory = modelDirectory.standardizedFileURL }
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    public func unload() async {
        guard let handles = currentHandles() else { return }
        let requestID = UUID()
        try? handles.input.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.unload(requestID)))
        _ = try? await readEvent(from: handles.output)
        connection.withLock { $0.loadedDirectory = nil }
        inferenceMemory.withLock { $0 = nil }
    }

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            // Minted out here so the termination handler below can name the
            // generation it speaks for.
            let generationID = UUID()
            let task = Task.detached(priority: .userInitiated) { [self] in
                // Claimed before anything else can fail: a cancel armed for this
                // run must be consumed by this run, or it would be left behind
                // to end the next one.
                switch claimGenerationStart(generationID) {
                case .cancelled:
                    // No `endGeneration()` here: the claim already left the
                    // phase idle, and wiping it again could discard a cancel
                    // that has since been armed for the following run.
                    continuation.yield(.cancelled(.cancelledBeforeGeneration(
                        runtimeOptions: request.runtimeOptions)))
                    continuation.finish(throwing: AppInferenceError.cancelled)
                    return
                case .busy:
                    continuation.yield(.failed(.generationInFlight, partial: nil))
                    continuation.finish(throwing: AppInferenceError.generationInFlight)
                    return
                case .claimed:
                    break
                }
                // Backstop only. Every terminal path below ends the generation
                // explicitly before finishing the stream; see `finish`.
                defer { endGeneration(generationID) }
                do {
                    try request.validate()
                    guard let handles = currentHandles() else {
                        throw AppInferenceError.modelNotLoaded
                    }
                    generationTranscriptMailbox.reset()
                    let command = DecodeGenerationRequest(
                        messages: request.messages, maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: Self.decodeRuntimeOptions(request.runtimeOptions),
                        generationID: generationID)
                    try handles.input.write(contentsOf: DecodeFrameCodec.encode(
                        DecodeServiceCommand.generate(command)))
                    // From here the service owns the request, so a cancel is
                    // written rather than held. One that arrived while this
                    // frame was being encoded is forwarded here, on this thread,
                    // so the two commands cannot interleave on the socket or
                    // reach the service before the request they refer to.
                    if markStreamingTakingPendingCancel(generationID) {
                        try? handles.input.write(contentsOf: DecodeFrameCodec.encode(
                            DecodeServiceCommand.cancel))
                    }

                    var expectedSequence: UInt64 = 1
                    var lastMetricYield = Date.distantPast
                    var hasYieldedVisibleText = false
                    while true {
                        let event = try DecodeFrameCodec.read(
                            DecodeServiceEvent.self, from: handles.output)
                        inferenceMemory.withLock { $0 = event.currentMemoryBytes }
                        guard event.generationID == generationID else { continue }

                        if event.kind == .prefill || event.kind == .snapshot {
                            guard event.sequence == expectedSequence else {
                                throw AppInferenceError.unknown(
                                    "decode service event sequence changed from \(expectedSequence) to \(event.sequence)")
                            }
                            expectedSequence &+= 1
                        }
                        if event.kind == .prefill,
                           let done = event.prefillDone,
                           let total = event.prefillTotal {
                            continuation.yield(.prefillProgress(done: done, total: total))
                            continue
                        }
                        if event.kind == .snapshot {
                            generationTranscriptMailbox.append(event.textDelta)
                            let now = Date()
                            let beginsVisibleText = !hasYieldedVisibleText
                                && event.textDelta.contains { !$0.isWhitespace }
                            if beginsVisibleText
                                || now.timeIntervalSince(lastMetricYield) >= 0.5 {
                                lastMetricYield = now
                                hasYieldedVisibleText = hasYieldedVisibleText || beginsVisibleText
                                continuation.yield(.token(AppTokenEvent(
                                    index: max(0, event.tokenCount - 1),
                                    textDelta: beginsVisibleText ? event.textDelta : "",
                                    elapsedDecodeSeconds: event.decodeSeconds)))
                            }
                            continue
                        }

                        let diagnostics = Self.diagnostics(
                            event, options: request.runtimeOptions)
                        switch event.kind {
                        case .finished:
                            continuation.yield(.finished(diagnostics))
                            finish(continuation, generationID)
                        case .cancelled:
                            continuation.yield(.cancelled(diagnostics))
                            finish(continuation, generationID)
                        case .failed:
                            let error = Self.failure(
                                event, maxContextTokens: request.maxContextTokens)
                            continuation.yield(.failed(error, partial: diagnostics))
                            finish(continuation, generationID, throwing: error)
                        default:
                            continue
                        }
                        return
                    }
                } catch {
                    finish(continuation, generationID, throwing: error)
                }
            }
            continuation.onTermination = { [weak self] _ in
                task.cancel()
                // Termination also fires on a normal finish, when the run is
                // already over. Stopping a request the service is still
                // producing is worth doing; arming a cancel is not, because
                // there is no longer a run for it to belong to and it would be
                // spent on the next one.
                self?.cancelStreamingGeneration(generationID)
            }
        }
    }

    /// Stops the current generation, whatever stage it has reached. Writing the
    /// cancel command only works once the request itself has been written;
    /// before that there is nothing on the far side to name, so the request is
    /// recorded and the generation is stopped where it still can be — locally,
    /// before it is ever sent.
    public func cancel() {
        let input = connection.withLock { state -> FileHandle? in
            switch state.generation {
            case .idle(let expectingGeneration):
                // Held only for a run that has started and not yet framed its
                // request. Outside that window this cancel followed a run that
                // is already over, and holding it would end the next one.
                if expectingGeneration { state.generation = .cancelPending }
                return nil
            case .cancelPending:
                return nil
            case .starting(let id, _):
                state.generation = .starting(id: id, cancelRequested: true)
                return nil
            case .streaming:
                return state.input
            }
        }
        writeCancelCommand(to: input)
    }

    public func expectGeneration() {
        connection.withLock { state in
            switch state.generation {
            case .idle, .cancelPending:
                // Also drops a cancel held for a run that never reached
                // `generate`: it cannot belong to the run starting now.
                state.generation = .idle(expectingGeneration: true)
            case .starting, .streaming:
                break
            }
        }
    }

    /// The half of `cancel()` that never arms anything: used where a cancel has
    /// no run of its own to belong to. Scoped to `generationID` so a stream that
    /// never owned the phase — one refused as busy, say — cannot stop the
    /// generation that does own it.
    private func cancelStreamingGeneration(_ generationID: UUID) {
        let input = connection.withLock { state -> FileHandle? in
            guard case .streaming(generationID) = state.generation else { return nil }
            return state.input
        }
        writeCancelCommand(to: input)
    }

    private func writeCancelCommand(to input: FileHandle?) {
        guard let input else { return }
        try? input.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.cancel))
    }

    /// Whether the generation that is starting may run. A cancel armed before
    /// `generate` was entered is consumed here — the one place it is consumed,
    /// so it can only ever end the generation it was armed for.
    private func claimGenerationStart(_ generationID: UUID) -> GenerationClaim {
        connection.withLock { state in
            switch state.generation {
            case .cancelPending:
                state.generation = .idle(expectingGeneration: false)
                return .cancelled
            case .starting, .streaming:
                // Another generation already owns the socket. Overwriting its
                // phase would let these two runs end each other, so this one is
                // refused instead — the same answer `RealInferenceClient` gives.
                return .busy
            case .idle:
                state.generation = .starting(id: generationID,
                                             cancelRequested: false)
                return .claimed
            }
        }
    }

    /// Marks the request as sent and reports whether a cancel arrived while it
    /// was being framed, which the caller must then forward.
    private func markStreamingTakingPendingCancel(_ generationID: UUID) -> Bool {
        connection.withLock { state in
            guard case .starting(generationID, let cancelRequested) = state.generation
            else { return false }
            state.generation = .streaming(id: generationID)
            return cancelRequested
        }
    }

    /// Releases the client when a generation ends. A cancel that arrives after
    /// this point finds no generation and is a no-op, which is what keeps a late
    /// Stop from being held against the next run. Scoped to `generationID` so a
    /// run that never owned the phase cannot release another run's.
    private func endGeneration(_ generationID: UUID) {
        connection.withLock { state in
            guard state.generation.owner == generationID else { return }
            state.generation = .idle(expectingGeneration: false)
        }
    }

    /// Ends the generation *before* finishing the stream.
    ///
    /// `onTermination` runs synchronously inside `finish()`, ahead of any
    /// `defer` in the task body. Left at `.streaming`, a run that ended
    /// perfectly normally would make `cancelStreamingGeneration()` write a
    /// cancel frame for a request the service has already completed. The
    /// service applies that frame out of band, finds no generation, and holds
    /// it against whatever runs next — so a successful generation would cancel
    /// the one after it.
    private func finish(
        _ continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
        _ generationID: UUID,
        throwing error: Error? = nil
    ) {
        endGeneration(generationID)
        if let error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
    }

    deinit {
        let state = connection.withLock { value -> Connection in
            defer { value = Connection() }
            return value
        }
        if let input = state.input {
            try? input.write(contentsOf: DecodeFrameCodec.encode(
                DecodeServiceCommand.shutdown))
            try? input.close()
        }
        if let label = state.launchLabel { Self.removeLaunchJob(label: label) }
        if let socketPath = state.socketPath { unlink(socketPath) }
    }

    private func ensureProcess() throws -> (input: FileHandle, output: FileHandle) {
        if let handles = currentHandles() { return handles }
        return try launchIndependentService()
    }

    private func launchIndependentService() throws
        -> (input: FileHandle, output: FileHandle) {
        guard FileManager.default.isExecutableFile(atPath: serviceURL.path) else {
            throw AppInferenceError.modelLoadFailed(
                "decode service executable is missing at \(serviceURL.path); run swift build -c release before launching the app")
        }
        let identifier = "\(getuid()).\(getpid()).\(UUID().uuidString.lowercased())"
        let label = "com.turbofieldfare.decode.\(identifier)"
        let socketPath = "/private/tmp/turbofieldfare-decode-\(identifier).sock"
        let propertyListURL = URL(
            fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("\(label).plist")
        let propertyList: [String: Any] = [
            "Label": label,
            "ProgramArguments": [
                serviceURL.path,
                "--socket", socketPath,
                "--launch-label", label,
            ],
            "RunAtLoad": true,
            "KeepAlive": false,
            "ProcessType": "Interactive",
        ]
        let propertyListData = try PropertyListSerialization.data(
            fromPropertyList: propertyList, format: .xml, options: 0)
        try propertyListData.write(to: propertyListURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: propertyListURL) }

        let launcher = Process()
        let errors = Pipe()
        launcher.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        launcher.arguments = [
            "bootstrap", "gui/\(getuid())", propertyListURL.path,
        ]
        launcher.standardOutput = FileHandle.nullDevice
        launcher.standardError = errors
        try launcher.run()
        launcher.waitUntilExit()
        guard launcher.terminationStatus == 0 else {
            let data = try? errors.fileHandleForReading.readToEnd()
            let detail = data.flatMap { String(data: $0, encoding: .utf8) }?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let message = detail.flatMap { $0.isEmpty ? nil : $0 }
                ?? "launchd could not start the decode service"
            throw AppInferenceError.modelLoadFailed(message)
        }

        var lastError: Error?
        for _ in 0..<200 {
            do {
                let handles = try DecodeUnixSocket.connect(path: socketPath)
                connection.withLock {
                    $0.input = handles.input
                    $0.output = handles.output
                    $0.launchLabel = label
                    $0.socketPath = socketPath
                }
                return handles
            } catch {
                lastError = error
                usleep(10_000)
            }
        }
        Self.removeLaunchJob(label: label)
        throw AppInferenceError.modelLoadFailed(
            "decode service socket did not become ready: \(lastError.map(String.init(describing:)) ?? "unknown error")")
    }

    private func currentHandles() -> (input: FileHandle, output: FileHandle)? {
        connection.withLock { state in
            guard let input = state.input, let output = state.output else {
                return nil
            }
            return (input, output)
        }
    }

    private func readEvent(from output: FileHandle) async throws
        -> DecodeServiceEvent {
        try await Task.detached(priority: .userInitiated) {
            try DecodeFrameCodec.read(DecodeServiceEvent.self, from: output)
        }.value
    }

    /// A `.failed` event only carries text; `errorCode` is what lets the UI tell
    /// a context overflow apart from a generic inference failure.
    static func failure(_ event: DecodeServiceEvent,
                        maxContextTokens: Int) -> AppInferenceError {
        if event.errorCode == "conversation-overflow" {
            return .conversationOverflow(prompt: event.promptTokenCount ?? 0,
                                         maxContext: maxContextTokens)
        }
        return .unknown(event.error ?? "decode service failed")
    }

    private static func diagnostics(_ event: DecodeServiceEvent,
                                    options: AppRuntimeOptions) -> AppDiagnostics {
        let stop = AppStopReason(rawValue: event.stopReason ?? "")
            ?? (event.kind == .cancelled
                ? .cancelled
                : event.kind == .failed ? .failed : .maxTokens)
        return AppDiagnostics(
            generatedTokens: event.tokenCount,
            stopReason: stop,
            promptTokenCount: event.promptTokenCount,
            cachedPromptTokens: event.cachedPromptTokens,
            prefillSeconds: event.prefillSeconds,
            timeToFirstTokenSeconds: event.timeToFirstTokenSeconds,
            decodeSeconds: event.decodeSeconds,
            tokensPerSecond: event.tokensPerSecond,
            peakMemoryBytes: event.peakMemoryBytes,
            runtimeOptions: options,
            prefill: prefillDiagnostics(event.prefill, options: options),
            runner: event.runner.map(runnerDiagnostics))
    }

    private static func prefillDiagnostics(
        _ value: DecodePrefillDiagnostics?, options: AppRuntimeOptions
    ) -> PrefillExecutionDiagnostics? {
        guard let value,
              let executedMode = PrefillExecutedMode(rawValue: value.executedMode),
              let completeness = PrefillChunkCompleteness(
                rawValue: value.chunkCompleteness) else { return nil }
        let kvStorage = value.kvStorageMode.flatMap(PrefillKVStorageMode.init(rawValue:))
        return PrefillExecutionDiagnostics(
            config: options.prefillConfig,
            executedMode: executedMode,
            kvStorageMode: kvStorage,
            chunkCompleteness: completeness,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: DecodeRunnerDiagnostics)
        -> AppRunnerDiagnostics {
        AppRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures)
    }

    private static func decodeRuntimeOptions(_ options: AppRuntimeOptions)
        -> DecodeRuntimeOptions {
        DecodeRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: options.expertCachePolicy.rawValue,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: options.rdadvisePolicy.rawValue,
            modelVerification: options.modelVerification.rawValue)
    }

    private static func removeLaunchJob(label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
    }

    private static func defaultServiceURL() -> URL {
        return Bundle.main.executableURL!
            .deletingLastPathComponent()
            .appendingPathComponent("TurboFieldfareDecodeService")
    }
}
