import Foundation
@testable import TurboFieldfareAppCore

/// Test double: streams canned word tokens behind the `AppInferenceClient`
/// protocol so AppModel control flow stays testable without a model load.
final class MockInferenceClient: AppInferenceClient, @unchecked Sendable {
    var response: String
    var tokenDelayNanos: UInt64
    var failureMessage: String?
    var prefillSteps: Int = 3

    private let lock = NSLock()
    private var activeTask: Task<Void, Never>?
    private var activeGenerationID: UUID?
    private var pendingCancel = false
    private var expectsGeneration = false
    private var entryGate: DispatchSemaphore?
    private var isHeldAtEntry = false
    private let memorySampler: AppMemorySampler

    init(response: String = "This is a lightweight mock response streaming through the TurboFieldfare Mac shell.",
         tokenDelayNanos: UInt64 = 35_000_000,
         memorySampler: AppMemorySampler = AppMemorySampler(),
         failureMessage: String? = nil) {
        self.response = response
        self.tokenDelayNanos = tokenDelayNanos
        self.memorySampler = memorySampler
        self.failureMessage = failureMessage
    }

    /// Holds the next `generate` at its first instruction, before it has
    /// registered anything a cancel could act on. That window is where the real
    /// clients live too — `RealInferenceClient` only becomes cancellable once
    /// `generate` has reserved its task, and the decode service only once the
    /// request has reached the socket — so this is what lets a test land a
    /// cancel inside it instead of racing for it.
    func holdBeforeGenerate() {
        lock.lock()
        entryGate = DispatchSemaphore(value: 0)
        lock.unlock()
    }

    /// True once a `generate` call is parked at that gate.
    var isHeldBeforeGenerate: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isHeldAtEntry
    }

    func releaseGenerate() {
        lock.lock()
        let gate = entryGate
        entryGate = nil
        lock.unlock()
        gate?.signal()
    }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        // Blocking is the point: `generate` is synchronous, so a caller that has
        // asked for a generation cannot be observed to have one until it
        // returns.
        lock.lock()
        let gate = entryGate
        if gate != nil { isHeldAtEntry = true }
        lock.unlock()
        gate?.wait()
        if gate != nil {
            lock.lock()
            isHeldAtEntry = false
            lock.unlock()
        }

        return AsyncThrowingStream { continuation in
            lock.lock()
            // In flight is decided first, as `GenerationTaskRegistry.reserve`
            // decides it: a second, concurrent `generate` never starts a run,
            // so it must not spend a cancel raised for the run already going.
            if activeTask != nil {
                lock.unlock()
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            // Consumed before any other outcome so that a cancel armed for this
            // generation is spent on it and cannot reach the next one.
            expectsGeneration = false
            let wasCancelledBeforeStart = pendingCancel
            pendingCancel = false
            lock.unlock()
            if wasCancelledBeforeStart {
                continuation.yield(.cancelled(.cancelledBeforeGeneration(
                    runtimeOptions: request.runtimeOptions)))
                continuation.finish(throwing: AppInferenceError.cancelled)
                return
            }

            do {
                try request.validate()
            } catch {
                let appError = error as? AppInferenceError ?? .unknown("\(error)")
                continuation.yield(.failed(appError, partial: nil))
                continuation.finish(throwing: appError)
                return
            }

            lock.lock()
            memorySampler.resetPeak()
            _ = memorySampler.sample()
            let generationID = UUID()
            let task = Task { [self] in
                await run(request: request, generationID: generationID, continuation: continuation)
            }
            activeTask = task
            activeGenerationID = generationID
            lock.unlock()

            continuation.onTermination = { [weak self] _ in
                // Never `cancel()`: termination also means "finished", and
                // arming a cancel here would leave it for the next generation.
                self?.cancelActiveTask()
            }
        }
    }

    func cancel() {
        lock.lock()
        let task = activeTask
        // Stop reached the client before `generate` registered anything. Hold
        // the request for the generation that is starting instead of dropping
        // it, which is what let a cancelled run stream to completion — but only
        // inside the window a starting run opened, mirroring the real clients.
        if task == nil, expectsGeneration { pendingCancel = true }
        activeTask = nil
        activeGenerationID = nil
        lock.unlock()
        task?.cancel()
    }

    /// Opens the window, and deliberately does no more than that. Wiping a
    /// leftover cancel here as well would answer every stale cancel by the wipe
    /// rather than by the window, and the AppModel tests that name the window
    /// would then pass against a fake that has no window at all.
    func expectGeneration() {
        lock.lock()
        expectsGeneration = true
        lock.unlock()
    }

    private func cancelActiveTask() {
        lock.lock()
        let task = activeTask
        activeTask = nil
        activeGenerationID = nil
        lock.unlock()
        task?.cancel()
    }

    private func clearActiveTask(generationID: UUID) {
        lock.lock()
        if activeGenerationID == generationID {
            activeTask = nil
            activeGenerationID = nil
        }
        lock.unlock()
    }

    private func run(request: AppGenerationRequest,
                     generationID: UUID,
                     continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        defer { clearActiveTask(generationID: generationID) }

        if let failureMessage {
            let error = AppInferenceError.unknown(failureMessage)
            let diagnostics = makeDiagnostics(request: request, generated: 0, start: Date(),
                                              firstToken: nil, stopReason: .failed)
            continuation.yield(.failed(error, partial: diagnostics))
            continuation.finish(throwing: error)
            return
        }

        let start = Date()
        var firstTokenDate: Date?
        var prefillEndDate: Date?
        let pieces = Array(responsePieces(for: request)).prefix(request.maxNewTokens)
        var generated = 0

        for step in 0..<max(prefillSteps, 0) {
            if Task.isCancelled {
                let diagnostics = makeDiagnostics(request: request, generated: 0,
                                                  start: start, firstToken: nil,
                                                  prefillEnd: Date(),
                                                  stopReason: .cancelled)
                continuation.yield(.cancelled(diagnostics))
                continuation.finish(throwing: AppInferenceError.cancelled)
                return
            }
            try? await Task.sleep(nanoseconds: tokenDelayNanos)
            continuation.yield(.prefillProgress(done: step + 1, total: prefillSteps))
        }
        prefillEndDate = Date()

        for (index, piece) in pieces.enumerated() {
            let decodeStart = prefillEndDate ?? start
            if Task.isCancelled {
                let diagnostics = makeDiagnostics(request: request, generated: generated,
                                                  start: start, firstToken: firstTokenDate,
                                                  prefillEnd: prefillEndDate,
                                                  stopReason: .cancelled)
                continuation.yield(.cancelled(diagnostics))
                continuation.finish(throwing: AppInferenceError.cancelled)
                return
            }
            try? await Task.sleep(nanoseconds: tokenDelayNanos)
            if Task.isCancelled {
                let diagnostics = makeDiagnostics(request: request, generated: generated,
                                                  start: start, firstToken: firstTokenDate,
                                                  prefillEnd: prefillEndDate,
                                                  stopReason: .cancelled)
                continuation.yield(.cancelled(diagnostics))
                continuation.finish(throwing: AppInferenceError.cancelled)
                return
            }
            if firstTokenDate == nil { firstTokenDate = Date() }
            generated += 1
            _ = memorySampler.sample()
            continuation.yield(.token(AppTokenEvent(
                index: index,
                textDelta: piece,
                elapsedDecodeSeconds: max(Date().timeIntervalSince(decodeStart), 0))))
        }

        let diagnostics = makeDiagnostics(request: request, generated: generated,
                                          start: start, firstToken: firstTokenDate,
                                          prefillEnd: prefillEndDate,
                                          stopReason: generated >= request.maxNewTokens ? .maxTokens : .eos)
        continuation.yield(.finished(diagnostics))
        continuation.finish()
    }

    private func responsePieces(for request: AppGenerationRequest) -> [String] {
        let trimmed = (request.latestUserContent ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let text = trimmed.isEmpty ? response : "\(response) Prompt received: \(trimmed)"
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        guard !words.isEmpty else { return [""] }
        return words.enumerated().map { index, word in
            index == 0 ? String(word) : " " + word
        }
    }

    private func makeDiagnostics(request: AppGenerationRequest,
                                 generated: Int,
                                 start: Date,
                                 firstToken: Date?,
                                 prefillEnd: Date? = nil,
                                 stopReason: AppStopReason) -> AppDiagnostics {
        _ = memorySampler.sample()
        let decodeStart = prefillEnd ?? start
        let decodeElapsed = max(Date().timeIntervalSince(decodeStart), 0)
        return AppDiagnostics(
            generatedTokens: generated,
            stopReason: stopReason,
            promptTokenCount: mockPromptTokenCount(request.latestUserContent ?? ""),
            prefillSeconds: prefillEnd.map { max($0.timeIntervalSince(start), 0) },
            timeToFirstTokenSeconds: firstToken.map { max($0.timeIntervalSince(decodeStart), 0) },
            decodeSeconds: decodeElapsed,
            tokensPerSecond: decodeElapsed > 0 ? Double(generated) / decodeElapsed : 0,
            peakMemoryBytes: memorySampler.peakBytes,
            runtimeOptions: request.runtimeOptions,
            runner: nil)
    }

    private func mockPromptTokenCount(_ prompt: String) -> Int {
        max(1, prompt.split(whereSeparator: \.isWhitespace).count)
    }
}
