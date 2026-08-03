import Foundation
import Testing
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

/// Cancel coverage for the decode client. The first group needs no service at
/// all: the client answers a Stop that arrives before the request reaches the
/// socket, where there is nothing on the far side to cancel. The second group
/// drives a full generation over a pair of pipes so the frames the client puts
/// on the wire can be read back.
@Suite struct DecodeServiceInferenceClientTests {
    private func makeClient() -> DecodeServiceInferenceClient {
        DecodeServiceInferenceClient(
            serviceURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("no-decode-service-\(UUID().uuidString)"))
    }

    private func makeRequest() -> AppGenerationRequest {
        AppGenerationRequest(
            modelDirectory: FileManager.default.temporaryDirectory,
            messages: [.init(role: .user, content: "hello")])
    }

    @Test func cancelBeforeGenerateEndsThatGenerationAsCancelled() async throws {
        let client = makeClient()
        let request = makeRequest()

        client.expectGeneration()
        client.cancel()

        var events: [AppInferenceEvent] = []
        var thrown: Error?
        do {
            for try await event in client.generate(request) { events.append(event) }
        } catch {
            thrown = error
        }

        #expect(events.count == 1)
        if case .cancelled(let diagnostics) = events.first {
            #expect(diagnostics.stopReason == .cancelled)
            #expect(diagnostics.generatedTokens == 0)
        } else {
            Issue.record("expected a cancelled event, got \(events)")
        }
        // Terminates the same way `RealInferenceClient` does, so AppModel takes
        // one path for one user action whichever client is installed.
        #expect(thrown as? AppInferenceError == .cancelled)
    }

    @Test func cancelBeforeGenerateIsSpentOnOneGenerationOnly() async throws {
        let client = makeClient()
        let request = makeRequest()

        client.expectGeneration()
        client.cancel()
        _ = try? await drain(client.generate(request))

        // The next generation must get as far as the connection check, which it
        // cannot do if the cancel is still armed.
        var thrown: Error?
        do {
            for try await event in client.generate(request) {
                if case .cancelled = event { Issue.record("stale cancel killed the next run") }
            }
        } catch {
            thrown = error
        }
        #expect(thrown as? AppInferenceError == .modelNotLoaded)
    }

    @Test func cancelWhenIdleIsRepeatableAndDoesNotAccumulate() async throws {
        let client = makeClient()
        let request = makeRequest()

        client.expectGeneration()
        client.cancel()
        client.cancel()
        _ = try? await drain(client.generate(request))

        var thrown: Error?
        do {
            for try await _ in client.generate(request) {}
        } catch {
            thrown = error
        }
        #expect(thrown as? AppInferenceError == .modelNotLoaded)
    }

    /// A cancel that lands while no run is starting belongs to no run — the
    /// button was pressed a moment after the last one ended — and must be
    /// dropped rather than held against whatever runs next.
    @Test func cancelOutsideAStartingRunIsDropped() async throws {
        let client = makeClient()
        let request = makeRequest()

        client.cancel()

        var thrown: Error?
        do {
            for try await event in client.generate(request) {
                if case .cancelled = event { Issue.record("stale cancel killed the next run") }
            }
        } catch {
            thrown = error
        }
        #expect(thrown as? AppInferenceError == .modelNotLoaded)
    }

    /// The regression: a generation that ends normally must leave nothing on the
    /// socket behind it. `onTermination` runs synchronously inside `finish()`,
    /// ahead of the task body's `defer`, so a client that only cleared its phase
    /// in the `defer` wrote a `cancel` frame after every successful run. The
    /// service applies cancel frames out of band, finds no generation, and holds
    /// them against the next one — so run N killed run N+1.
    @Test func aFinishedGenerationWritesNoCancelFrame() async throws {
        let socket = FakeDecodeSocket()
        let client = DecodeServiceInferenceClient(input: socket.clientInput,
                                                  output: socket.clientOutput)
        let request = makeRequest()

        client.expectGeneration()
        let first = Task { try await drain(client.generate(request)) }
        let firstID = try await socket.readGenerationID()
        try socket.send(DecodeServiceEvent(kind: .finished, generationID: firstID,
                                           stopReason: AppStopReason.eos.rawValue))
        let firstEvents = try await first.value
        #expect(firstEvents.count == 1)
        if case .finished = firstEvents.first {} else {
            Issue.record("expected a finished event, got \(firstEvents)")
        }

        // The very next frame on the wire has to be the second request. Reading
        // it is the assertion: a stray cancel would be read here instead.
        client.expectGeneration()
        let second = Task { try await drain(client.generate(request)) }
        let secondID = try await socket.readGenerationID()
        #expect(secondID != firstID)
        try socket.send(DecodeServiceEvent(kind: .finished, generationID: secondID,
                                           stopReason: AppStopReason.eos.rawValue))
        _ = try await second.value
    }

    /// A Stop raised while the request is on the wire is still forwarded: the
    /// fix must not silence the cancel that has somewhere to go.
    @Test func cancelDuringAStreamingGenerationWritesACancelFrame() async throws {
        let socket = FakeDecodeSocket()
        let client = DecodeServiceInferenceClient(input: socket.clientInput,
                                                  output: socket.clientOutput)
        let request = makeRequest()

        client.expectGeneration()
        let run = Task { try await drain(client.generate(request)) }
        let generationID = try await socket.readGenerationID()

        client.cancel()
        guard case .cancel = try await socket.readCommand() else {
            Issue.record("expected a cancel frame")
            return
        }

        try socket.send(DecodeServiceEvent(kind: .cancelled, generationID: generationID,
                                           stopReason: AppStopReason.cancelled.rawValue))
        let events = try await run.value
        if case .cancelled = events.last {} else {
            Issue.record("expected a cancelled event, got \(events)")
        }
    }

    /// Two overlapping generations would otherwise overwrite each other's phase
    /// and end each other's runs; the second is refused instead.
    ///
    /// The refusal still hands back a stream, and that stream still terminates.
    /// Its termination handler must stay off the live generation: a handler that
    /// only asked "is anything streaming?" saw the *other* run's phase and wrote
    /// a cancel frame for it, so a second Generate killed the first.
    @Test func aSecondGenerationWhileOneIsStreamingIsRejectedWithoutEndingIt() async throws {
        let socket = FakeDecodeSocket()
        let client = DecodeServiceInferenceClient(input: socket.clientInput,
                                                  output: socket.clientOutput)
        let request = makeRequest()

        client.expectGeneration()
        let first = Task { try await drain(client.generate(request)) }
        let firstID = try await socket.readGenerationID()

        var thrown: Error?
        do {
            for try await _ in client.generate(request) {}
        } catch {
            thrown = error
        }
        #expect(thrown as? AppInferenceError == .generationInFlight)

        try socket.send(DecodeServiceEvent(kind: .finished, generationID: firstID,
                                           stopReason: AppStopReason.eos.rawValue))
        let firstEvents = try await first.value
        if case .finished = firstEvents.last {} else {
            Issue.record("expected the first run to finish, got \(firstEvents)")
        }

        // The refusal left nothing on the wire: the next frame the service sees
        // is the following request, not a cancel aimed at the run it refused to
        // join. Reading it is the assertion.
        client.expectGeneration()
        let second = Task { try await drain(client.generate(request)) }
        let secondID = try await socket.readGenerationID()
        try socket.send(DecodeServiceEvent(kind: .finished, generationID: secondID,
                                           stopReason: AppStopReason.eos.rawValue))
        _ = try await second.value
    }

    private func drain(_ stream: AsyncThrowingStream<AppInferenceEvent, Error>)
        async throws -> [AppInferenceEvent] {
        var events: [AppInferenceEvent] = []
        for try await event in stream { events.append(event) }
        return events
    }
}

/// A pair of pipes standing in for the decode service's unix socket, so a test
/// can read the frames the client writes and answer them.
///
/// `@unchecked Sendable`: each pipe end is used from one side of the exchange
/// only, and every read is handed to a dispatch queue rather than blocking a
/// cooperative thread the client's own task needs.
private final class FakeDecodeSocket: @unchecked Sendable {
    private let toService = Pipe()
    private let toClient = Pipe()

    var clientInput: FileHandle { toService.fileHandleForWriting }
    var clientOutput: FileHandle { toClient.fileHandleForReading }

    init() {
        // The client outlives this pair — its deinit still writes a shutdown
        // frame, and a failing test leaves it mid-exchange besides. Writing into
        // a pipe whose reader is gone raises SIGPIPE, which kills the whole test
        // bundle and reports the defect as "signal 13" instead of as the
        // expectation that caught it. Ignored so the write fails as EPIPE, which
        // every writer here already tolerates.
        signal(SIGPIPE, SIG_IGN)
    }

    func readCommand() async throws -> DecodeServiceCommand {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                continuation.resume(with: Result {
                    try DecodeFrameCodec.read(DecodeServiceCommand.self,
                                              from: toService.fileHandleForReading)
                })
            }
        }
    }

    /// Reads up to the generate request, recording every frame that arrives
    /// ahead of it. Those strays are the defect under test, and they are
    /// reported rather than thrown past: the exchange has to carry on to the
    /// request behind them so the run it belongs to can still be answered and
    /// the test can finish normally.
    func readGenerationID(
        sourceLocation: SourceLocation = #_sourceLocation
    ) async throws -> UUID {
        while true {
            let command = try await readCommand()
            if case .generate(let request) = command { return request.generationID }
            Issue.record("unexpected \(command) frame ahead of the generate request",
                         sourceLocation: sourceLocation)
        }
    }

    func send(_ event: DecodeServiceEvent) throws {
        try toClient.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(event))
    }
}

