import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Model-free state coverage for the real client: load failure surfaces
/// before any network or Metal work, idle cancel is a no-op, and a bad
/// request fails the stream with a typed error.
@Suite struct RealInferenceClientStateTests {
    @Test func generationRegistryScopesTerminationToOwningID() async {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        let second = UUID()
        #expect(registry.reserve(first) == .reserved)
        registry.clear(first)
        #expect(registry.reserve(second) == .reserved)
        let secondTask = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(secondTask, to: second)

        #expect(registry.take(first) == nil)
        #expect(!secondTask.isCancelled)
        registry.take(second)?.cancel()
        #expect(secondTask.isCancelled)
    }

    @Test func generationRegistryRejectsConcurrentReservationAndClearsByOwner() {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        let second = UUID()
        #expect(registry.reserve(first) == .reserved)
        #expect(registry.reserve(second) == .busy)
        registry.clear(second)
        #expect(registry.reserve(second) == .busy)
        registry.clear(first)
        #expect(registry.reserve(second) == .reserved)
        registry.clear(second)
    }

    @Test func generationRegistryCancelsTaskAttachedAfterReservationEnded() async {
        let registry = GenerationTaskRegistry()
        let id = UUID()
        #expect(registry.reserve(id) == .reserved)
        registry.clear(id)
        let task = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(task, to: id)
        #expect(task.isCancelled)
        let next = UUID()
        #expect(registry.reserve(next) == .reserved)
        registry.clear(next)
    }

    @Test func generationRegistryHoldsACancelThatArrivesBeforeTheGeneration() {
        let registry = GenerationTaskRegistry()

        registry.expectGeneration()
        #expect(registry.cancelCurrentOrArmNext() == nil)
        #expect(registry.reserve(UUID()) == .cancelled)
        // Spent on that one generation only: the next reserve is a normal one.
        let next = UUID()
        #expect(registry.reserve(next) == .reserved)
        registry.clear(next)
    }

    /// The regression: a cancel that lands once a generation is over belongs to
    /// no run. The decode service delivers exactly that — it applies cancel
    /// frames out of band, and a frame written by a finished stream arrives
    /// after its generation ended — so holding it here would kill the run after.
    @Test func generationRegistryDropsACancelThatFollowsAFinishedGeneration() {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        registry.expectGeneration()
        #expect(registry.reserve(first) == .reserved)
        registry.clear(first)

        #expect(registry.cancelCurrentOrArmNext() == nil)

        let second = UUID()
        #expect(registry.reserve(second) == .reserved)
        registry.clear(second)
    }

    /// The decode service announces a generation the moment its frame arrives
    /// and can then refuse it — an unloaded model, mismatched runtime options —
    /// without ever reserving. Taking the announcement back is what stops the
    /// window staying open over a run that is never coming, where a later stray
    /// cancel would arm itself and be spent on whichever run does arrive.
    @Test func generationRegistryDropsACancelHeldForAnAbandonedGeneration() {
        let registry = GenerationTaskRegistry()
        registry.expectGeneration()
        #expect(registry.cancelCurrentOrArmNext() == nil)

        registry.abandonExpectedGeneration()

        // Neither the cancel already held for that run nor the next one to
        // arrive can reach the generation that eventually does reserve.
        #expect(registry.cancelCurrentOrArmNext() == nil)

        let id = UUID()
        #expect(registry.reserve(id) == .reserved)
        registry.clear(id)
    }

    @Test func generationRegistryTeardownDoesNotArmTheNextGeneration() {
        let registry = GenerationTaskRegistry()
        let id = UUID()
        #expect(registry.reserve(id) == .reserved)
        let task = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(task, to: id)

        registry.takeCurrent()?.cancel()

        #expect(task.isCancelled)
        #expect(registry.reserve(UUID()) == .reserved)
    }

    @Test func cancelBeforeGenerateEndsThatGenerationAsCancelled() async throws {
        let client = RealInferenceClient()
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            messages: [.init(role: .user, content: "hello")])

        client.expectGeneration()
        client.cancel()

        var events: [AppInferenceEvent] = []
        var thrown: Error?
        do {
            for try await event in client.generate(request) { events.append(event) }
        } catch {
            thrown = error
        }

        // Cancelled, not failed: the two are not interchangeable downstream.
        #expect(events.count == 1)
        if case .cancelled(let diagnostics) = events.first {
            #expect(diagnostics.stopReason == .cancelled)
            #expect(diagnostics.generatedTokens == 0)
        } else {
            Issue.record("expected a cancelled event, got \(events)")
        }
        #expect(thrown as? AppInferenceError == .cancelled)

        // The next generation is unaffected: it reaches the load check and fails
        // there, which it could not do if the cancel were still armed.
        var secondFailure: AppInferenceError?
        do {
            for try await event in client.generate(request) {
                if case .cancelled = event { Issue.record("stale cancel killed the next run") }
            }
        } catch let error as AppInferenceError {
            secondFailure = error
        }
        #expect(secondFailure != nil)
        #expect(secondFailure != .cancelled)
    }

    /// The client the decode service runs is cancelled out of band, so a cancel
    /// can land after the generation it named has ended. Driven through a real
    /// stream so the terminal ordering — `onTermination` firing inside
    /// `finish()` — is the production one rather than a stand-in for it.
    @Test func cancelLandingAfterAGenerationEndedDoesNotCancelTheNextOne() async throws {
        let client = RealInferenceClient()
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            messages: [.init(role: .user, content: "hello")])

        client.expectGeneration()
        var firstFailure: AppInferenceError?
        do {
            for try await _ in client.generate(request) {}
        } catch let error as AppInferenceError {
            firstFailure = error
        }
        #expect(firstFailure != nil)
        #expect(firstFailure != .cancelled)

        client.cancel()

        var secondFailure: AppInferenceError?
        do {
            for try await event in client.generate(request) {
                if case .cancelled = event { Issue.record("stale cancel killed the next run") }
            }
        } catch let error as AppInferenceError {
            secondFailure = error
        }
        // Reaches and fails at the same check the first run did, which it could
        // not do if the trailing cancel were still armed.
        #expect(secondFailure == firstFailure)
    }

    @Test func generationRunnerPolicyKeepsFusionHeadForPureGreedyChunkedPrefill() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            messages: [.init(role: .user, content: "hello")],
            temperature: 0,
            repetitionPenalty: 1)

        #expect(!RealInferenceSession.forceLogitsHead(for: request))
    }

    @Test func generationRunnerPolicyForcesLogitsForSamplingChunkedPrefill() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            messages: [.init(role: .user, content: "hello")],
            temperature: 0.7,
            repetitionPenalty: 1)

        #expect(RealInferenceSession.forceLogitsHead(for: request))
    }

    @Test func generationConfigCarriesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            messages: [.init(role: .user, content: "hello")])

        let config = RealInferenceSession.generationConfig(for: request)
        #expect(config.temperature == 0.2)
        #expect(config.topK == 64)
        #expect(config.topP == 0.95)
        #expect(config.repetitionPenalty == 1)
    }

    @Test func tokenizerDirectoryCacheReloadsOnlyWhenModelDirectoryChanges() {
        var cache = TokenizerDirectoryCache()
        let first = URL(fileURLWithPath: "/tmp/first.gturbo")
        let second = URL(fileURLWithPath: "/tmp/second.gturbo")

        #expect(cache.shouldReload(for: first))
        cache.markLoaded(for: first)
        #expect(!cache.shouldReload(for: first))
        #expect(cache.shouldReload(for: second))
        cache.clear()
        #expect(cache.shouldReload(for: first))
    }

    @Test func generateWithoutLoadedModelFailsWithoutPartialDiagnostics() async throws {
        let client = RealInferenceClient()
        let modelDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-prefill-off-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDirectory,
                                                withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let request = AppGenerationRequest(
            modelDirectory: modelDirectory,
            messages: [.init(role: .user, content: "hello")],
            runtimeOptions: AppRuntimeOptions(prefillEnabled: false))

        var failure: AppInferenceError?
        var partial: AppDiagnostics?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, let diagnostics) = event {
                    failure = error
                    partial = diagnostics
                }
            }
        } catch let error as AppInferenceError {
            failure = failure ?? error
        } catch {
            Issue.record("unexpected error type: \(error)")
        }

        #expect(failure != nil)
        #expect(partial == nil)
    }

    @Test func ensureLoadedFailsFastForMissingDirectory() async {
        let client = RealInferenceClient()
        var states: [AppModelLoadState] = []
        let recorder = StateRecorder()

        await #expect(throws: AppInferenceError.self) {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
                maxContextTokens: 1024,
                options: AppRuntimeOptions(),
                forceLogitsHead: false,
                onState: { recorder.append($0) })
        }
        states = recorder.snapshot()
        #expect(states.first == .loading(.validatingDirectory))
        #expect(states.last?.isFailed == true)
        #expect(!states.contains(.loading(.tokenizer)))
    }

    @Test func generateWithMissingDirectoryFailsStream() async {
        let client = RealInferenceClient()
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            messages: [.init(role: .user, content: "hello")])

        var failure: AppInferenceError?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, _) = event { failure = error }
            }
        } catch let error as AppInferenceError {
            failure = failure ?? error
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(failure == .modelNotFound("/nonexistent/model.gturbo"))
    }

    @Test func prefillFailureDiagnosticsMarksUnsupportedModeAndReason() {
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)

        let diagnostics = RealInferenceSession.prefillFailureDiagnostics(
            config: config,
            kvStorageMode: .fp16,
            reason: "chunked prefill synthetic unsupported diagnostic")

        #expect(diagnostics.requestedMode == .chunked)
        #expect(diagnostics.executedMode == .unsupported)
        #expect(diagnostics.chunkCompleteness == .unsupported)
        #expect(diagnostics.kvStorageMode == .fp16)
        #expect(diagnostics.unsupportedReason?.contains("synthetic unsupported") == true)
    }

    /// A cancel raised while the run that is starting has no stream yet is held
    /// for it, so repeating it must stay harmless — and must not stack up into
    /// several cancels.
    @Test func repeatedCancelWhenIdleIsSafe() async throws {
        let client = RealInferenceClient()
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            messages: [.init(role: .user, content: "hello")])

        client.expectGeneration()
        client.cancel()
        client.cancel()

        var cancelledGenerations = 0
        for _ in 0..<2 {
            do {
                for try await event in client.generate(request) {
                    if case .cancelled = event { cancelledGenerations += 1 }
                }
            } catch {}
        }
        #expect(cancelledGenerations == 1)
    }

    @Test func unloadWhenIdleIsSafe() async {
        let client = RealInferenceClient()
        await client.unload()
    }
}

private final class StateRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [AppModelLoadState] = []

    func append(_ state: AppModelLoadState) {
        lock.lock()
        states.append(state)
        lock.unlock()
    }

    func snapshot() -> [AppModelLoadState] {
        lock.lock()
        defer { lock.unlock() }
        return states
    }
}
