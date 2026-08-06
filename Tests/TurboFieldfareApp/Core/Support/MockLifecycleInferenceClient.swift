import Foundation
import Testing
@testable import TurboFieldfareAppCore

final class MockLifecycleInferenceClient: AppModelLifecycleClient, @unchecked Sendable {
    var suspendUnloads = false
    var suspendLoads = false

    private let lock = NSLock()
    private var unloadContinuations: [CheckedContinuation<Void, Never>] = []
    private var unloadStartedCount = 0
    private var loadStartedCount = 0
    private var loadStateHandlers: [@Sendable (AppModelLoadState) -> Void] = []
    private var nextLoadFailure: AppInferenceError?
    private(set) var ensureLoadedCalls: [(URL, Int, AppRuntimeOptions, Bool)] = []

    func ensureLoaded(modelDirectory: URL,
                      maxContextTokens: Int,
                      options: AppRuntimeOptions,
                      forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        recordEnsureLoaded(modelDirectory: modelDirectory,
                           maxContextTokens: maxContextTokens,
                           options: options,
                           forceLogitsHead: forceLogitsHead)
        recordLoadStart(onState)

        onState(.loading(.validatingDirectory))
        while shouldSuspendLoad {
            try await Task.sleep(for: .milliseconds(5))
        }
        if let failure = takeLoadFailure() {
            onState(.failed(failure))
            throw failure
        }
        try Task.checkCancellation()
        onState(.ready(modelDirectory: modelDirectory.standardizedFileURL,
                       loadSeconds: 0))
    }

    func unload() async {
        if beginUnload() {
            await withCheckedContinuation { continuation in
                appendUnloadContinuation(continuation)
            }
        }

    }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(.finished(AppDiagnostics(
                    generatedTokens: 0,
                    stopReason: .eos,
                    promptTokenCount: 1,
                    prefillSeconds: nil,
                    timeToFirstTokenSeconds: nil,
                    decodeSeconds: 0,
                    tokensPerSecond: 0,
                    peakMemoryBytes: nil,
                    runtimeOptions: request.runtimeOptions)))
                continuation.finish()
            }

            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func cancel() {}

    func releaseUnloads() {
        let continuations: [CheckedContinuation<Void, Never>]
        lock.lock()
        continuations = unloadContinuations
        unloadContinuations.removeAll()
        suspendUnloads = false
        lock.unlock()
        continuations.forEach { $0.resume() }
    }

    func releaseLoads() {
        lock.lock()
        suspendLoads = false
        lock.unlock()
    }

    func failNextLoad(_ error: AppInferenceError) {
        lock.lock()
        nextLoadFailure = error
        suspendLoads = false
        lock.unlock()
    }

    func emitLoadState(_ state: AppModelLoadState, callIndex: Int) {
        let handler: (@Sendable (AppModelLoadState) -> Void)?
        lock.lock()
        handler = loadStateHandlers.indices.contains(callIndex)
            ? loadStateHandlers[callIndex]
            : nil
        lock.unlock()
        handler?(state)
    }

    func ensureLoadedCallCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return ensureLoadedCalls.count
    }

    /// Wall clock, and it reports its own expiry.
    ///
    /// These three waits used to be `for _ in 0..<200 { ...sleep(5ms) }` with
    /// no failure on running out, which is the same pair of defects
    /// `Tests/TurboFieldfareApp/Core/Support/WaitUntil.swift` documents: an
    /// iteration budget is worth about a second idle and far less under a
    /// loaded scheduler, and falling out silently leaves the caller's next
    /// `#expect` to fail on stale state and blame the wrong thing.
    private func wait(_ what: String,
                      sourceLocation: SourceLocation,
                      until condition: () -> Bool) async {
        let deadline = ContinuousClock.now + .seconds(30)
        while !condition() {
            guard ContinuousClock.now < deadline else {
                Issue.record("timed out waiting until \(what)",
                             sourceLocation: sourceLocation)
                return
            }
            try? await Task.sleep(for: .milliseconds(2))
        }
    }

    func waitForUnloadStart(sourceLocation: SourceLocation = #_sourceLocation) async {
        await wait("an unload starts", sourceLocation: sourceLocation) {
            unloadHasStarted
        }
    }

    func waitForLoadStart(_ expected: Int = 1,
                          sourceLocation: SourceLocation = #_sourceLocation) async {
        await wait("\(expected) load(s) have started",
                   sourceLocation: sourceLocation) {
            loadStartCount >= expected
        }
    }

    func waitForEnsureLoadedCallCount(_ expected: Int,
                                      sourceLocation: SourceLocation = #_sourceLocation) async {
        await wait("ensureLoaded has been called \(expected) time(s)",
                   sourceLocation: sourceLocation) {
            ensureLoadedCallCount() >= expected
        }
    }

    private var unloadHasStarted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return unloadStartedCount > 0
    }

    private var loadStartCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return loadStartedCount
    }

    private var shouldSuspendLoad: Bool {
        lock.lock()
        defer { lock.unlock() }
        return suspendLoads
    }

    private func recordEnsureLoaded(modelDirectory: URL,
                                    maxContextTokens: Int,
                                    options: AppRuntimeOptions,
                                    forceLogitsHead: Bool) {
        lock.lock()
        ensureLoadedCalls.append((modelDirectory.standardizedFileURL, maxContextTokens,
                                  options, forceLogitsHead))
        lock.unlock()
    }

    private func recordLoadStart(_ handler: @escaping @Sendable (AppModelLoadState) -> Void) {
        lock.lock()
        loadStartedCount += 1
        loadStateHandlers.append(handler)
        lock.unlock()
    }

    private func takeLoadFailure() -> AppInferenceError? {
        lock.lock()
        defer { lock.unlock() }
        defer { nextLoadFailure = nil }
        return nextLoadFailure
    }

    private func beginUnload() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        unloadStartedCount += 1
        return suspendUnloads
    }

    private func appendUnloadContinuation(_ continuation: CheckedContinuation<Void, Never>) {
        lock.lock()
        unloadContinuations.append(continuation)
        lock.unlock()
    }

}
