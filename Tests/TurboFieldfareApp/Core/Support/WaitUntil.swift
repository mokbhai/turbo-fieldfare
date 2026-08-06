import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// Polls `condition` until it holds or the wall-clock budget runs out, and
/// records an issue naming the wait if it runs out.
///
/// Replaces the hand-rolled
/// `for _ in 0..<200 where <not yet> { try? await Task.sleep(...) }` loops this
/// target used to carry. Those had two defects that compounded into an
/// intermittent failure with a misleading message:
///
/// 1. **The budget was an iteration count, not a duration.** Every `AppModel`
///    test here is `@MainActor`, so all of them serialize onto one executor.
///    Run serially — `Scripts/test.sh`, which is `swift test --no-parallel` —
///    a resumption costs microseconds and 200 x 5 ms means about a second.
///    Run in parallel, every resumption queues behind every other MainActor
///    test's work and the same loop stretches past twenty seconds, so it can
///    exhaust its iterations while the work it is waiting for is still in
///    flight. A wall-clock deadline means the same thing under both.
///
/// 2. **Expiry was silent.** The loop just fell out and left the next
///    assertion to read stale state, so a timeout was indistinguishable from a
///    genuine regression: a run that never finished surfaced as "expected the
///    second chat's error, got the first chat's" three lines further down —
///    an accurate description of a bug that was not there. Recording the
///    timeout at the wait names the actual failure.
///
/// The budget is deliberately generous. A healthy test satisfies its condition
/// on the first or second poll and pays nothing for the headroom; only a test
/// that is already failing waits out the deadline.
@MainActor
func waitUntil(_ what: @autoclosure () -> String,
               within timeout: Duration = .seconds(30),
               pollingEvery interval: Duration = .milliseconds(2),
               sourceLocation: SourceLocation = #_sourceLocation,
               _ condition: @MainActor () -> Bool) async {
    let deadline = ContinuousClock.now + timeout
    while !condition() {
        guard ContinuousClock.now < deadline else {
            Issue.record("timed out waiting until \(what())",
                         sourceLocation: sourceLocation)
            return
        }
        try? await Task.sleep(for: interval)
    }
}

/// The wait every generation test needs: the run task has finished and the
/// model is idle again.
@MainActor
func waitUntilIdle(_ model: AppModel,
                   sourceLocation: SourceLocation = #_sourceLocation) async {
    await waitUntil("the generation finishes",
                    sourceLocation: sourceLocation) { !model.isRunning }
}
