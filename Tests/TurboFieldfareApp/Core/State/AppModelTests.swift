import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelTests {
    @MainActor
    @Test func defaultsUseSampledRequest() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        let request = try model.makeRequest()
        #expect(request.temperature == 0.2)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.maxNewTokens == 4_096)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
        #expect(request.runtimeOptions.expertCacheSlots == 16)
        #expect(request.runtimeOptions.expertCachePolicy == .lfu)
        #expect(request.runtimeOptions.rdadvisePolicy == .off)
        #expect(request.runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func runDisabledWhenPromptEmpty() {
        let model = AppModel()
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "   "
        #expect(!model.canRun)
    }

    @MainActor
    @Test func runDisabledUntilModelReady() {
        let model = AppModel()
        model.promptText = "go"
        #expect(!model.canRun)
    }

    @MainActor
    @Test func disablingTopKNeutralizesBothTruncationControls() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.topKEnabled = false
        model.topPEnabled = true

        let request = try model.makeRequest()
        #expect(request.topK == nil)
        #expect(request.topP == nil)
    }

    @MainActor
    @Test func prefillToggleSurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        model.runtimeOptions.prefillEnabled = false
        #expect(try !model.makeRequest().runtimeOptions.prefillEnabled)

        model.runtimeOptions.prefillEnabled = true
        #expect(try model.makeRequest().runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func adaptiveRDAdvicePolicySurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.runtimeOptions.rdadvisePolicy = .adaptive

        let request = try model.makeRequest()
        #expect(request.runtimeOptions.rdadvisePolicy == .adaptive)
    }

    @MainActor
    @Test func loadAffectingRuntimeChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func contextChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.maxContextTokens = 8_192
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func appResponseLimitUsesSelectedContext() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.maxContextTokens = 65_536

        #expect(try model.makeRequest().maxNewTokens == 65_536)
    }

    @MainActor
    @Test func requestTimePrefillChangeDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.runtimeOptions.prefillEnabled = false

        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func mockRunUpdatesOutputAndDiagnostics() async throws {
        let client = MockInferenceClient(response: "alpha beta", tokenDelayNanos: 1)
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.maxNewTokensOverride = 4
        model.run()

        await waitUntilIdle(model)

        #expect(!model.isRunning)
        #expect(model.viewedConversationPlainText.contains("alpha beta"))
        #expect(model.diagnostics != nil)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func runSnapshotsPromptIntoOutputTranscript() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1
        model.run()

        #expect(model.outputPromptText == "original prompt")
        #expect(model.hasOutputTranscript)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.viewedConversationPlainText == "You:\noriginal prompt")

        model.promptText = "edited prompt"
        await waitForIdle(model)

        // A completed exchange moves from the live turn into committed history.
        #expect(model.outputPromptText.isEmpty)
        #expect(model.committedTurns.map(\.content) == ["original prompt", "answer"])
        #expect(model.viewedConversationPlainText
            == "You:\noriginal prompt\n\nAnswer:\nanswer")
        #expect(!model.viewedConversationPlainText.contains("edited prompt"))
    }

    @MainActor
    @Test func acceptedRunClearsComposerButKeepsPromptInTranscript() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1

        // The composer still feeds the request; only the accepted send clears it.
        let request = try model.makeRequest()
        #expect(request.messages.last?.content == "original prompt")
        #expect(request.messages.last?.role == .user)

        model.run()

        #expect(model.promptText.isEmpty)
        #expect(model.outputPromptText == "original prompt")
        #expect(model.viewedConversationPlainText == "You:\noriginal prompt")

        // A prompt typed while the answer streams must survive the terminal event.
        model.promptText = "next prompt"
        await waitForIdle(model)

        #expect(model.promptText == "next prompt")
        #expect(model.committedTurns.map(\.content) == ["original prompt", "answer"])
    }

    @MainActor
    @Test func refusedRunKeepsComposerText() async throws {
        let blankPrompt = readyModel(client: MockInferenceClient())
        blankPrompt.promptText = "   "
        #expect(!blankPrompt.canRun)
        blankPrompt.run()
        #expect(blankPrompt.promptText == "   ")

        let notReady = AppModel(client: MockInferenceClient())
        notReady.promptText = "go"
        #expect(!notReady.canRun)
        notReady.run()
        #expect(notReady.promptText == "go")

        let overflowing = readyModel(client: MockInferenceClient())
        overflowing.maxContextTokens = 4_096
        let longPrompt = String(repeating: "x", count: 4_096 * 4)
        overflowing.promptText = longPrompt
        #expect(overflowing.isConversationOverflowing)
        #expect(overflowing.canRun)
        overflowing.run()
        #expect(overflowing.isContextOverflowNoticeVisible)
        #expect(overflowing.promptText == longPrompt)

        let invalidRequest = readyModel(client: MockInferenceClient())
        invalidRequest.promptText = "go"
        invalidRequest.topK = 300
        invalidRequest.run()
        #expect(invalidRequest.error != nil)
        #expect(!invalidRequest.isRunning)
        #expect(invalidRequest.promptText == "go")

        // The `isRunning` arm of `canRun`: hitting send while an answer streams
        // must not swallow the text typed for the next send.
        let busy = readyModel(
            client: MockInferenceClient(response: "one two three", tokenDelayNanos: 20_000_000))
        busy.promptText = "first"
        busy.run()
        #expect(busy.isRunning)
        busy.promptText = "second"
        #expect(!busy.canRun)
        busy.run()
        #expect(busy.promptText == "second")
        busy.cancel()
        await waitForIdle(busy)
        #expect(busy.promptText == "second")
    }

    @MainActor
    @Test func cancelledRunKeepsComposerText() async throws {
        let client = MockInferenceClient(response: "one two three", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "cancel me"
        model.run()
        #expect(model.promptText.isEmpty)

        model.promptText = "typed during run"
        await waitUntil("the run emits its first token") { model.liveTokenCount > 0 }
        #expect(model.liveTokenCount > 0)
        model.cancel()
        await waitForIdle(model)

        #expect(model.error == .cancelled)
        #expect(model.promptText == "typed during run")
    }

    @MainActor
    @Test func failedRunRestoresThePromptToAnEmptyComposer() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "retry me"
        model.run()
        #expect(model.promptText.isEmpty)

        await waitForIdle(model)

        // Nothing was committed, so the composer is the only place the prompt can
        // survive an offer to clear the output. Retrying must cost one click.
        #expect(model.error != nil)
        #expect(model.committedTurns.isEmpty)
        #expect(model.promptText == "retry me")
        #expect(model.canRun)
    }

    @MainActor
    @Test func failedRunRestoresThePromptOverWhitespaceTypedMidStream() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "retry me"
        model.run()
        // A stray space is not a prompt the user is protecting. Measuring
        // emptiness untrimmed here would treat it as one and drop "retry me".
        model.promptText = "  \n "
        await waitForIdle(model)

        #expect(model.error != nil)
        #expect(model.promptText == "retry me")
        #expect(model.canRun)
    }

    @MainActor
    @Test func cancelledRunRestoresThePromptToAnEmptyComposer() async throws {
        let client = MockInferenceClient(response: "one two three", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "cancel me"
        model.run()
        #expect(model.promptText.isEmpty)

        await waitUntil("the run emits its first token") { model.liveTokenCount > 0 }
        #expect(model.liveTokenCount > 0)
        model.cancel()
        await waitForIdle(model)

        #expect(model.error == .cancelled)
        #expect(model.committedTurns.isEmpty)
        #expect(model.promptText == "cancel me")
        #expect(model.canRun)
    }

    @MainActor
    @Test func failedRunKeepsTypedAheadTextInsteadOfTheRestoredPrompt() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "fail"
        model.run()

        model.promptText = "typed during run"
        await waitForIdle(model)

        // The restore is for an untouched composer only: text typed while the
        // run was in flight is newer than the submitted prompt and wins.
        #expect(model.error != nil)
        #expect(model.promptText == "typed during run")
        #expect(model.outputPromptText == "fail")
    }

    @MainActor
    @Test func successfulRunLeavesTheComposerClear() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        model.promptText = "a question"
        model.maxNewTokensOverride = 1
        model.run()
        await waitForIdle(model)

        // A committed turn is a recoverable copy, so there is nothing to restore
        // and the composer stays ready for the next prompt.
        #expect(model.error == nil)
        #expect(model.committedTurns.map(\.content) == ["a question", "answer"])
        #expect(model.promptText.isEmpty)
    }

    @MainActor
    @Test func promptExamplesCardShowsOnlyInAFreshConversation() {
        let model = readyModel(client: MockInferenceClient())
        #expect(model.isPromptExamplesCardVisible)

        model.promptText = "a question"
        #expect(!model.isPromptExamplesCardVisible)

        model.promptText = ""
        #expect(model.isPromptExamplesCardVisible)
    }

    @MainActor
    @Test func promptExamplesCardStaysHiddenWhileTheAnswerStreams() async throws {
        let client = MockInferenceClient(response: "one two three", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "a question"
        model.run()

        // Accepting the request empties the composer, so `promptText.isEmpty` is
        // true from here on. The card must stay hidden anyway: showing it would
        // animate a full-width panel in over the streaming answer.
        #expect(model.promptText.isEmpty)
        #expect(model.isRunning)
        #expect(!model.isPromptExamplesCardVisible)

        await waitUntil("the run emits its first token") { model.liveTokenCount > 0 }
        #expect(model.liveTokenCount > 0)
        #expect(!model.isPromptExamplesCardVisible)
    }

    @MainActor
    @Test func promptExamplesCardStaysHiddenAfterASuccessfulRun() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        model.promptText = "a question"
        model.maxNewTokensOverride = 1
        model.run()
        await waitForIdle(model)

        // A cleared composer over a committed transcript is not a fresh chat.
        #expect(!model.committedTurns.isEmpty)
        #expect(model.promptText.isEmpty)
        #expect(!model.isPromptExamplesCardVisible)
    }

    @MainActor
    @Test func promptExamplesCardStaysHiddenAfterACancelledOrFailedRunUntilCleared() async throws {
        // Neither terminal path commits its turn, but both leave the exchange on
        // screen, so the state is not fresh. Only clearing the output is.
        let cancelledClient = MockInferenceClient(response: "one two three", tokenDelayNanos: 20_000_000)
        cancelledClient.prefillSteps = 0
        let cancelled = readyModel(client: cancelledClient)
        cancelled.promptText = "a question"
        cancelled.run()
        await waitUntil("the run emits its first token") { cancelled.liveTokenCount > 0 }
        #expect(cancelled.liveTokenCount > 0)
        cancelled.cancel()
        await waitForIdle(cancelled)

        #expect(cancelled.error == .cancelled)
        #expect(cancelled.hasOutputTranscript)
        // The prompt came back to the composer, which hides the card on its own.
        #expect(cancelled.promptText == "a question")
        #expect(!cancelled.isPromptExamplesCardVisible)

        cancelled.clearOutput()
        #expect(!cancelled.isPromptExamplesCardVisible)
        cancelled.promptText = ""
        #expect(cancelled.isPromptExamplesCardVisible)

        let failed = readyModel(
            client: MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure"))
        failed.promptText = "a question"
        failed.run()
        await waitForIdle(failed)

        #expect(failed.error != nil)
        #expect(failed.hasOutputTranscript)
        #expect(!failed.isPromptExamplesCardVisible)
    }

    @MainActor
    @Test func successiveRunsAccumulateTurnsAndSendTheWholeConversation() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.maxNewTokensOverride = 1

        model.promptText = "first"
        model.run()
        await waitForIdle(model)

        model.promptText = "second"
        let request = try model.makeRequest()
        #expect(request.messages.map(\.content) == ["first", "answer", "second"])
        #expect(request.messages.map(\.role) == [.user, .assistant, .user])

        model.run()
        await waitForIdle(model)

        #expect(model.committedTurns.map(\.content)
            == ["first", "answer", "second", "answer"])
        #expect(model.activeConversation?.title == "first")
    }

    @MainActor
    @Test func overflowingConversationRefusesToSendAndExplains() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.maxContextTokens = 4_096
        model.promptText = String(repeating: "x", count: 4_096 * 4)

        #expect(model.isConversationOverflowing)
        #expect(model.canRun)
        model.run()

        #expect(!model.isRunning)
        #expect(model.isContextOverflowNoticeVisible)
        #expect(model.committedTurns.isEmpty)

        model.promptText = "short"
        #expect(!model.isConversationOverflowing)
        model.run()
        await waitForIdle(model)
        #expect(!model.isContextOverflowNoticeVisible)
        #expect(model.committedTurns.map(\.role) == [.user, .assistant])
        #expect(model.committedTurns.first?.content == "short")
    }

    @MainActor
    @Test func staleReadySessionDisablesGenerationUntilReload() throws {
        let client = MockLifecycleInferenceClient()
        let directory = try makeCompleteModelInstall("stale-runtime")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: client)
        model.promptText = "go"
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(model.canRun)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
        #expect(!model.canRun)
        #expect(model.canReloadModel)
        #expect(client.ensureLoadedCallCount() == 0)
    }

    @MainActor
    @Test func cancelAfterPartialOutputCanBeCleared() async throws {
        let client = MockInferenceClient(response: "one two three four five", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "stop after token"
        model.maxNewTokensOverride = 10
        model.run()

        await waitUntil("the run emits its first token") { model.liveTokenCount > 0 }

        #expect(model.liveTokenCount > 0)
        model.cancel()
        #expect(model.isCancellationPending)
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(!model.isCancellationPending)
        #expect(model.error == .cancelled)
        #expect(model.hasOutputTranscript)
        #expect(!model.outputResponsePlainText.isEmpty)
        #expect(model.viewedConversationPlainText.hasPrefix(
            "You:\nstop after token\n\nAnswer:\n"))

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
        #expect(model.outputPromptText.isEmpty)
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.viewedConversationPlainText.isEmpty)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func cancelDuringPrefillKeepsPromptSnapshotUntilClear() async throws {
        let client = MockInferenceClient(response: "unused", tokenDelayNanos: 1_000_000)
        client.prefillSteps = 20
        let model = readyModel(client: client)
        model.promptText = "prefill prompt"
        model.run()

        await waitUntil("prefill reports progress") { model.livePrefillDone > 0 }

        #expect(model.outputPromptText == "prefill prompt")
        model.cancel()
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.outputPromptText == "prefill prompt")
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.viewedConversationPlainText == "You:\nprefill prompt")
        #expect(model.hasOutputTranscript)

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
    }

    @MainActor
    @Test func failedEventThenThrownErrorKeepsFirstTerminalState() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "fail"

        model.run()
        await waitForIdle(model)

        #expect(model.error?.userMessage == "synthetic failure")
        #expect(model.diagnostics?.stopReason == .failed)
    }

    @MainActor
    @Test func changingModelPathInvalidatesLoadedStateAndDiagnostics() {
        let model = AppModel(client: MockInferenceClient())
        let oldURL = FileManager.default.temporaryDirectory.appendingPathComponent("old.gturbo")
        let newURL = FileManager.default.temporaryDirectory.appendingPathComponent("new.gturbo")
        model.modelPathText = oldURL.path
        model.loadState = .ready(modelDirectory: oldURL, loadSeconds: 1)
        model.diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 1,
            tokensPerSecond: 1,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        model.setGlobalError(.unknown("old error"))

        model.setModelURL(newURL)

        #expect(model.modelPathText == newURL.standardizedFileURL.path)
        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
        #expect(model.diagnostics == nil)
        #expect(model.error == nil)
        #expect(model.presentation.label == "Model required")
        #expect(!model.canRun)
    }

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        return model
    }

    @MainActor
    private func waitForIdle(_ model: AppModel,
                             sourceLocation: SourceLocation = #_sourceLocation) async {
        await waitUntilIdle(model, sourceLocation: sourceLocation)
    }
}
