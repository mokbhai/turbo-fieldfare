import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// Switching conversations while an answer streams. The property every test
/// here defends: a generated answer is never lost and never appended to a
/// conversation other than the one that asked for it.
@Suite struct AppModelConversationSwitchTests {
    @MainActor
    @Test func switchingConversationsMidRunKeepsTheRunAlive() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await startRun(model, prompt: "a question")
        #expect(model.liveTokenCount > 0)
        let tokensAtSwitch = model.liveTokenCount

        model.newConversation()
        let chatB = try #require(model.activeConversationID)

        #expect(chatB != chatA)
        #expect(model.isRunning)
        #expect(!model.isCancellationPending)
        #expect(model.generatingConversationID == chatA)
        #expect(model.isGenerating(chatA))
        #expect(!model.isGenerating(chatB))
        #expect(!model.isRunningInActiveConversation)

        for _ in 0..<400 where model.liveTokenCount <= tokensAtSwitch {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(model.liveTokenCount > tokensAtSwitch)

        model.cancel()
        await waitForIdle(model)
    }

    /// The regression test for the misfiling: the commit target used to be
    /// whichever conversation was on screen when generation ended.
    @MainActor
    @Test func answerCommitsToTheConversationItStartedIn() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await startRun(model, prompt: "a question")
        #expect(model.liveTokenCount > 0)

        model.newConversation()
        let chatB = try #require(model.activeConversationID)
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.error == nil)

        let origin = try #require(model.conversations.first { $0.id == chatA })
        #expect(origin.turns.count == 2)
        #expect(origin.turns.first?.role == .user)
        #expect(origin.turns.first?.content == "a question")
        #expect(origin.turns.last?.role == .assistant)
        #expect(origin.turns.last?.content.hasPrefix(Self.streamedResponse) == true)
        #expect(origin.lastPromptTokenCount != nil)
        #expect(origin.title == "a question")

        let viewed = try #require(model.conversations.first { $0.id == chatB })
        #expect(viewed.turns.isEmpty)
        #expect(viewed.lastPromptTokenCount == nil)
        #expect(viewed.title == Conversation.untitled)
        #expect(model.activeConversationID == chatB)
    }

    @MainActor
    @Test func switchingMidRunDoesNotWipeTheLiveTurn() async throws {
        let model = readyModel(client: streamingClient())
        await startRun(model, prompt: "a question")
        #expect(model.liveTokenCount > 0)
        let promptBefore = model.outputPromptText
        let responseBefore = model.outputResponsePlainText
        #expect(!responseBefore.isEmpty)

        model.newConversation()

        #expect(model.outputPromptText == promptBefore)
        #expect(model.outputResponsePlainText.hasPrefix(responseBefore))

        await waitForIdle(model)

        // The committed answer starts at the first token, not at the switch.
        let answer = try #require(model.conversations.first { $0.turns.count == 2 }?.turns.last)
        #expect(answer.content.hasPrefix(responseBefore))
        #expect(answer.content.hasPrefix(Self.streamedResponse))
    }

    /// `viewedTranscriptMailbox` is not asserted here: `MockInferenceClient` is
    /// not an `AppInferenceTranscriptReporting` client, so the mailbox is nil
    /// whatever the model does and the assertion could not fail.
    /// `AppModelServiceReportingTests.aRunsTranscriptIsInvisibleFromAnotherConversation`
    /// covers it with a client that has a mailbox, checking both polarities.
    @MainActor
    @Test func liveTurnIsNotVisibleFromAnotherConversation() async throws {
        let model = readyModel(client: streamingClient())
        await startRun(model, prompt: "a question")
        #expect(model.liveTokenCount > 0)

        model.newConversation()

        #expect(!model.hasOutputTranscript)
        #expect(model.viewedOutputPromptText.isEmpty)
        #expect(model.viewedOutputText.isEmpty)
        #expect(model.viewedResponsePlainText.isEmpty)
        #expect(model.viewedConversationPlainText.isEmpty)
        #expect(model.isPromptExamplesCardVisible)
        // The run's own copy is untouched; it is only invisible from here.
        #expect(model.outputPromptText == "a question")

        model.cancel()
        await waitForIdle(model)
    }

    @MainActor
    @Test func newConversationDuringARunDoesNotReuseTheGeneratingChat() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        #expect(model.conversations.count == 1)
        await startRun(model, prompt: "a question")
        #expect(model.liveTokenCount > 0)

        // The generating chat has no committed turns, so the empty-reuse
        // shortcut would otherwise hand it straight back.
        model.newConversation()
        let chatB = try #require(model.activeConversationID)
        #expect(model.conversations.count == 2)
        #expect(chatB != chatA)
        #expect(model.liveTurnConversationID == chatA)
        #expect(model.outputPromptText == "a question")

        // A second new chat from the untouched, non-generating one still reuses.
        model.newConversation()
        #expect(model.conversations.count == 2)
        #expect(model.activeConversationID == chatB)
        #expect(model.outputPromptText == "a question")

        model.cancel()
        await waitForIdle(model)
    }

    @MainActor
    @Test func deletingTheGeneratingConversationCancelsTheRun() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await startRun(model, prompt: "a question")
        #expect(model.liveTokenCount > 0)

        model.deleteConversation(chatA)

        #expect(!model.conversations.contains { $0.id == chatA })
        #expect(model.isCancellationPending)
        await waitForIdle(model)

        // Including the replacement conversation synthesised when the list
        // emptied: an orphaned turn must not land anywhere.
        #expect(!model.conversations.isEmpty)
        #expect(model.conversations.flatMap(\.turns).isEmpty)
        #expect(model.liveTurnConversationID == nil)
        #expect(model.outputPromptText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(!model.hasOutputTranscript)
        // Nothing to say and nobody to say it to: the chat that would have shown
        // the "Cancelled" banner is the one the user just deleted, and the
        // prompt went with it rather than into the replacement chat's draft.
        #expect(model.error == nil)
        #expect(model.promptText.isEmpty)
        #expect(model.draftedConversationCount == 0)
    }

    @MainActor
    @Test func deletingAnotherConversationLeavesTheRunTargetIntact() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await seedFinishedExchange(model, prompt: "seed a")
        model.newConversation()
        let chatB = try #require(model.activeConversationID)
        await seedFinishedExchange(model, prompt: "seed b")
        model.newConversation()
        let chatC = try #require(model.activeConversationID)
        #expect(Set([chatA, chatB, chatC]).count == 3)

        model.selectConversation(chatA)
        await startRun(model, prompt: "the real question")
        #expect(model.liveTokenCount > 0)

        // Removing a row shifts every index after it; the commit target is an
        // ID for exactly this reason.
        model.deleteConversation(chatC)
        #expect(model.isRunning)
        #expect(model.liveTurnConversationID == chatA)
        await waitForIdle(model)

        let origin = try #require(model.conversations.first { $0.id == chatA })
        #expect(origin.turns.map(\.role) == [.user, .assistant, .user, .assistant])
        #expect(origin.turns[2].content == "the real question")
        #expect(origin.turns[3].content.hasPrefix(Self.streamedResponse))

        let untouched = try #require(model.conversations.first { $0.id == chatB })
        #expect(untouched.turns.count == 2)
        #expect(!model.conversations.contains { $0.id == chatC })
    }

    /// A stopped run hands its prompt back to the draft of the chat that sent
    /// it, not to the composer in front of the user. B's draft is empty
    /// throughout, so nothing but the ownership of the live turn can keep the
    /// cancelled prompt out of it.
    @MainActor
    @Test func aCancelledRunsPromptGoesBackToItsOwnChatsDraft() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await startRun(model, prompt: "cancel me")
        #expect(model.liveTokenCount > 0)

        model.newConversation()
        let chatB = try #require(model.activeConversationID)
        #expect(model.promptText.isEmpty)
        model.cancel()
        await waitForIdle(model)

        #expect(model.activeConversationID == chatB)
        // Nothing arrives in the chat the user is reading: not the prompt, not
        // the partial answer, not the banner.
        #expect(model.promptText.isEmpty)
        #expect(!model.canRun)
        #expect(!model.hasOutputTranscript)
        #expect(model.error == nil)
        #expect(model.conversations.first { $0.id == chatA }?.turns.isEmpty == true)

        // All three are waiting in the chat that ran.
        model.selectConversation(chatA)
        #expect(model.promptText == "cancel me")
        #expect(model.error == .cancelled)
        #expect(model.hasOutputTranscript)
        #expect(model.viewedOutputPromptText == "cancel me")
        #expect(!model.viewedResponsePlainText.isEmpty)
    }

    /// The same rule seen from the chat that ran: the prompt is in this
    /// conversation's draft the moment the run stops, because this composer is
    /// that chat's composer.
    @MainActor
    @Test func cancellingInTheViewedChatRestoresThePromptToTheComposer() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await startRun(model, prompt: "cancel me")
        #expect(model.liveTokenCount > 0)
        #expect(model.promptText.isEmpty)

        model.cancel()
        await waitForIdle(model)

        #expect(model.error == .cancelled)
        #expect(model.activeConversationID == chatA)
        #expect(model.promptText == "cancel me")
    }

    /// A run that fails while the user is reading another chat puts its prompt
    /// and its banner in its own conversation, so the retry is waiting where it
    /// was typed. The failure is injected directly because the mock decides to
    /// fail before the first switch could happen.
    @MainActor
    @Test func aFailedRunsPromptGoesBackToItsOwnChatsDraft() async throws {
        let client = streamingClient()
        let model = readyModel(client: client)
        let chatA = try #require(model.activeConversationID)
        await startRun(model, prompt: "retry me")
        #expect(model.liveTokenCount > 0)

        model.newConversation()
        let chatB = try #require(model.activeConversationID)
        let failure = AppInferenceError.unknown("decode service died")
        model.apply(.failed(failure, partial: nil))
        client.cancel()

        #expect(!model.isRunning)
        #expect(model.activeConversationID == chatB)
        #expect(model.promptText.isEmpty)
        #expect(model.error == nil)

        model.selectConversation(chatA)
        #expect(model.promptText == "retry me")
        #expect(model.error == failure)
        #expect(model.conversations.first { $0.id == chatA }?.turns.isEmpty == true)
    }

    /// A run that reaches its terminal event without producing a token is not an
    /// exchange. Committing it would leave the chat holding a question the model
    /// never answered, and every later request from that chat would carry two
    /// user messages in a row. The prompt is not thrown away: it goes back to
    /// that chat's own draft, armed for a retry.
    @MainActor
    @Test func aRunWithNoAnswerIsNeverCommittedToItsConversation() async throws {
        let client = streamingClient()
        let model = readyModel(client: client)
        let chatA = try #require(model.activeConversationID)
        await seedFinishedExchange(model, prompt: "seed a")

        // Long enough that the terminal event below lands before any token does.
        client.prefillSteps = 400
        model.promptText = "unanswered"
        model.run()
        #expect(model.isRunning)
        #expect(model.outputPromptText == "unanswered")

        model.apply(.finished(AppDiagnostics(
            generatedTokens: 0,
            stopReason: .eos,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())))
        client.cancel()

        #expect(!model.isRunning)
        #expect(model.error == nil)
        // The unanswered turn is gone from the transcript and from history.
        #expect(model.liveTurnConversationID == nil)
        #expect(model.outputPromptText.isEmpty)
        let origin = try #require(model.conversations.first { $0.id == chatA })
        #expect(origin.turns.map(\.role) == [.user, .assistant])
        #expect(origin.turns.first?.content == "seed a")

        // Nothing was lost, and the next request is still an alternating
        // conversation ending in exactly one user message.
        #expect(model.promptText == "unanswered")
        let request = try model.makeRequest()
        #expect(request.messages.map(\.role) == [.user, .assistant, .user])
        #expect(request.messages.map(\.content).last == "unanswered")
    }

    /// Deleting the chat being read lands the user on another one. That is a
    /// switch like any other: the turn waiting in the chat landed on becomes
    /// visible where it always was, beside the draft it was restored to.
    @MainActor
    @Test func deletingTheViewedChatShowsTheStrandedTurnOfTheChatLandedOn() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await startRun(model, prompt: "stranded question")
        #expect(model.liveTokenCount > 0)

        model.newConversation()
        let chatB = try #require(model.activeConversationID)
        model.cancel()
        await waitForIdle(model)
        #expect(model.promptText.isEmpty)

        // B is the only other conversation, so deleting it lands on A.
        model.deleteConversation(chatB)

        #expect(model.activeConversationID == chatA)
        #expect(model.promptText == "stranded question")
        #expect(model.viewedOutputPromptText == "stranded question")
        #expect(!model.viewedResponsePlainText.isEmpty)
    }

    /// The banner says "This chat has filled the … context". It describes the
    /// conversation being read and a send just refused in it, so it cannot
    /// survive a switch to a chat that has not filled anything.
    @MainActor
    @Test func theContextOverflowNoticeDoesNotFollowTheUserToAnotherChat() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await seedFinishedExchange(model, prompt: "seed a")
        model.newConversation()
        let chatB = try #require(model.activeConversationID)
        model.selectConversation(chatA)

        model.maxContextTokens = 4_096
        model.promptText = String(repeating: "x", count: 4_096 * 4)
        model.run()
        #expect(!model.isRunning)
        #expect(model.isContextOverflowNoticeVisible)

        model.selectConversation(chatB)
        #expect(!model.isContextOverflowNoticeVisible)

        // Cleared, not merely hidden while away: coming back does not re-raise
        // it, and the next refused send in A does.
        model.selectConversation(chatA)
        #expect(!model.isContextOverflowNoticeVisible)
        #expect(model.isConversationOverflowing)
        model.run()
        #expect(model.isContextOverflowNoticeVisible)
    }

    /// The counterpart decision: `diagnostics` is decode-service telemetry, so
    /// it follows the run and stays put across a switch. The HUD it feeds
    /// already reports a run owned by a conversation off screen.
    @MainActor
    @Test func lastRunDiagnosticsSurviveASwitchBecauseTheyDescribeTheRuntime() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await seedFinishedExchange(model, prompt: "seed a")
        model.newConversation()
        let chatB = try #require(model.activeConversationID)
        model.selectConversation(chatA)
        await seedFinishedExchange(model, prompt: "seed a again")
        let diagnostics = try #require(model.diagnostics)

        model.selectConversation(chatB)

        #expect(model.diagnostics == diagnostics)
    }

    @MainActor
    @Test func stopWorksAfterReturningToTheGeneratingChat() async throws {
        let model = readyModel(client: streamingClient())
        let chatA = try #require(model.activeConversationID)
        await startRun(model, prompt: "a question")
        #expect(model.liveTokenCount > 0)

        model.newConversation()
        #expect(!model.isRunningInActiveConversation)
        #expect(model.canCancel)

        model.selectConversation(chatA)
        #expect(model.isRunningInActiveConversation)
        #expect(model.canCancel)
        #expect(model.viewedOutputPromptText == "a question")
        #expect(!model.viewedResponsePlainText.isEmpty)

        model.cancel()
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.error == .cancelled)
        // A cancelled run commits nothing but leaves its turn on screen.
        #expect(model.conversations.first { $0.id == chatA }?.turns.isEmpty == true)
        #expect(model.hasOutputTranscript)
    }

    @MainActor
    @Test func theAnswerIsPersistedToTheOriginatingConversation() async throws {
        // `supportDirectory` strips three components off the model directory, so
        // the temp model path has to be nested at least that deep.
        let support = FileManager.default.temporaryDirectory
            .appendingPathComponent("switch-persistence-\(UUID().uuidString)", isDirectory: true)
        let modelDirectory = support
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("slug", isDirectory: true)
            .appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: support) }

        let model = AppModel(modelDirectory: modelDirectory,
                             client: streamingClient(),
                             conversationsPersistenceEnabled: true)
        model.applyLoadState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
        let chatA = try #require(model.activeConversationID)

        await startRun(model, prompt: "persist me")
        #expect(model.liveTokenCount > 0)
        model.newConversation()
        await waitForIdle(model)

        let store = ConversationFileStore.loadGlobal(inSupportDirectory: support)
        let persisted = try #require(store.conversations.first { $0.id == chatA })
        #expect(persisted.turns.map(\.role) == [.user, .assistant])
        #expect(persisted.turns.first?.content == "persist me")
        #expect(persisted.turns.last?.content.hasPrefix(Self.streamedResponse) == true)
    }

    // MARK: - Helpers

    /// Long enough to survive a conversation switch, short enough that
    /// `waitForIdle` still sees the run finish.
    private static let streamedResponse =
        "one two three four five six seven eight nine ten eleven twelve"

    private func streamingClient() -> MockInferenceClient {
        let client = MockInferenceClient(response: Self.streamedResponse,
                                         tokenDelayNanos: 20_000_000)
        // Decoding starts immediately, so a poll on `liveTokenCount` is not
        // waiting out a prefill that never produces one.
        client.prefillSteps = 0
        return client
    }

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory,
                                 loadSeconds: 1)
        return model
    }

    /// Sends `prompt` and returns once the first token has arrived. Callers
    /// assert `liveTokenCount > 0`: a poll that times out would otherwise leave
    /// them asserting a pre-decode state.
    @MainActor
    private func startRun(_ model: AppModel, prompt: String) async {
        model.promptText = prompt
        model.run()
        for _ in 0..<400 where model.liveTokenCount == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    /// Gives a conversation committed turns so it is no longer eligible for the
    /// empty-chat reuse shortcut.
    @MainActor
    private func seedFinishedExchange(_ model: AppModel, prompt: String) async {
        let previousLimit = model.maxNewTokensOverride
        model.maxNewTokensOverride = 1
        model.promptText = prompt
        model.run()
        await waitForIdle(model)
        model.maxNewTokensOverride = previousLimit
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<400 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}
