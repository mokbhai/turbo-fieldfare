import Foundation
import Testing
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

/// System prompts, the extra sampling controls, and the last-exchange actions
/// (regenerate, edit, export).
@Suite struct AppModelChatControlsTests {
    // MARK: - System prompt

    @MainActor
    @Test func systemPromptLeadsTheRequest() throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        model.setActiveSystemPrompt("Answer in French.")
        model.promptText = "hello"

        let request = try model.makeRequest()
        #expect(request.messages.map(\.role) == [.system, .user])
        #expect(request.messages.first?.content == "Answer in French.")
        #expect(request.latestUserContent == "hello")
    }

    @MainActor
    @Test func blankSystemPromptSendsNothing() throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        model.setActiveSystemPrompt("  \n ")
        model.promptText = "hello"

        #expect(!model.hasActiveSystemPrompt)
        #expect(model.activeConversation?.systemPrompt == nil)
        #expect(try model.makeRequest().messages.map(\.role) == [.user])
    }

    @MainActor
    @Test func systemPromptBelongsToItsOwnConversation() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "hello")
        let first = try #require(model.activeConversationID)
        model.setActiveSystemPrompt("Be terse.")

        model.newConversation()
        #expect(!model.hasActiveSystemPrompt)

        model.selectConversation(first)
        #expect(model.activeSystemPrompt == "Be terse.")
    }

    @MainActor
    @Test func defaultSystemPromptSeedsOnlyNewConversations() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "hello")
        let existing = try #require(model.activeConversationID)

        model.setDefaultSystemPrompt("You are a pirate.")
        #expect(!model.hasActiveSystemPrompt)

        model.newConversation()
        #expect(model.activeSystemPrompt == "You are a pirate.")

        model.selectConversation(existing)
        #expect(!model.hasActiveSystemPrompt)
    }

    @MainActor
    @Test func changingTheSystemPromptDropsTheExactContextAnchor() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "hello")
        #expect(model.activeConversation?.lastPromptTokenCount != nil)

        model.setActiveSystemPrompt(String(repeating: "x", count: 350))

        #expect(model.activeConversation?.lastPromptTokenCount == nil)
        #expect(model.isContextUsageEstimated)
        #expect(model.contextUsedTokens >= 100)
    }

    @Test func conversationsWrittenBeforeSystemPromptsStillDecode() throws {
        let legacy = """
        {"id":"\(UUID().uuidString)","title":"Old","titleIsCustom":false,
         "createdAt":0,"updatedAt":0,"turns":[]}
        """
        let conversation = try JSONDecoder().decode(Conversation.self, from: Data(legacy.utf8))
        #expect(conversation.systemPrompt == nil)
        #expect(conversation.effectiveSystemPrompt == nil)
    }

    // MARK: - Sampling controls

    @MainActor
    @Test func repetitionPenaltyAndResponseLimitReachTheRequest() throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        model.maxNewTokensOverride = nil
        model.promptText = "hello"
        model.repetitionPenalty = 1.15
        model.maxResponseTokensEnabled = true
        model.maxResponseTokens = 512

        let request = try model.makeRequest()
        #expect(request.repetitionPenalty == Float(1.15))
        #expect(request.maxNewTokens == 512)

        model.maxResponseTokensEnabled = false
        #expect(try model.makeRequest().maxNewTokens == model.maxContextTokens)
    }

    /// A greedy run with a penalty still needs the logits head. If the model's
    /// idea of the loaded runtime disagreed with the request's, every such send
    /// would fail with "reload required" right after a reload.
    @MainActor
    @Test func repetitionPenaltyForcesTheLogitsHeadLikeTheRequestDoes() throws {
        let model = AppModel()
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.temperature = 0
        model.repetitionPenalty = 1.2
        model.promptText = "go"

        let request = try model.makeRequest()
        #expect(!request.isPureGreedy)
    }

    @MainActor
    @Test func presetsAndResetRestoreKnownValues() {
        let model = AppModel()
        model.applySamplingPreset(.creative)
        #expect(model.temperature == AppSamplingPreset.creative.temperature)
        #expect(model.activeSamplingPreset == .creative)

        model.topK = 12
        #expect(model.activeSamplingPreset == nil)

        model.repetitionPenalty = 1.3
        model.maxResponseTokensEnabled = true
        #expect(!model.generationParametersAreDefault)

        model.resetGenerationParameters()
        #expect(model.generationParametersAreDefault)
        #expect(model.activeSamplingPreset == .balanced)
    }

    // MARK: - Last exchange

    @MainActor
    @Test func regenerateReplacesTheLastAnswer() async throws {
        let client = MockInferenceClient(response: "first", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        await commitOneExchange(model, prompt: "hello")
        #expect(model.committedTurns.map(\.content) == ["hello", "first"])

        client.response = "second"
        model.promptText = "unsent draft"
        #expect(model.canRegenerateLastResponse)
        model.regenerateLastResponse()
        #expect(model.isRunning)
        #expect(model.promptText == "unsent draft")
        await waitUntilIdle(model)

        #expect(model.committedTurns.map(\.content) == ["hello", "second"])
        #expect(model.promptText == "unsent draft")
    }

    @MainActor
    @Test func failedRegenerateKeepsThePreviousAnswer() async throws {
        let client = MockInferenceClient(response: "first", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        await commitOneExchange(model, prompt: "hello")

        client.failureMessage = "boom"
        model.regenerateLastResponse()
        await waitUntilIdle(model)

        #expect(model.committedTurns.map(\.content) == ["hello", "first"])
        #expect(model.promptText.isEmpty)
        #expect(model.viewedOutputPromptText.isEmpty)
        #expect(model.error != nil)
        #expect(model.canRegenerateLastResponse)
    }

    @MainActor
    @Test func editLastPromptMovesItBackToTheComposer() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "one")
        await commitOneExchange(model, prompt: "two")

        #expect(model.canEditLastPrompt)
        model.editLastPrompt()

        #expect(model.committedTurns.map(\.content) == ["one", "answer"])
        #expect(model.promptText == "two")
        #expect(!model.canEditLastPrompt)
    }

    @MainActor
    @Test func editLastPromptNeverOverwritesADraft() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "one")
        model.promptText = "half-typed"

        #expect(!model.canEditLastPrompt)
        model.editLastPrompt()
        #expect(model.committedTurns.count == 2)
        #expect(model.promptText == "half-typed")
    }

    @MainActor
    @Test func markdownExportIncludesSystemPromptAndTurns() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        #expect(model.activeConversationMarkdown == nil)
        model.setActiveSystemPrompt("Be kind.")
        await commitOneExchange(model, prompt: "hello")

        let markdown = try #require(model.activeConversationMarkdown)
        #expect(markdown.hasPrefix("# hello\n"))
        #expect(markdown.contains("## System\n\nBe kind."))
        #expect(markdown.contains("## You\n\nhello"))
        #expect(markdown.contains("## Answer\n\nanswer"))
    }

    // MARK: - Helpers

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory,
                                 loadSeconds: 1)
        model.maxNewTokensOverride = 1
        return model
    }

    @MainActor
    private func commitOneExchange(_ model: AppModel, prompt: String) async {
        model.promptText = prompt
        model.run()
        await waitUntilIdle(model)
    }
}
