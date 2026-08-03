import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// Drafts and run failures belong to one conversation each. A single global
/// composer let a prompt typed in one chat be sent from another, and a single
/// global error slot let a failure in one chat erase another chat's banner.
@Suite struct AppModelConversationScopeTests {
    @MainActor
    @Test func eachConversationKeepsItsOwnDraft() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "hello")
        let first = try #require(model.activeConversationID)
        model.promptText = "draft for the first chat"

        model.newConversation()
        let second = try #require(model.activeConversationID)
        #expect(second != first)
        #expect(model.promptText.isEmpty)

        model.promptText = "draft for the second chat"
        model.selectConversation(first)
        #expect(model.promptText == "draft for the first chat")

        model.selectConversation(second)
        #expect(model.promptText == "draft for the second chat")
    }

    @MainActor
    @Test func aDraftTypedInOneConversationCannotBeSentFromAnother() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "hello")
        let first = try #require(model.activeConversationID)
        model.promptText = "draft for the first chat"

        model.newConversation()

        // Nothing has been typed here, so there is nothing to send.
        #expect(!model.canRun)
        model.run()
        #expect(!model.isRunning)
        #expect(model.committedTurns.isEmpty)

        // And a send from here carries this conversation's text only.
        model.promptText = "second chat question"
        let request = try model.makeRequest()
        #expect(request.messages.map(\.content) == ["second chat question"])

        model.run()
        await waitForIdle(model)
        #expect(model.committedTurns.map(\.content) == ["second chat question", "answer"])
        #expect(!model.viewedConversationPlainText.contains("draft for the first chat"))

        model.selectConversation(first)
        #expect(model.promptText == "draft for the first chat")
    }

    @MainActor
    @Test func deletingAConversationDropsItsDraft() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "hello")
        let first = try #require(model.activeConversationID)
        model.promptText = "draft for the first chat"

        model.newConversation()
        model.promptText = "draft for the second chat"
        #expect(model.draftedConversationCount == 2)

        model.deleteConversation(first)

        // The deleted draft is unreachable, so it must not be held either.
        #expect(model.draftedConversationCount == 1)
        #expect(model.promptText == "draft for the second chat")
    }

    @MainActor
    @Test func failedRunRestoresThePromptToItsOwnConversationsDraft() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        await commitOneExchange(model, prompt: "hello")
        let first = try #require(model.activeConversationID)

        client.failureMessage = "synthetic failure"
        model.promptText = "retry me"
        model.run()
        await waitForIdle(model)
        #expect(model.promptText == "retry me")

        // The restore went to the conversation that ran, so a different chat's
        // composer is still empty and cannot send it.
        model.newConversation()
        #expect(model.promptText.isEmpty)
        #expect(!model.canRun)

        model.selectConversation(first)
        #expect(model.promptText == "retry me")
    }

    @MainActor
    @Test func failuresInDifferentConversationsDoNotEraseEachOther() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        await commitOneExchange(model, prompt: "hello")
        let first = try #require(model.activeConversationID)

        client.failureMessage = "failure in the first chat"
        model.promptText = "retry first"
        model.run()
        await waitForIdle(model)
        #expect(model.error?.userMessage == "failure in the first chat")

        model.newConversation()
        let second = try #require(model.activeConversationID)
        // A new chat starts clean: the other chat's failure is not its to show.
        #expect(model.error == nil)

        client.failureMessage = "failure in the second chat"
        model.promptText = "ask second"
        model.run()
        await waitForIdle(model)
        #expect(model.error?.userMessage == "failure in the second chat")

        model.selectConversation(first)
        #expect(model.error?.userMessage == "failure in the first chat")

        model.selectConversation(second)
        #expect(model.error?.userMessage == "failure in the second chat")
    }

    @MainActor
    @Test func aModelLoadFailureShowsInEveryConversation() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "hello")
        let first = try #require(model.activeConversationID)
        model.newConversation()
        let second = try #require(model.activeConversationID)

        model.applyLoadState(.failed(.modelLoadFailed("no weights")))

        #expect(model.error == .modelLoadFailed("no weights"))
        model.selectConversation(first)
        #expect(model.error == .modelLoadFailed("no weights"))
        model.selectConversation(second)
        #expect(model.error == .modelLoadFailed("no weights"))
    }

    @MainActor
    @Test func dismissingAModelFailureRevealsTheConversationsOwnFailure() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        await commitOneExchange(model, prompt: "hello")

        client.failureMessage = "failure in this chat"
        model.promptText = "retry"
        model.run()
        await waitForIdle(model)

        // A blocked model is the more urgent message, but it does not consume
        // the run failure underneath it.
        model.applyLoadState(.failed(.modelLoadFailed("no weights")))
        #expect(model.error == .modelLoadFailed("no weights"))

        model.dismissError()
        #expect(model.error?.userMessage == "failure in this chat")

        model.dismissError()
        #expect(model.error == nil)
    }

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory,
                                 loadSeconds: 1)
        model.maxNewTokensOverride = 1
        return model
    }

    /// Switching model directories reloads conversations from a different
    /// store, so drafts and per-conversation errors keyed by the old store's
    /// IDs would never be reachable or released again. Both cleanups are
    /// invisible from the outside, so without this they can be deleted with the
    /// whole suite still green.
    @MainActor
    @Test func switchingModelDirectoriesReleasesTheOldStoresDraftsAndErrors() async throws {
        let model = readyModel(client: MockInferenceClient(response: "answer", tokenDelayNanos: 1))
        await commitOneExchange(model, prompt: "hello")
        let stranded = try #require(model.activeConversationID)
        model.promptText = "draft that is about to be orphaned"
        model.setRunError(.unknown("failure in the old store"), for: stranded)
        #expect(model.draftedConversationCount == 1)
        #expect(model.error != nil)

        model.setModelURL(
            FileManager.default.temporaryDirectory
                .appendingPathComponent("a-different-model.gturbo"))

        #expect(model.draftedConversationCount == 0)
        #expect(model.promptText.isEmpty)
        #expect(model.error == nil)
        #expect(!model.conversations.contains { $0.id == stranded })
    }

    /// Commits one exchange so the conversation is no longer empty. Without it
    /// `newConversation()` reuses the untouched active chat instead of making a
    /// second one, and there would be nothing to scope drafts and errors to.
    @MainActor
    private func commitOneExchange(_ model: AppModel, prompt: String) async {
        model.promptText = prompt
        model.run()
        await waitForIdle(model)
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}
