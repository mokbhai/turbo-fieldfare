import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelServiceReportingTests {
    @MainActor
    @Test func serviceMemoryAndCanonicalTranscriptOverrideUIProcessState() {
        let client = ReportingInferenceClient(memoryBytes: 2_100_000_000)
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        client.generationTranscriptMailbox.append("lossless output")

        // Only the two properties this test is about. The composed transcript
        // is deliberately not asserted here: a live turn with no owning
        // conversation stopped being reachable once `run()` began pinning one,
        // so composing one would assert on a state the app cannot produce.
        // `startingAnotherRunClearsPreviousServiceTranscriptSynchronously`
        // covers composition through a real run.
        #expect(model.currentProcessMemoryBytes == 2_100_000_000)
        #expect(model.outputResponsePlainText == "lossless output")

        model.clearOutput()
        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
    }

    @MainActor
    @Test func startingAnotherRunClearsPreviousServiceTranscriptSynchronously() {
        let client = ReportingInferenceClient(memoryBytes: 2_100_000_000)
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        client.generationTranscriptMailbox.append("previous completion")
        model.promptText = "new prompt"

        model.run()

        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
        #expect(model.outputPromptText == "new prompt")
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.viewedConversationPlainText == "You:\nnew prompt")
    }

    @MainActor
    @Test func aRunsTranscriptIsInvisibleFromAnotherConversation() {
        let client = ReportingInferenceClient(memoryBytes: 2_100_000_000)
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        model.promptText = "a question"
        model.run()
        client.generationTranscriptMailbox.append("streaming answer")

        #expect(model.isRunning)
        #expect(model.viewedTranscriptMailbox === client.generationTranscriptMailbox)

        model.newConversation()

        // The transcript view drains this mailbox on a timer of its own, outside
        // SwiftUI's observation graph, so a non-owning conversation must be
        // handed nil rather than merely told to ignore it.
        #expect(model.viewedTranscriptMailbox == nil)
        #expect(model.viewedResponsePlainText.isEmpty)
        #expect(model.viewedConversationPlainText.isEmpty)
        #expect(!model.hasOutputTranscript)
        // The run's own transcript is untouched by the switch.
        #expect(model.outputResponsePlainText == "streaming answer")
    }
}

private final class ReportingInferenceClient: AppInferenceClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, @unchecked Sendable {
    let currentInferenceMemoryBytes: UInt64?
    let generationTranscriptMailbox = GenerationTranscriptMailbox()

    init(memoryBytes: UInt64) {
        currentInferenceMemoryBytes = memoryBytes
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func cancel() {}
}
