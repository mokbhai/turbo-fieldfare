import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelLoadStateTests {
    @MainActor
    @Test func runRequiresReadyState() {
        let model = AppModel()
        model.promptText = "go"

        model.loadState = .notLoaded
        #expect(!model.canRun)
        model.loadState = .loading(.verifyingWeights)
        #expect(!model.canRun)
        model.loadState = .failed(.modelLoadFailed("boom"))
        #expect(!model.canRun)
        model.loadState = .ready(modelDirectory: URL(fileURLWithPath: "/tmp/m.gturbo"), loadSeconds: 1.2)
        #expect(model.canRun)
    }

    @MainActor
    @Test func canLoadModelOnlyFromNotLoadedOrFailed() throws {
        let directory = try makeCompleteModelInstall("can-load")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)
        model.loadState = .notLoaded
        #expect(model.canLoadModel)
        model.loadState = .failed(.modelLoadFailed("boom"))
        #expect(model.canLoadModel)
        model.loadState = .loading(.tokenizer)
        #expect(!model.canLoadModel)
        model.loadState = .ready(modelDirectory: URL(fileURLWithPath: "/tmp/m.gturbo"), loadSeconds: 1.2)
        #expect(!model.canLoadModel)
    }

    @MainActor
    @Test func loadingStateBlocksRun() {
        let model = AppModel()
        model.promptText = "go"
        model.loadState = .loading(.preparingRunner)
        #expect(!model.canRun)
        #expect(model.loadState.isLoading)
    }

    @MainActor
    @Test func explicitLoadReloadAndUnloadActionsAreMutuallyExclusive() throws {
        let directory = try makeCompleteModelInstall("action-matrix")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())
        #expect(model.canLoadModel)
        #expect(!model.canReloadModel)
        #expect(!model.canUnloadModel)

        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        #expect(!model.canLoadModel)
        #expect(model.canUnloadModel)

        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.canReloadModel)
        #expect(model.canUnloadModel)
        #expect(!model.canRun)
    }

    @MainActor
    @Test func unloadWaitsForLifecycleAndPreservesTranscript() async throws {
        let directory = try makeCompleteModelInstall("unload")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        client.suspendUnloads = true
        let model = AppModel(modelDirectory: directory, client: client)
        model.outputText = "keep me"
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.unloadModel()
        await client.waitForUnloadStart()
        #expect(model.loadState == .unloading)
        client.releaseUnloads()
        await waitUntil("the unload settles back to notLoaded") {
            model.loadState == .notLoaded
        }
        #expect(model.loadState == .notLoaded)
        #expect(model.outputText == "keep me")
    }

    @MainActor
    @Test func cancelLoadWaitsForUnloadAndRejectsLateReady() async throws {
        let directory = try makeCompleteModelInstall("cancel-load")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        client.suspendLoads = true
        client.suspendUnloads = true
        let model = AppModel(modelDirectory: directory, client: client)

        model.loadModel()
        await client.waitForLoadStart()
        await waitUntil("the in-flight load becomes cancellable") { model.canCancelLoad }
        model.cancelLoad()
        await client.waitForUnloadStart()
        #expect(model.loadState == .cancelling)

        client.emitLoadState(.ready(modelDirectory: directory, loadSeconds: 0), callIndex: 0)
        await Task.yield()
        #expect(model.loadState == .cancelling)

        client.releaseUnloads()
        await waitUntil("the cancelled load settles back to notLoaded") {
            model.loadState == .notLoaded
        }
        #expect(model.loadState == .notLoaded)
    }

    @MainActor
    @Test func failedLoadIsRetryableAndOldAttemptCannotCompleteRetry() async throws {
        let directory = try makeCompleteModelInstall("retry-load")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        client.suspendLoads = true
        let model = AppModel(modelDirectory: directory, client: client)

        model.loadModel()
        await client.waitForLoadStart()
        client.failNextLoad(.modelLoadFailed("synthetic"))
        await waitUntil("the load reports its failure") { model.loadState.isFailed }
        #expect(model.loadState == .failed(.modelLoadFailed("synthetic")))
        #expect(model.canLoadModel)

        client.suspendLoads = true
        model.loadModel()
        await client.waitForLoadStart(2)
        client.emitLoadState(.ready(modelDirectory: directory, loadSeconds: 0), callIndex: 0)
        await Task.yield()
        #expect(model.loadState.isLoading)

        client.releaseLoads()
        await waitUntil("the retried load becomes ready") { model.loadState.isReady }
        #expect(model.loadState.isReady)
        #expect(client.ensureLoadedCallCount() == 2)
    }

    @MainActor
    @Test func loadModelWithoutLifecycleClientFails() throws {
        let directory = try makeCompleteModelInstall("no-lifecycle")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: MockInferenceClient())
        model.loadModel()
        #expect(model.loadState.isFailed)
    }

    @MainActor
    @Test func applyLoadStateUpdatesPresentation() throws {
        let directory = try makeCompleteModelInstall("load-presentation")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)
        model.applyLoadState(.loading(.verifyingWeights))
        #expect(model.presentation.label == AppModelLoadPhase.verifyingWeights.label)
        model.applyLoadState(.failed(.modelLoadFailed("boom")))
        #expect(model.presentation.detail == "Model load failed: boom")
    }

    @MainActor
    @Test func loadModelWaitsForPendingUnloadAfterModelPathChange() async throws {
        let client = MockLifecycleInferenceClient()
        client.suspendUnloads = true
        let oldURL = try makeCompleteModelInstall("old-load")
        let newURL = try makeCompleteModelInstall("new-load")
        defer {
            try? FileManager.default.removeItem(at: oldURL)
            try? FileManager.default.removeItem(at: newURL)
        }
        let model = AppModel(modelDirectory: oldURL, client: client)
        model.loadState = .ready(modelDirectory: oldURL, loadSeconds: 1)

        model.setModelURL(newURL)
        await client.waitForUnloadStart()
        model.loadModel()

        #expect(client.ensureLoadedCallCount() == 0)
        client.releaseUnloads()
        await client.waitForEnsureLoadedCallCount(1)
        #expect(client.ensureLoadedCallCount() == 1)
    }

    @MainActor
    @Test func staleReadyStateForOldModelPathIsIgnored() {
        let model = AppModel(client: MockInferenceClient())
        let oldURL = URL(fileURLWithPath: "/tmp/old.gturbo")
        let newURL = URL(fileURLWithPath: "/tmp/new.gturbo")
        model.modelPathText = oldURL.path

        model.setModelURL(newURL)
        model.applyLoadState(.ready(modelDirectory: oldURL, loadSeconds: 0))

        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
    }
}
