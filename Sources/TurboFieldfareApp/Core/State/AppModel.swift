import Foundation
import TurboFieldfareDecodeProtocol
import TurboFieldfareRepackCore
import Observation

@MainActor
@Observable
public final class AppModel {
    public enum RunState: Equatable {
        case idle
        case running
    }

    public var modelPathText: String
    /// Composer text per conversation, held in memory only.
    ///
    /// Deliberately not a field on the persisted `Conversation`: that changes
    /// the stored schema, and a draft changes on every keystroke with no
    /// app-termination hook to flush on, so persisting it means either writing
    /// the whole store per character or losing the tail of what was typed. A
    /// draft therefore does not survive relaunch.
    ///
    /// An empty draft is stored as no entry at all, and a deleted conversation's
    /// entry is dropped, so this cannot grow past the conversations that exist.
    private var drafts: [UUID: String] = [:]
    /// The composer's text for the conversation on screen.
    ///
    /// One composer serving every chat was the bug: text typed in one
    /// conversation stayed in the box across a switch, so ⌘↩ delivered it into
    /// a different conversation. Reading and writing the active conversation's
    /// draft keeps every existing reader — `canRun`, `makeRequest`, the composer
    /// binding, the examples card — meaning "what the user typed *here*", and
    /// makes sending A's prompt from B impossible rather than merely unlikely.
    ///
    /// Cleared at exactly one place — the point in `run()` where the request is
    /// accepted — so a sent prompt does not linger as if it were still unsent, a
    /// refused send never destroys what the user typed, and a prompt typed while
    /// an answer streams survives the terminal event. The sent copy lives on in
    /// `outputPromptText` and then in the committed turns. The single path back
    /// is `restoreSubmittedPromptIfComposerIsEmpty()`, which a cancelled or
    /// failed run uses because it commits no turn.
    ///
    /// Because of that, an empty composer no longer implies "nothing has been
    /// sent yet" — anything wanting that older meaning must ask
    /// `isPromptExamplesCardVisible` instead.
    public var promptText: String {
        get { activeConversationID.flatMap { drafts[$0] } ?? "" }
        set {
            guard let activeConversationID else { return }
            drafts[activeConversationID] = newValue.isEmpty ? nil : newValue
        }
    }
    /// The in-flight exchange. Committed turns live in `activeConversation`;
    /// these two hold the turn being generated right now, so the transcript's
    /// committed prefix stays immutable while tokens stream in.
    public private(set) var outputPromptText: String = ""
    public var outputText: String = ""
    public private(set) var conversations: [Conversation] = []
    public private(set) var activeConversationID: UUID?
    /// Set when a send was refused because the conversation no longer leaves
    /// room for a reply. Distinct from `error`, which means a run actually failed.
    public private(set) var isContextOverflowNoticeVisible = false
    public var runState: RunState = .idle
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    public var maxContextTokens: Int = 4096
    public var temperature: Double = 0.2
    public var topKEnabled: Bool = true
    public var topK: Int = 64
    public var topPEnabled: Bool = true
    public var topP: Double = 0.95
    public var diagnostics: AppDiagnostics?
    /// A failure that blocks every conversation equally — a model load, an
    /// install, or the runtime itself. Set only through `setGlobalError`.
    private var globalError: AppInferenceError?
    /// Failures owned by the conversation whose run produced them, keyed by
    /// conversation. One slot for N chats meant a second failure erased the
    /// first chat's banner for good; a per-conversation slot cannot.
    private var conversationErrors: [UUID: AppInferenceError] = [:]

    /// The failure to show for the conversation on screen.
    ///
    /// The global one wins while it is present: it blocks this conversation too,
    /// so it is the message the user has to act on first. A conversation's own
    /// failure is not destroyed by it — it reappears once the global one is
    /// cleared or dismissed.
    ///
    /// Read-only on purpose. Scope is decided at the point a failure is raised,
    /// never inferred from the case, so callers use `setGlobalError` or
    /// `setRunError(_:for:)`; the UI dismisses through `dismissError()`.
    public var error: AppInferenceError? {
        globalError ?? activeConversationID.flatMap { conversationErrors[$0] }
    }
    /// Mirrored into `installStates` on every write so the picker's per-row
    /// progress stays in step with the single-model state the existing install
    /// UI reads, without duplicating the assignment at each transition.
    public var installState: AppModelInstallState = .idle {
        didSet { installStates.setState(installState, for: selectedRepoID) }
    }
    public private(set) var installETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var installETAText: String?
    public private(set) var installReadiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus

    /// Curated entries merged with whatever the user has added.
    public private(set) var catalog: ModelCatalog = ModelCatalog(custom: [])
    /// Install progress for every known model, so the picker can render each
    /// row independently of the single-model `installState` above.
    public private(set) var installStates = ModelInstallStates()
    /// Repository ID of the selected model, loaded or not.
    public private(set) var selectedRepoID: String = ModelCatalog.curated.first?.repoID ?? ""

    public var loadState: AppModelLoadState = .notLoaded
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey?
    public private(set) var phase: AppGenerationPhase = .idle
    public private(set) var liveTokenCount: Int = 0
    public private(set) var liveElapsedDecodeSeconds: Double = 0
    public private(set) var livePrefillDone: Int = 0
    public private(set) var livePrefillTotal: Int = 0
    public private(set) var liveMemoryBytes: UInt64?
    public private(set) var isCancellationPending: Bool = false

    private let client: any AppInferenceClient
    private let installer: any AppModelInstallerClient
    private var runTask: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var unloadTask: Task<Void, Never>?
    private var loadGeneration: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    /// Which `run()` a streamed event belongs to. Bumped by every `run()`, in
    /// the same shape as `loadGeneration`.
    ///
    /// `hasHandledTerminalEvent` is not enough on its own, and the gap is not
    /// theoretical. **A failing stream signals terminally twice**: once as a
    /// `.failed` event through `apply`, once as the thrown error through the
    /// `catch` around the `for try await` — and each is a separate hop onto
    /// this actor from the detached run task. The first already sets
    /// `runState = .idle`, so from the instant it lands the model is idle and
    /// the next `run()` may start; that `run()` clears
    /// `hasHandledTerminalEvent`, and the *previous* run's second hop, still
    /// queued behind it, is then handled as though it belonged to the new run
    /// — filing the old failure against the new conversation and tearing the
    /// new run down mid-flight. Whether the window is hit is decided entirely
    /// by how busy this actor's queue is, which is why it showed up as an
    /// intermittent conversation-scoping failure rather than as anything that
    /// named a run boundary.
    private var runGeneration: UInt64 = 0
    private var installGeneration: UInt64 = 0
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private var activeRunRuntimeKey: AppLoadedRuntimeKey?
    /// The conversation the live turn belongs to, pinned when the request is
    /// accepted. Switching chats moves the viewport and nothing else, so without
    /// this pin the answer would commit to whichever conversation happened to be
    /// on screen when generation ended, and a cancelled or failed run would give
    /// its prompt and its banner to a stranger.
    ///
    /// Deliberately outlives `runState`: a cancelled or failed run leaves its
    /// partial turn on screen, and that turn still belongs to one chat. It is
    /// cleared only when the turn is committed, discarded, or replaced by the
    /// next `run()`.
    private(set) var liveTurnConversationID: UUID?
    private var hasHandledTerminalEvent = false
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    private let conversationsPersistenceEnabled: Bool
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()

    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                settingsPersistenceEnabled: Bool = false,
                conversationsPersistenceEnabled: Bool = false,
                migratesLegacyInstall: Bool = false) {
        // Adopt a pre-multi-model install before probing, so an existing model
        // is found at the new slug path instead of appearing missing.
        //
        // Off by default and opted into only by the real app: this moves a
        // multi-gigabyte directory on the user's disk, and a test run that
        // constructs an AppModel must never do that.
        if migratesLegacyInstall, modelDirectory == nil,
           let curated = ModelCatalog.curated.first {
            AppModelLocation.migrateLegacyInstallIfNeeded(repoID: curated.repoID)
        }
        let directory = (modelDirectory ?? AppModelLocation.curatedDefaultURL()).standardizedFileURL
        let installETAClock = SuspendingClock()
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
            : MacAppSettings()
        self.modelPathText = directory.path
        self.runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled)
        self.maxContextTokens = settings.contextTokens
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = settings.topK
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.memorySampler = memorySampler
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.conversationsPersistenceEnabled = conversationsPersistenceEnabled
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        refreshInstallReadiness()
        loadCatalog()
        loadConversations()
    }

    /// Support directory holding `catalog.json` and `conversations.json`.
    /// Derived from the model directory's ancestry so a dev build writing into
    /// `scratch/models/<slug>/model.gturbo` keeps its state beside the repo
    /// rather than in Application Support.
    private var supportDirectory: URL {
        URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func loadCatalog() {
        let custom = conversationsPersistenceEnabled
            ? ModelCatalogStore.load(inSupportDirectory: supportDirectory).customEntries
            : []
        catalog = ModelCatalog(custom: custom)
        if case .complete = installationStatus {
            installStates.setState(
                .installed(modelDirectory: URL(fileURLWithPath: modelPathText, isDirectory: true)),
                for: selectedRepoID)
        }
    }

    /// How many conversations are holding a draft. Internal because no view
    /// wants it, but dropping a deleted conversation's draft is otherwise
    /// unobservable: that conversation can never be selected again.
    var draftedConversationCount: Int { drafts.count }

    public var isRunning: Bool { runState == .running }

    public var isModelAvailable: Bool { loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    public var canLoadModel: Bool {
        isModelInstalled && !isRunning && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady
    }

    public var isModelInstalled: Bool { installationStatus == .complete }

    public var requiresModelInstallation: Bool { !isModelInstalled }

    public var installDescriptor: AppModelInstallDescriptor { installer.descriptor }

    public var installRequirement: AppModelInstallRequirement? {
        installReadiness.requirement
    }

    public var isInstallingModel: Bool { installState.isInstalling }

    public var canInstallModel: Bool {
        guard case .ready = installReadiness else { return false }
        return !isRunning && !loadState.isLoading && !isInstallingModel
            && requiresModelInstallation
    }

    public var canCancelInstall: Bool { installState.canCancel }

    public var installDownloadedBytes: UInt64? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        return min(addition.overflow ? UInt64.max : addition.partialValue, total)
    }

    public var installTotalBytes: UInt64? {
        guard case .copyingPayload(_, _, let total) = installState else {
            return nil
        }
        return total
    }

    public var installReusedBytes: UInt64? {
        guard case .copyingPayload(let reused, _, _) = installState else {
            return nil
        }
        return reused
    }

    public var installDownloadedThisRunBytes: UInt64? {
        guard case .copyingPayload(_, let downloaded, _) = installState else {
            return nil
        }
        return downloaded
    }

    public var installProgressFraction: Double? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState,
              total > 0 else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var installPhaseLabel: String {
        switch installState {
        case .idle: return "Model required"
        case .checking: return "Checking installation"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning installation"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading model"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing installation"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding download"
        case .cancelled: return "Download paused"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Model installed"
        case .failed: return "Installation failed"
        }
    }

    public var canRun: Bool {
        !isRunning && isModelAvailable && !loadState.isLoading
            && !hasStaleLoadedRuntime
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var canCancel: Bool { isRunning && !isCancellationPending }

    /// The conversation generating right now, or nil when nothing is running.
    public var generatingConversationID: UUID? {
        isRunning ? liveTurnConversationID : nil
    }

    /// Non-optional on purpose: comparing `generatingConversationID` against an
    /// optional would report a match when both sides are nil, marking every row
    /// in the sidebar as generating while the app sits idle.
    public func isGenerating(_ id: UUID) -> Bool {
        generatingConversationID == id
    }

    /// True only when the run in flight belongs to the conversation on screen.
    /// The Stop pill, the composer's clear actions and the examples card key off
    /// this; the HUD and the model-lifecycle locks stay on the global
    /// `isRunning`, because they describe the machine rather than the document.
    public var isRunningInActiveConversation: Bool {
        isRunning && liveTurnConversationID == activeConversationID
    }

    /// Whether the live turn belongs to the conversation on screen. Unlike
    /// `isRunningInActiveConversation` this stays true after a cancelled or
    /// failed run, because that partial turn stays on screen for its own chat.
    private var isLiveTurnVisible: Bool {
        liveTurnConversationID != nil && liveTurnConversationID == activeConversationID
    }

    /// The live turn's conversation has been deleted, so the turn has nowhere to
    /// land. Reachable in normal use because deleting a generating conversation
    /// cancels the run and removes the row before the terminal event arrives.
    private var isLiveTurnOwnerMissing: Bool {
        guard let liveTurnConversationID else { return false }
        return !conversations.contains(where: { $0.id == liveTurnConversationID })
    }

    public var hasOutputTranscript: Bool {
        !committedTurns.isEmpty || !viewedOutputPromptText.isEmpty
            || !viewedOutputText.isEmpty
    }

    /// Whether the prompt-examples card belongs on screen.
    ///
    /// The invariant it encodes is "this conversation is genuinely fresh":
    /// nothing typed, nothing on screen from an earlier exchange, and no run in
    /// flight. Testing `promptText.isEmpty` alone used to be equivalent, because
    /// an empty composer meant nothing had been sent — but `run()` now clears
    /// the composer the instant a request is accepted, so that test would let
    /// the full-width card animate back in *while* the answer streams and resize
    /// the chrome under it.
    ///
    /// `hasOutputTranscript` covers both the committed turns and the live turn,
    /// which a cancelled or failed run deliberately leaves on screen — those
    /// states are not fresh, and `clearOutput()` or a new conversation is what
    /// makes them fresh again.
    ///
    /// The run clause is belt-and-braces, not a discriminator: `run()` assigns
    /// `outputPromptText` before it marks the run running, so
    /// `hasOutputTranscript` is already true for every instant of a run. It is
    /// kept because that equivalence rests on `latestUserContent` finding a user
    /// message, and `run()` falls back to `""` when it does not — under that
    /// fallback it is the only thing left holding the card off screen.
    ///
    /// Both clauses are scoped to the conversation on screen, and they have to
    /// move together. `hasOutputTranscript` ignores a live turn owned by another
    /// chat, so a global `isRunning` here would be the sole reason a genuinely
    /// fresh chat opened during a background run hid its card.
    public var isPromptExamplesCardVisible: Bool {
        promptText.isEmpty && !hasOutputTranscript && !isRunningInActiveConversation
    }

    public var outputResponsePlainText: String {
        generationTranscriptMailbox?.completeText ?? outputText
    }

    /// The live turn as the conversation on screen sees it: empty whenever the
    /// turn belongs to another chat, so a background run never bleeds into the
    /// transcript, the placeholder, or the copy actions of the chat being read.
    public var viewedOutputPromptText: String {
        isLiveTurnVisible ? outputPromptText : ""
    }

    public var viewedOutputText: String {
        isLiveTurnVisible ? outputText : ""
    }

    public var viewedResponsePlainText: String {
        isLiveTurnVisible ? outputResponsePlainText : ""
    }

    /// Nil unless the live turn is on screen. Load-bearing: the transcript view
    /// drains this mailbox on a repeating timer of its own, outside SwiftUI's
    /// observation graph, so handing a non-owning chat a non-nil mailbox would
    /// stream another conversation's tokens into it regardless of any other
    /// guard.
    public var viewedTranscriptMailbox: GenerationTranscriptMailbox? {
        isLiveTurnVisible ? generationTranscriptMailbox : nil
    }


    public var viewedConversationPlainText: String {
        conversationPlainText(prompt: viewedOutputPromptText,
                              response: viewedResponsePlainText)
    }

    private func conversationPlainText(prompt: String, response: String) -> String {
        var sections: [String] = []
        for turn in committedTurns {
            sections.append(turn.role == .user
                ? "You:\n\(turn.content)"
                : "Answer:\n\(turn.content)")
        }
        if !prompt.isEmpty { sections.append("You:\n\(prompt)") }
        if !response.isEmpty { sections.append("Answer:\n\(response)") }
        return sections.joined(separator: "\n\n")
    }

    // MARK: - Conversations

    public var activeConversation: Conversation? {
        guard let activeConversationID else { return nil }
        return conversations.first { $0.id == activeConversationID }
    }

    /// Turns already committed to the active conversation. Immutable, and the
    /// stable prefix the transcript renderer appends to.
    public var committedTurns: [ChatTurn] { activeConversation?.turns ?? [] }

    /// Newest first, so the sidebar shows recent chats at the top.
    public var conversationsByRecency: [Conversation] {
        conversations.sorted { $0.updatedAt > $1.updatedAt }
    }

    /// A reply needs room too: blocking exactly at the context limit would
    /// permit sends that can only produce a truncated answer, because
    /// `effectiveMaxNewTokens` clamps the response to whatever is left.
    public static let responseHeadroomTokens = 256

    /// Character-based, deliberately conservative. Real Gemma ratios run about
    /// 4 chars/token for prose and 3 for code, so 3.5 errs toward over-counting:
    /// blocking slightly early is recoverable by raising the context length,
    /// while under-counting fails mid-send inside the decode service.
    static func estimateTokens(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        return max(1, Int((Double(text.count) / 3.5).rounded(.up)))
    }

    /// Anchored on the exact prompt count the service reported for the previous
    /// generation, so only the turns added since then are estimated.
    public var estimatedNextPromptTokens: Int {
        let pending = Self.estimateTokens(
            promptText.trimmingCharacters(in: .whitespacesAndNewlines))
        guard let conversation = activeConversation else { return pending }
        guard let anchor = conversation.lastPromptTokenCount else {
            return conversation.turns.reduce(0) {
                $0 + Self.estimateTokens($1.content)
            } + pending
        }
        let sinceAnchor = conversation.turns
            .last(where: { $0.role == .assistant })
            .map { Self.estimateTokens($0.content) } ?? 0
        return anchor + sinceAnchor + pending
    }

    public var contextCapacityTokens: Int { maxContextTokens }

    /// What the meter shows: the conversation as it stands, excluding whatever
    /// is still being typed.
    public var contextUsedTokens: Int {
        guard let conversation = activeConversation else { return 0 }
        guard let anchor = conversation.lastPromptTokenCount else {
            return conversation.turns.reduce(0) {
                $0 + Self.estimateTokens($1.content)
            }
        }
        let sinceAnchor = conversation.turns
            .last(where: { $0.role == .assistant })
            .map { Self.estimateTokens($0.content) } ?? 0
        return anchor + sinceAnchor
    }

    /// True when the context count is a guess rather than a service-reported
    /// figure, so the UI can mark it.
    public var isContextUsageEstimated: Bool {
        activeConversation?.lastPromptTokenCount == nil
    }

    public var isConversationOverflowing: Bool {
        estimatedNextPromptTokens + Self.responseHeadroomTokens > maxContextTokens
    }

    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    public var presentation: AppPresentationState {
        AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: requiresModelInstallation,
            installState: installState,
            installReadiness: installReadiness,
            loadState: loadState,
            hasStaleRuntime: hasStaleLoadedRuntime,
            isRunning: isRunning,
            isGenerationCancellationPending: isCancellationPending,
            generationPhase: phase,
            livePrefillDone: livePrefillDone,
            livePrefillTotal: livePrefillTotal,
            lastStopReason: diagnostics?.stopReason))
    }

    public var currentProcessMemoryBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        if let reporter = client as? any AppInferenceMemoryReporting,
           let bytes = reporter.currentInferenceMemoryBytes {
            return bytes
        }
        return memorySampler.sample()
    }

    public var generationTranscriptMailbox: GenerationTranscriptMailbox? {
        (client as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
    }

    private var currentRuntimeKey: AppLoadedRuntimeKey {
        AppLoadedRuntimeKey(modelDirectory: URL(fileURLWithPath: modelPathText),
                            maxContextTokens: maxContextTokens,
                            options: runtimeOptions,
                            forceLogitsHead: currentForceLogitsHead)
    }

    private var currentForceLogitsHead: Bool {
        temperature != 0
    }

    public func setModelURL(_ url: URL) {
        guard !isRunning else { return }
        let path = url.standardizedFileURL.path
        guard path != modelPathText else { return }

        modelPathText = path
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
        loadConversations()
        resetLiveTurn()
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        installGeneration &+= 1
        installTask?.cancel()
        installer.cancel()
        installTask = nil
        resetInstallETA()
        installState = .idle
        pendingExplicitLoadRuntimeKey = nil
        activeRunRuntimeKey = nil
        loadedRuntimeKey = nil
        loadState = .notLoaded
        diagnostics = nil
        // Every failure on screen was about the model being replaced or was
        // produced by it, and the conversations themselves may have been
        // reloaded from a different store, so none of them survive.
        setGlobalError(nil)
        conversationErrors.removeAll()
        phase = .idle
        installationStatus = AppModelInstallationProbe.status(at: URL(fileURLWithPath: path))
        refreshInstallReadiness()

        if let lifecycle = client as? AppModelLifecycleClient {
            unloadGeneration &+= 1
            let generation = unloadGeneration
            let task = Task { [weak self, lifecycle] in
                await lifecycle.unload()
                self?.clearUnloadTask(generation: generation)
            }
            unloadTask = task
        }
    }

    public func loadModel() {
        guard canLoadModel else { return }
        beginLoad()
    }

    public func perform(_ action: AppModelAction) {
        switch action {
        case .install: installModel()
        case .cancelInstall: cancelInstall()
        case .load, .retryLoad: loadModel()
        case .cancelLoad: cancelLoad()
        case .reload: reloadModel()
        case .unload: unloadModel()
        }
    }

    public func reloadModel() {
        guard canReloadModel else { return }
        beginLoad()
    }

    private func beginLoad() {
        guard let lifecycle = client as? AppModelLifecycleClient else {
            loadState = .failed(.modelLoadFailed("This client has no model load lifecycle."))
            return
        }
        let directory = URL(fileURLWithPath: modelPathText)
        let maxContext = maxContextTokens
        let options = runtimeOptions
        let forceLogitsHead = currentForceLogitsHead
        let runtimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
                                             maxContextTokens: maxContext,
                                             options: options,
                                             forceLogitsHead: forceLogitsHead)
        let pendingUnload = unloadTask
        loadGeneration &+= 1
        let generation = loadGeneration
        pendingExplicitLoadRuntimeKey = runtimeKey
        // Only the model-scoped banner: a conversation's run failure is not
        // about this load and is not this load's to erase.
        setGlobalError(nil)
        loadState = .loading(.validatingDirectory)
        loadTask = Task.detached { [weak self, lifecycle, pendingUnload] in
            do {
                await pendingUnload?.value
                try Task.checkCancellation()
                try await lifecycle.ensureLoaded(modelDirectory: directory,
                                                 maxContextTokens: maxContext,
                                                 options: options,
                                                 forceLogitsHead: forceLogitsHead) { [weak self] state in
                    Task { @MainActor in
                        self?.applyLoadState(state, generation: generation)
                    }
                }
            } catch is CancellationError {
            } catch let appError as AppInferenceError {
                await self?.applyLoadState(.failed(appError), generation: generation)
            } catch {
                await self?.applyLoadState(
                    .failed(.modelLoadFailed("\(error)")),
                    generation: generation)
            }
            await self?.clearLoadTask(generation: generation)
        }
    }

    public func cancelLoad() {
        guard canCancelLoad, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .cancelling
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func installModel() {
        guard !isRunning, !loadState.isLoading, !isInstallingModel,
              requiresModelInstallation else {
            return
        }
        refreshInstallReadiness()
        guard canInstallModel else { return }
        installTask?.cancel()
        installer.cancel()
        resetInstallETA()
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .checking
        let entry = selectedEntry
            ?? ModelCatalogEntry(descriptor: installer.descriptor, trustTier: .curated)
        installTask = Task { [weak self, installer] in
            do {
                for try await event in installer.install(entry: entry,
                                                         outputDirectory: outputDirectory) {
                    guard let self else { return }
                    self.applyInstallEvent(event, generation: generation)
                }
                self?.finishInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishInstallCancellation(generation: generation)
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelInstall() {
        guard canCancelInstall else { return }
        installState = .cancelling
        installer.cancel()
    }

    // MARK: - Model selection

    public var selectedEntry: ModelCatalogEntry? {
        catalog.entry(forRepoID: selectedRepoID)
    }

    /// Descriptor for the model currently being installed or probed.
    ///
    /// Must not be `installer.descriptor`: that is the pinned curated model, so
    /// validating a custom install against it compares the finished download's
    /// snapshot hash to Gemma's fingerprint and rejects a perfectly good
    /// install after the user has already waited for the whole transfer.
    public var descriptorForSelection: AppModelInstallDescriptor {
        guard let selectedEntry else { return installer.descriptor }
        return AppModelInstallDescriptor(entry: selectedEntry)
    }

    /// Resident bytes of the selected model, measured from its manifest when
    /// installed. Drives the context picker's memory maths.
    public var residentWeightBytes: UInt64 {
        AppResidentWeightProbe.residentBytesOrEstimate(
            atModelDirectory: URL(fileURLWithPath: modelPathText, isDirectory: true))
    }

    public func switchVerdict(for entry: ModelCatalogEntry) -> ModelSwitchVerdict {
        ModelSwitchGuard.evaluate(
            target: entry,
            currentRepoID: selectedRepoID,
            loadState: loadState,
            isGenerating: isRunning,
            installStates: installStates)
    }

    /// Switches the loaded model.
    ///
    /// Prompt-cache correctness rides on `AppPromptCacheDomain`, which already
    /// keys on `modelID` and `sourceSnapshotHash`, so an entry cached under the
    /// previous model cannot match after the switch. The cache itself is
    /// private to `RealInferenceClient` and is not reachable from here.
    public func switchModel(to entry: ModelCatalogEntry) {
        switch switchVerdict(for: entry) {
        case .alreadyLoaded:
            return
        case .blockedByGeneration:
            // About the model picker, which is app-wide chrome.
            setGlobalError(.generationInFlight)
            return
        case .busy:
            return
        case .notInstalled:
            selectModel(entry)
            return
        case .allowed:
            break
        }
        selectModel(entry)
        loadModel()
    }

    /// Repoints state at a model without loading it, so the picker can select
    /// an uninstalled entry and then download it.
    private func selectModel(_ entry: ModelCatalogEntry) {
        guard let directory = try? AppModelLocation.defaultURL(forRepoID: entry.repoID) else {
            setGlobalError(.invalidRequest("Invalid repository ID: \(entry.repoID)"))
            return
        }
        if loadState.isReady { unloadModel() }
        selectedRepoID = entry.repoID
        modelPathText = directory.standardizedFileURL.path
        installationStatus = AppModelInstallationProbe.status(
            at: directory,
            descriptor: AppModelInstallDescriptor(entry: entry))
        if case .complete = installationStatus {
            installStates.setState(.installed(modelDirectory: directory), for: entry.repoID)
        }
        installState = installStates.state(for: entry.repoID)
        refreshInstallReadiness()
    }

    public func startInstall(for entry: ModelCatalogEntry) {
        selectModel(entry)
        installModel()
    }

    /// Removes installed weights but keeps the catalog entry, so the model can
    /// be re-downloaded without retyping the repository.
    public func deleteInstall(for entry: ModelCatalogEntry) {
        guard !(selectedRepoID == entry.repoID && loadState.isReady) else {
            setGlobalError(.invalidRequest("Unload \(entry.displayName) before deleting it."))
            return
        }
        guard let directory = try? AppModelLocation.defaultURL(forRepoID: entry.repoID) else {
            return
        }
        // Removes the slug directory, not just model.gturbo, so a partial
        // install's staging files go with it.
        // Only the weights directory. The slug directory also holds this
        // model's settings and the install lock, and removing it wholesale
        // destroyed unrelated state — and could fail partway on the held lock
        // while having already deleted the weights.
        do {
            try FileManager.default.removeItem(at: directory)
        } catch {
            // Never silent: losing weights is the most expensive thing this app
            // can do to a user, so a failure has to be visible.
            setGlobalError(.invalidRequest(
                "Could not delete \(entry.displayName): \(error.localizedDescription)"))
            return
        }
        installStates.setState(.idle, for: entry.repoID)
        if selectedRepoID == entry.repoID {
            installState = .idle
            installationStatus = AppModelInstallationProbe.status(
                at: directory,
                descriptor: AppModelInstallDescriptor(entry: entry))
            refreshInstallReadiness()
        }
    }

    /// Adds a user-supplied repository after the caller has run the
    /// architecture pre-flight and obtained consent.
    @discardableResult
    public func addCustomModel(_ entry: ModelCatalogEntry) -> Bool {
        guard let updated = try? catalog.addingCustom(entry) else { return false }
        catalog = updated
        if conversationsPersistenceEnabled {
            try? ModelCatalogStore.save(
                ModelCatalogFile(customEntries: updated.customEntries),
                inSupportDirectory: supportDirectory)
        }
        return true
    }

    public var hasPartialModelDownload: Bool {
        guard let paths = try? RemoteInstallPaths(outputDirectory: modelPathText) else {
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscardModelDownload: Bool {
        hasPartialModelDownload && !isInstallingModel && !isRunning
    }

    public func discardModelDownload() {
        guard canDiscardModelDownload else { return }
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .discarding
        installTask = Task { [weak self, installer] in
            do {
                try await installer.discardPartialInstall(
                    outputDirectory: outputDirectory)
                guard let self, generation == self.installGeneration else { return }
                self.installTask = nil
                self.installState = .idle
                self.refreshInstallReadiness()
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func refreshInstallReadiness() {
        refreshInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true).standardizedFileURL)
    }

    public func recheckModelAtCurrentLocation() {
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        modelPathText = directory.path
        refreshInstallReadiness(at: directory)
    }

    private func refreshInstallReadiness(at outputDirectory: URL) {
        installationStatus = AppModelInstallationProbe.status(
            at: outputDirectory,
            descriptor: descriptorForSelection)
        guard !isModelInstalled else { return }
        installReadiness = .checking
        do {
            let requirement = try installer.checkInstallRequirement(
                outputDirectory: outputDirectory)
            installReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            installReadiness = .failed("\(error)")
        }
    }

    private func applyInstallEvent(_ event: AppModelInstallEvent, generation: UInt64) {
        guard generation == installGeneration else { return }
        switch event {
        case .checking:
            resetInstallETA()
            installState = .checking
        case .downloadingMetadata:
            resetInstallETA()
            installState = .downloadingMetadata
        case .planning:
            resetInstallETA()
            installState = .planning
        case .reservingOutput:
            resetInstallETA()
            installState = .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            installState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
            updateInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            resetInstallETA()
            installState = .hashingOutput(file)
        case .finalizing:
            resetInstallETA()
            installState = .finalizing
        case .installed(let directory):
            resetInstallETA()
            let directory = directory.standardizedFileURL
            installationStatus = AppModelInstallationProbe.status(
                at: directory,
                descriptor: descriptorForSelection)
            guard installationStatus == .complete else {
                finishInstallFailure(
                    RepackError.configurationInvalid(detail: "completed install did not pass metadata validation"),
                    generation: generation)
                return
            }
            installState = .installed(modelDirectory: directory)
            installTask = nil
            modelPathText = directory.path
            loadState = .notLoaded
        }
    }

    private func finishInstallStream(generation: UInt64) {
        guard generation == installGeneration, installTask != nil else { return }
        if installState == .cancelling {
            finishInstallCancellation(generation: generation)
        } else if !isModelInstalled {
            finishInstallFailure(
                RepackError.configurationInvalid(detail: "installer ended before completion"),
                generation: generation)
        }
    }

    private func finishInstallCancellation(generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        installState = .cancelled
        resetInstallETA()
        refreshInstallReadiness()
    }

    private func updateInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let timestamp = installETATimestamp
        setInstallETAPresentation(
            installETAEstimator.update(observation, timestamp: timestamp))
    }

    private var installETATimestamp: Double {
        let components = installETAOrigin.duration(to: installETAClock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func resetInstallETA() {
        installETAEstimator.reset()
        installETAPresentation = .hidden
        installETAText = nil
    }

    private func setInstallETAPresentation(
        _ presentation: DownloadETAPresentation
    ) {
        installETAPresentation = presentation
        installETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func applyPersistedSettings(forModelDirectory modelDirectory: URL) {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled)
        maxContextTokens = settings.contextTokens
        temperature = settings.temperature
        topKEnabled = settings.topKEnabled
        topK = settings.topK
        topPEnabled = settings.topPEnabled
        topP = settings.topP
    }

    private func persistSettings() {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettings(
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
        try? MacAppSettingsFileStore.save(
            settings,
            forModelDirectory: modelDirectory)
    }

    private func finishInstallFailure(_ error: Error, generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        resetInstallETA()
        let hasSavedDownload = hasPartialModelDownload
        installState = hasSavedDownload ? .recoverable("\(error)") : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            let requirement = AppModelInstallRequirement(probePath: path,
                                                          requiredBytes: required,
                                                          availableBytes: available)
            installReadiness = .insufficientSpace(requirement)
        } else {
            refreshInstallReadiness()
            if hasSavedDownload {
                installState = .recoverable("\(error)")
            }
        }
    }

    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }

    private func applyLoadState(_ state: AppModelLoadState, generation: UInt64) {
        guard generation == loadGeneration else { return }
        if case .ready(let directory, _) = state,
           directory.standardizedFileURL.path
            != URL(fileURLWithPath: modelPathText).standardizedFileURL.path {
            return
        }
        loadState = state
        switch state {
        case .notLoaded:
            loadedRuntimeKey = nil
        case .loading, .cancelling, .unloading:
            break
        case .ready(_, let seconds):
            loadedRuntimeKey = pendingExplicitLoadRuntimeKey
                ?? activeRunRuntimeKey
                ?? currentRuntimeKey
            pendingExplicitLoadRuntimeKey = nil
            _ = seconds
        case .failed(let loadError):
            pendingExplicitLoadRuntimeKey = nil
            // A model that will not load blocks every conversation.
            setGlobalError(loadError)
        }
    }

    /// Kept for the existing "clear" affordances: starting a fresh chat is the
    /// least surprising reading of clearing the transcript now that history
    /// persists.
    public func clearOutput() {
        newConversation()
    }

    public func newConversation() {
        // A live turn is this action's to discard only when the chat being left
        // owns it and nothing is in flight. One owned by another chat, or by a
        // run still streaming, has to survive.
        let discardsLiveTurn = !isRunning
            && (liveTurnConversationID == nil
                || liveTurnConversationID == activeConversationID)
        // Never accumulate empty chats: reuse the active one if it is untouched.
        // A chat generating right now has no committed turns either, so
        // `isEmpty` alone would reuse the very conversation the answer is bound
        // for and hand it to a new exchange.
        if let active = activeConversation, active.isEmpty,
           generatingConversationID != active.id {
            if discardsLiveTurn { resetLiveTurn() }
            // Reuse means this is also the "Clear output" path, which has to
            // clear the banner of the conversation it is emptying. The fresh
            // branch below cannot need it: a new id has no failure yet.
            clearActiveConversationError()
            return
        }
        let conversation = Conversation()
        conversations.append(conversation)
        setActiveConversation(conversation.id)
        if discardsLiveTurn { resetLiveTurn() }
        persistConversations()
    }

    /// Switching conversations moves the viewport and nothing else. It must not
    /// touch the live turn: a run started elsewhere keeps streaming, and its
    /// prompt and partial answer have to be intact when the user comes back.
    public func selectConversation(_ id: UUID) {
        guard id != activeConversationID,
              conversations.contains(where: { $0.id == id }) else { return }
        setActiveConversation(id)
    }

    /// The single door onto `activeConversationID`. Every path that changes
    /// which conversation is on screen — selecting, starting a new chat,
    /// deleting the one being read, restoring at launch — goes through here, so
    /// arrival-time work cannot be implemented by one of them and forgotten by
    /// the rest.
    ///
    /// What belongs here is state describing the chat being *read*, as opposed
    /// to the run. `resetLiveTurn()` used to do this job by being called from
    /// every switch, which also destroyed the live turn; the turn now follows
    /// its owner, so only the viewer-scoped part is left to clear. Drafts and
    /// run failures need nothing here at all — they are already stored per
    /// conversation, so a switch simply reads a different entry.
    private func setActiveConversation(_ id: UUID?) {
        guard id != activeConversationID else { return }
        activeConversationID = id
        // "This chat has filled the … context" is a statement about the chat in
        // front of the user and about a send they just made in it. Carried
        // across a switch it accuses a conversation that may be empty. The next
        // send in the full chat raises it again, so clearing loses nothing.
        isContextOverflowNoticeVisible = false
    }

    public func deleteConversation(_ id: UUID) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else {
            return
        }
        // Cancel first, never refuse: leaving a chat undeletable for the length
        // of a 26B generation is worse than stopping a reply the user has just
        // asked to throw away. `cancel()` only requests cancellation, so the
        // terminal event lands after the row is gone — the terminal handler
        // detects the missing owner and discards the orphaned turn rather than
        // appending it to whichever conversation is active by then.
        if id == generatingConversationID { cancel() }
        let removesLiveTurnOwner = id == liveTurnConversationID
        conversations.remove(at: index)
        // The conversation is gone and can never be selected again, so its
        // draft and its failure go with it rather than sitting in memory.
        drafts[id] = nil
        conversationErrors[id] = nil
        if activeConversationID == id {
            setActiveConversation(conversationsByRecency.first?.id)
        }
        if conversations.isEmpty {
            let conversation = Conversation()
            conversations.append(conversation)
            setActiveConversation(conversation.id)
        }
        // A live turn goes with its conversation, but only once no run is
        // writing to it: while one is, `liveTurnConversationID` keeps pointing
        // at the departed UUID so the terminal handler can tell it is orphaned.
        // `clearLiveTurn`, not `resetLiveTurn`: the latter also drops
        // `diagnostics` and the context-overflow notice, and neither of those
        // belonged to the deleted conversation. Removing a chat must not blank
        // the HUD's account of the last run.
        if removesLiveTurnOwner && !isRunning { clearLiveTurn() }
        persistConversations()
    }

    public func renameConversation(_ id: UUID, to title: String) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            // Clearing a custom title hands naming back to the first user turn.
            conversations[index].titleIsCustom = false
            conversations[index].retitleIfNeeded()
        } else {
            conversations[index].title = trimmed
            conversations[index].titleIsCustom = true
        }
        persistConversations()
    }

    /// Throws away everything the chat being left had to show, live turn
    /// included. Only reachable from an explicit user action on the turn's own
    /// conversation — New Chat, Clear, or deleting the owner — and no longer
    /// from a plain switch, which must leave the run alone. What a switch still
    /// owes is in `setActiveConversation`.
    ///
    /// Says nothing about errors: a conversation's failure belongs to that
    /// conversation and outlives a switch, so the callers that really are
    /// clearing a conversation say so themselves.
    private func resetLiveTurn() {
        clearLiveTurn()
        diagnostics = nil
        isContextOverflowNoticeVisible = false
    }

    /// Drops the live turn and its ownership. Separate from `resetLiveTurn` so a
    /// terminal handler can dispose of an orphaned turn without also wiping the
    /// diagnostics that describe the run itself.
    private func clearLiveTurn() {
        outputPromptText = ""
        outputText = ""
        generationTranscriptMailbox?.reset()
        liveTurnConversationID = nil
    }

    public func dismissContextOverflowNotice() {
        isContextOverflowNoticeVisible = false
    }

    // MARK: - Errors

    /// Records a failure that blocks every conversation: a model load, an
    /// install, or the runtime. Pass `nil` to clear it.
    func setGlobalError(_ error: AppInferenceError?) {
        globalError = error
    }

    /// Records a failure produced by one conversation's run. Pass `nil` to clear
    /// that conversation's failure.
    func setRunError(_ error: AppInferenceError?, for conversationID: UUID?) {
        guard let conversationID else { return }
        conversationErrors[conversationID] = error
    }

    /// Clears the failure this conversation's own run produced. Not the global
    /// one: a model or install failure is not this conversation's to clear, and
    /// still applies once the transcript is gone.
    private func clearActiveConversationError() {
        setRunError(nil, for: activeConversationID)
    }

    /// Dismisses the failure currently on screen. When a conversation's own
    /// failure is hidden behind a global one, dismissing the global one reveals
    /// it — each dismissal dismisses the message it was aimed at.
    public func dismissError() {
        if globalError != nil {
            globalError = nil
            return
        }
        clearActiveConversationError()
    }

    /// Conversations live in one global store rather than inside a model
    /// directory: history has to survive a model switch, and deleting a model
    /// must not delete the chats made with it.
    private func loadConversations() {
        let store = conversationsPersistenceEnabled
            ? ConversationFileStore.loadGlobal(inSupportDirectory: supportDirectory)
            : ConversationStoreFile()
        conversations = store.conversations
        if conversations.isEmpty {
            conversations = [Conversation()]
        }
        // A reload replaces the whole set, and a model switch reads a different
        // store, so drafts keyed by conversations that are no longer here would
        // never be reachable or released again.
        let liveIDs = Set(conversations.map(\.id))
        drafts = drafts.filter { liveIDs.contains($0.key) }
        setActiveConversation(conversationsByRecency.first?.id)
    }

    private func persistConversations() {
        guard conversationsPersistenceEnabled else { return }
        let store = ConversationStoreFile(conversations: conversations)
        try? ConversationFileStore.saveGlobal(store, inSupportDirectory: supportDirectory)
    }

    /// Commits the finished exchange. Both turns land together so the committed
    /// list is never half-written, and only here does the store get touched —
    /// writing per token would thrash the disk at decode speed.
    private func commitLiveTurn() {
        let response = outputResponsePlainText
        guard !outputPromptText.isEmpty || !response.isEmpty else { return }
        // Resolved against the conversation the run started in, never the one on
        // screen and never a stored index: the user may have switched — or
        // deleted another chat, moving every index — while the answer streamed.
        guard let owner = liveTurnConversationID,
              let index = conversations.firstIndex(where: { $0.id == owner })
        else {
            // The owning conversation was deleted mid-run. Nothing can be
            // committed, and leaving the text in place would let it be appended
            // to whichever conversation commits next.
            clearLiveTurn()
            return
        }
        // An answer is what makes a turn history. A run that ended before its
        // first token leaves a prompt and nothing else, and committing that
        // would write a user message the model never replied to: the transcript
        // would show a question hanging in silence, and `makeRequest` would put
        // two user messages in a row into the next request — a shape the pinned
        // IT chat template renders but the checkpoint was never tuned on, and
        // one that compounds with every failure. Nothing is lost by refusing:
        // the prompt goes back to its own conversation's draft, armed for a
        // retry where the user typed it.
        guard !response.isEmpty else {
            restoreSubmittedPromptIfComposerIsEmpty()
            clearLiveTurn()
            return
        }
        let now = Date()
        if !outputPromptText.isEmpty {
            conversations[index].turns.append(
                ChatTurn(role: .user, content: outputPromptText, timestamp: now))
        }
        if !response.isEmpty {
            conversations[index].turns.append(
                ChatTurn(role: .assistant, content: response, timestamp: now))
        }
        conversations[index].lastPromptTokenCount = diagnostics?.promptTokenCount
        conversations[index].updatedAt = now
        conversations[index].retitleIfNeeded()
        clearLiveTurn()
        persistConversations()
    }

    public func run() {
        guard canRun else { return }
        // Refuse before the request crosses the IPC boundary: the decode service
        // would reject it anyway, and blocking here keeps an oversized prompt
        // from being framed at all.
        guard !isConversationOverflowing else {
            isContextOverflowNoticeVisible = true
            return
        }
        isContextOverflowNoticeVisible = false
        let request: AppGenerationRequest
        do {
            request = try makeRequest()
        } catch let appError as AppInferenceError {
            // A refused send belongs to the conversation it was sent from.
            setRunError(appError, for: activeConversationID)
            return
        } catch {
            setRunError(.unknown("\(error)"), for: activeConversationID)
            return
        }
        persistSettings()

        generationTranscriptMailbox?.reset()
        // Pinned here, beside the prompt snapshot: from this instant the live
        // turn belongs to this conversation, whatever the user switches to while
        // it streams.
        liveTurnConversationID = activeConversationID
        outputPromptText = request.latestUserContent ?? ""
        // The request is accepted from here on; see `promptText` for why the
        // composer is cleared here and nowhere else.
        promptText = ""
        outputText = ""
        diagnostics = nil
        // A fresh send clears the banner: this conversation's own last failure
        // is what is being retried, and a global one the user has since worked
        // past should not sit over the answer they just asked for.
        setRunError(nil, for: liveTurnConversationID)
        setGlobalError(nil)
        hasHandledTerminalEvent = false
        activeRunRuntimeKey = AppLoadedRuntimeKey(
            modelDirectory: request.modelDirectory,
            maxContextTokens: request.maxContextTokens,
            options: request.runtimeOptions,
            forceLogitsHead: !request.isPureGreedy)
        isCancellationPending = false
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        livePrefillDone = 0
        livePrefillTotal = 0
        liveMemoryBytes = nil
        phase = .prefill
        runState = .running

        // Everything this stream delivers is stamped with the run it came
        // from, so a hop that arrives after the next `run()` has started is
        // dropped instead of being applied to it. See `runGeneration`.
        runGeneration &+= 1
        let generation = runGeneration
        runTask = Task.detached { [weak self, client, request, generation] in
            guard let self else { return }
            do {
                for try await event in client.generate(request) {
                    await self.apply(event, generation: generation)
                }
            } catch let appError as AppInferenceError {
                await self.finishStreamFailure(appError, generation: generation)
            } catch {
                await self.finishStreamFailure(.unknown("\(error)"), generation: generation)
            }
        }
    }

    public func cancel() {
        guard canCancel else { return }
        isCancellationPending = true
        client.cancel()
    }

    public func makeRequest() throws -> AppGenerationRequest {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            messages: committedTurns.map(\.decodeMessage)
                + [DecodeChatMessage(role: .user, content: promptText)],
            maxNewTokens: maxNewTokensOverride ?? maxContextTokens,
            maxContextTokens: maxContextTokens,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            repetitionPenalty: 1.0,
            runtimeOptions: runtimeOptions)
        try request.validate(requireModelDirectory: true)
        return request
    }

    /// `apply`, but only if the event still belongs to the current run.
    ///
    /// Applied to every event, not only the terminal ones: a `.token` from a
    /// superseded run would append that run's text to the new run's output, and
    /// a `.prefillProgress` would drive the new run's HUD backwards. The
    /// terminal case is simply the one with a visible consequence.
    func apply(_ event: AppInferenceEvent, generation: UInt64) {
        guard generation == runGeneration else { return }
        apply(event)
    }

    func apply(_ event: AppInferenceEvent) {
        switch event {
        case .prefillProgress(let done, let total):
            phase = .prefill
            livePrefillDone = done
            livePrefillTotal = total
        case .token(let token):
            phase = .decode
            liveTokenCount = token.index + 1
            liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
            if let reporter = client as? any AppInferenceMemoryReporting {
                liveMemoryBytes = reporter.currentInferenceMemoryBytes
            } else {
                liveMemoryBytes = memorySampler.sample()
            }
            if !token.textDelta.isEmpty {
                outputText += token.textDelta
            }
        case .finished(let diagnostics):
            finishSuccessfully(diagnostics)
        case .cancelled(let diagnostics):
            finishCancelled(diagnostics)
        case .failed(let appError, let partial):
            diagnostics = partial
            materializeServiceTranscript()
            finishWithError(appError)
        }
    }

    private func finishSuccessfully(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        // Only a completed exchange is committed. A cancelled or failed run
        // stays in the live turn: it is still on screen, but it never becomes
        // history the model is asked to continue from, and the next run clears
        // it, so the transcript and the conversation never disagree.
        commitLiveTurn()
        finishTerminalRun()
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        settleUncommittedLiveTurn(.cancelled)
        finishTerminalRun()
    }

    /// The conversation a terminal event belongs to. Falls back to the one on
    /// screen only when no run was recorded, which happens when a caller drives
    /// `apply` directly rather than through `run()`.
    private var runOwnerConversationID: UUID? {
        liveTurnConversationID ?? activeConversationID
    }

    /// Disposes of a live turn that no terminal path committed.
    ///
    /// Normally that means recording the failure against the conversation that
    /// asked for the run and putting its prompt back into that conversation's
    /// own draft, so both are waiting where the user left them however far they
    /// have wandered since. Nothing is deferred and nothing is handed over on
    /// arrival: per-conversation drafts and errors make the delivery final at
    /// the moment the run ends.
    ///
    /// When the owning conversation was deleted mid-run there is nobody to tell
    /// and nowhere to put the turn. A banner keyed to a conversation that no
    /// longer exists could never be read or dismissed, and leaving the text in
    /// place would let the next commit inherit it, so the turn simply goes.
    private func settleUncommittedLiveTurn(_ appError: AppInferenceError) {
        guard !isLiveTurnOwnerMissing else {
            clearLiveTurn()
            return
        }
        setRunError(appError, for: runOwnerConversationID)
        restoreSubmittedPromptIfComposerIsEmpty()
    }

    /// Puts the submitted prompt back in the composer after a terminal path that
    /// committed nothing.
    ///
    /// Clearing the composer the moment a request is accepted is safe for a
    /// successful run, because the prompt has landed in the committed turns and
    /// the user can read it back there. A cancelled or failed run deliberately
    /// commits nothing, so the only surviving copy is the live turn — and the
    /// live turn is exactly what the composer's "Clear output" button offers to
    /// throw away. Without this, one failed generation would destroy what the
    /// user typed with no way back; with it, Generate is armed with the same
    /// text and retrying is a single click.
    ///
    /// The empty-composer guard is what stops this from fighting the user: if
    /// they began typing the next prompt while the answer streamed, that text is
    /// newer than the one being restored and must win. Emptiness is measured the
    /// same way `canRun` measures it — trimmed — because a composer holding only
    /// a stray space the user tapped mid-stream has nothing worth keeping, and
    /// treating it as occupied would discard the submitted prompt instead.
    ///
    /// Both the target and the text come from the live turn's owner. Reading the
    /// text from `outputPromptText` while resolving the target some other way
    /// would copy one chat's prompt into another chat's draft the moment the two
    /// disagreed — the very misdelivery the pin exists to prevent, arriving by
    /// the back door. With no live turn there is nothing to restore.
    private func restoreSubmittedPromptIfComposerIsEmpty() {
        guard let owner = liveTurnConversationID else { return }
        let draft = drafts[owner] ?? ""
        guard draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        drafts[owner] = outputPromptText.isEmpty ? nil : outputPromptText
    }

    private func materializeServiceTranscript() {
        guard let reporter = client as? any AppInferenceTranscriptReporting else { return }
        outputText = reporter.generationTranscriptMailbox.completeText
    }

    private func finishWithError(_ appError: AppInferenceError) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        // The run failed, not the app: the failure belongs to the conversation
        // that asked for it, and no other conversation's banner is touched.
        settleUncommittedLiveTurn(appError)
        finishTerminalRun()
    }

    /// The second of a failing stream's two terminal signals. Dropped when the
    /// run it belongs to has already been superseded — see `runGeneration`.
    func finishStreamFailure(_ appError: AppInferenceError, generation: UInt64) {
        guard generation == runGeneration else { return }
        finishStreamFailure(appError)
    }

    private func finishStreamFailure(_ appError: AppInferenceError) {
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishTerminalRun() {
        phase = .idle
        runState = .idle
        isCancellationPending = false
        activeRunRuntimeKey = nil
        // `liveTurnConversationID` is deliberately not cleared here: a cancelled
        // or failed run leaves its partial turn on screen, and that turn is
        // still one conversation's.
        runTask = nil
    }

    private func clearLoadTask(generation: UInt64) {
        guard generation == loadGeneration else { return }
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
    }

    private func clearUnloadTask(generation: UInt64) {
        guard generation == unloadGeneration else { return }
        unloadTask = nil
    }
}
