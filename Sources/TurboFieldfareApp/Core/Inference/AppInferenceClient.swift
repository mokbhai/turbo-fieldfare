import Foundation

public protocol AppInferenceClient: Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel()

    /// Announces that a run has begun and will reach `generate` shortly.
    ///
    /// A cancel raised before its run's stream exists has to be held until that
    /// stream starts, or it is lost. What a client cannot tell on its own is a
    /// cancel raised just *before* a run from one raised just *after* one — both
    /// arrive while it is idle, and holding the second would end the following
    /// run instead. The caller can tell: it knows when a run begins. So this
    /// call, and only this call, opens the window in which a cancel may be
    /// held; `generate` closes it again. A cancel that lands outside that
    /// window belongs to no run and is dropped rather than carried into the
    /// next one.
    func expectGeneration()
}

public extension AppInferenceClient {
    /// A client that never holds a cancel has no window to open.
    func expectGeneration() {}
}

/// A client that owns a loadable model session. Loading is split from
/// generation so the UI can pre-load the ~1.6 GB resident weights once and
/// keep them warm across runs. Generation never loads or replaces a session.
public protocol AppModelLifecycleClient: AnyObject, AppInferenceClient {
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws
    func unload() async
}

public protocol AppInferenceMemoryReporting: AnyObject {
    var currentInferenceMemoryBytes: UInt64? { get }
}

public protocol AppInferenceTranscriptReporting: AnyObject {
    var generationTranscriptMailbox: GenerationTranscriptMailbox { get }
}
