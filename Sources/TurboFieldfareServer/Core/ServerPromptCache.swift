import Foundation
import TurboFieldfare

public enum ServerPromptCacheMode: String, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
}

struct ServerPromptCacheDomain: Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Sendable, Equatable {
    let message: GFTokenizer.Message
    let rawStopReason: StopReason
}

struct ServerPromptCacheEntry: Sendable, Equatable {
    let domain: ServerPromptCacheDomain
    let inputMessages: [GFTokenizer.Message]
    let tools: [GFTokenizer.FunctionDefinition]
    let assistantTurn: CachedAssistantTurn
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss
    case hit(effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

struct ServerPromptCache: Sendable {
    private(set) var entry: ServerPromptCacheEntry?

    mutating func invalidate() {
        entry = nil
    }

    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool = false
    ) {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            entry = nil
            return
        }
        let historicalCalls = calls.map {
            GFTokenizer.HistoricalToolCall(
                id: $0.id,
                name: $0.name,
                arguments: $0.arguments)
        }
        let assistant = GFTokenizer.Message(
            role: .assistant,
            content: calls.isEmpty ? content : nil,
            toolCalls: historicalCalls)
        entry = ServerPromptCacheEntry(
            domain: domain,
            inputMessages: request.messages,
            tools: request.tools,
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition)
    }

    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        guard let entry,
              entry.domain == domain,
              entry.tools == request.tools,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return .miss
        }

        let shared = min(renderedPromptIDs.count, entry.kvPosition)
        // When the whole rendered prompt is already in the KV — an identical
        // resubmission — the resume must still leave one token to prefill,
        // because a completion needs logits to sample from and
        // `runRawCompletion` rejects a cached count equal to the prompt.
        // Guarding on the reusable count rather than on `shared` is what keeps a
        // prompt that merely extends a one-token KV a hit.
        let cached = shared == renderedPromptIDs.count ? shared - 1 : shared
        if cached > 0,
           renderedPromptIDs.prefix(shared)
            .elementsEqual(entry.kvBackedTokenIDs.prefix(shared)) {
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: cached)
        }

        let inputCount = entry.inputMessages.count
        guard request.messages.count > inputCount + 1,
              request.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
              assistantMatches(
                request.messages[inputCount],
                entry.assistantTurn.message) else {
            return .miss
        }
        let continuation = Array(request.messages.dropFirst(inputCount + 1))

        if entry.assistantTurn.message.toolCalls.isEmpty {
            return matchTextContinuation(
                entry: entry,
                continuation: continuation,
                tokenizer: tokenizer)
        }
        return matchToolContinuation(
            entry: entry,
            request: request,
            continuation: continuation,
            tokenizer: tokenizer)
    }

    private func assistantMatches(
        _ incoming: GFTokenizer.Message,
        _ cached: GFTokenizer.Message
    ) -> Bool {
        guard incoming.role == .assistant,
              cached.role == .assistant,
              incoming.toolCalls == cached.toolCalls,
              incoming.toolCallID == cached.toolCallID,
              incoming.name == cached.name else {
            return false
        }
        if !cached.toolCalls.isEmpty {
            return (incoming.content ?? "").isEmpty
                && (cached.content ?? "").isEmpty
        }
        return incoming.content == cached.content
    }

    private func matchTextContinuation(
        entry: ServerPromptCacheEntry,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        guard continuation.count == 1,
              continuation[0].role == .user,
              let content = continuation[0].content,
              continuation[0].toolCalls.isEmpty,
              continuation[0].toolCallID == nil,
              entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens else {
            return .miss
        }
        var bridge = tokenizer.encodeTextContinuation(userContent: content)
        if entry.assistantTurn.rawStopReason == .maxTokens {
            bridge = entry.uncommittedBoundaryTokenIDs + bridge
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }

    private func matchToolContinuation(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        let calls = entry.assistantTurn.message.toolCalls
        guard entry.assistantTurn.rawStopReason == .toolCalls,
              continuation.count == calls.count,
              zip(continuation, calls).allSatisfy({ message, call in
                  message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
              }) else {
            return .miss
        }
        guard let bridge = try? tokenizer.encodeToolResultContinuation(
            cachedMessages: entry.inputMessages,
            assistant: entry.assistantTurn.message,
            incomingMessages: request.messages,
            tools: request.tools),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }
}

/// How a matched prefix turns into the start of one completion.
///
/// This is deliberately separate from `ServerModelSession`, which cannot be
/// built without the real checkpoint: the branch that matters most here is a hit
/// the KV can no longer reach, and that fallback is the only thing between a
/// refused rewind and a failed request.
enum ServerPromptCacheDecision {
    struct Resolution: Sendable, Equatable {
        let effectivePromptIDs: [Int32]
        let start: RawCompletionStart
        /// Whether the caller must drop the retained entry. An entry that
        /// survived a reset would describe a KV prefix the next request can no
        /// longer resume onto.
        let invalidatesCache: Bool
    }

    /// `canResume` is a probe, not a command: it is asked only about a count the
    /// match already produced, so the caller never commits to a resume it cannot
    /// perform.
    static func resolve(
        match: ServerPromptCacheMatch,
        canResume: (Int) -> Bool,
        promptIDs: [Int32]
    ) -> Resolution {
        let fallback = Resolution(
            effectivePromptIDs: promptIDs,
            start: .reset,
            invalidatesCache: true)
        guard case .hit(let effective, let cached) = match else { return fallback }
        // The KV can no longer reach the resume point once its ring has wrapped
        // past it. Rebuilding the prefix costs time; failing the request costs
        // the answer.
        guard canResume(cached) else { return fallback }
        // `match` verified the cached count against the KV's own token IDs, so a
        // count short of the cursor is a genuine prefix and the rewind onto it
        // is sound.
        return Resolution(
            effectivePromptIDs: effective,
            start: .resume(cachedPromptTokens: cached, allowingRewind: true),
            invalidatesCache: false)
    }
}
