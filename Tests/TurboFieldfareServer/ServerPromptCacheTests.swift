import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server prompt cache")
struct ServerPromptCacheTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func textContinuationUsesActualGeneratedHistoryAndOnlyPrefillsSuffix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func identicalResubmissionReusesAllButTheFinalPromptToken() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let sampled = try #require(tokenizer.encode("answer", addBOS: false).first)
        var cache = ServerPromptCache()
        // The `max_tokens: 1` shape: the single sampled token never reaches the
        // KV, so the entry holds exactly the prompt.
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: sampled,
                reason: .maxTokens))
        #expect(cache.entry != nil)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: prompt,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected identical resubmission hit")
            return
        }
        #expect(effective == prompt)
        #expect(cached == prompt.count - 1)
        #expect(cached > 0 && cached < effective.count)
    }

    @Test func shorterRenderThatIsAPrefixOfTheCachedKVStillHits() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt + generated,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: prompt,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected shorter-prefix hit")
            return
        }
        #expect(effective == prompt)
        #expect(cached == prompt.count - 1)
        #expect(cached > 0 && cached < effective.count)
    }

    @Test func grownPromptStillReusesTheWholeCachedKV() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        // This branch compares raw token IDs, so a render that merely extends
        // the cached ones is a hit however the extra tokens were produced.
        let grown = prompt + tokenizer.encode(" more", addBOS: false)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: grown,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected grown-prompt hit")
            return
        }
        #expect(effective == grown)
        #expect(cached == prompt.count)
        #expect(cached < effective.count)
    }

    @Test func grownPromptReusesEvenASingleCachedToken() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let firstToken = try #require(prompt.first)
        var cache = ServerPromptCache()
        // One cached token is the smallest KV `publish` accepts. It is still a
        // usable prefix for a longer render, so the reusable count — not the
        // shared count — is what decides the hit.
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: [firstToken],
                kvBacked: [firstToken],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: prompt,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected a hit on a one-token cached prefix")
            return
        }
        #expect(effective == prompt)
        #expect(cached == 1)
    }

    @Test func identicalResubmissionKeepsHittingAsTheCachedKVGrowsPastThePrompt() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let answer = tokenizer.encode("answer one two three", addBOS: false)
        try #require(answer.count >= 4)
        var cache = ServerPromptCache()
        var reused: [Int] = []
        var cachedPositions: [Int] = []

        // Four rounds of the same request, as the server runs them: each round
        // resumes on the count the previous round reported, decodes, and
        // publishes the KV it ended on — prompt plus that round's generation.
        // The render never changes but the entry it is matched against is longer
        // and different every round, so no round repeats its predecessor.
        for round in 1...4 {
            cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt + answer.prefix(round),
                    boundary: tokenizer.endOfTurnID,
                    reason: .endOfTurn,
                    cachedPromptTokens: reused.last ?? 0))
            cachedPositions.append(try #require(cache.entry?.kvPosition))

            let match = cache.match(
                domain: domain,
                request: initial,
                renderedPromptIDs: prompt,
                tokenizer: tokenizer)

            guard case .hit(let effective, let cached) = match else {
                Issue.record("round \(round) of an identical resubmission missed")
                return
            }
            #expect(effective == prompt)
            reused.append(cached)
        }

        // The reuse is bounded by the render, not by the entry: however far past
        // the prompt the cached KV has run, an identical resubmission reuses all
        // but the final prompt token and prefills that one for its logits.
        #expect(reused == Array(repeating: prompt.count - 1, count: 4))
        // Publishing a *resumed* result keeps the entry alive, which is what
        // makes the next round a hit rather than a cold reset.
        #expect(cachedPositions == (1...4).map { prompt.count + $0 })
    }

    @Test func capturedOpenCodeToolResultUsesFrozenToolBoundary() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let continuation = try validatedFixture("opencode-1.15.11-tool-result.json")
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: initial.messages,
            tools: initial.tools)
        let assistant = continuation.messages[initial.messages.count]
        let prefix = try tokenizer.encodeToolChat(
            messages: initial.messages + [assistant],
            tools: initial.tools)
        let callStart = try #require(prefix.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(prefix.lastIndex(of: tokenizer.toolCallEndID))
        let generatedCall = Array(prefix[callStart...callEnd])
        let kvBacked = initialPrompt + generatedCall
        let historicalCall = try #require(assistant.toolCalls.first)
        let parsedCall = ParsedToolCall(
            id: historicalCall.id,
            name: historicalCall.name,
            arguments: historicalCall.arguments,
            argumentsJSON: try historicalCall.arguments.encoded())
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [parsedCall],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))
        let rendered = try tokenizer.encodeToolChat(
            messages: continuation.messages,
            tools: continuation.tools)

        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected captured OpenCode tool-result hit")
            return
        }
        let bridge = try tokenizer.encodeToolResultContinuation(
            cachedMessages: initial.messages,
            assistant: assistant,
            incomingMessages: continuation.messages,
            tools: continuation.tools)
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(bridge.first == tokenizer.toolResponseID)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .eos] {
            cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            #expect(cache.entry == nil)
        }

        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt + tokenizer.encode("answer", addBOS: false),
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let changed = request(messages: [
            GFTokenizer.Message(role: .user, content: "changed"),
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(changed.messages),
            addBOS: false)
        #expect(cache.match(
            domain: domain,
            request: changed,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) == .miss)
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer ",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn),
            stopStringFiltered: matcher.isStopped)
        #expect(cache.entry == nil)
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    /// `cachedPromptTokens` defaults to a cold run; pass it to model the result
    /// a *resumed* run publishes, which is what every request after the first
    /// one in a repeated conversation actually stores.
    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason,
        cachedPromptTokens: Int = 0
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: cachedPromptTokens,
            computedPrefillTokens: prompt.count - cachedPromptTokens,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary])
    }

    private func validatedFixture(_ name: String) throws -> ValidatedChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name,
            withExtension: nil,
            subdirectory: "Fixtures"))
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(contentsOf: url))
        return try OpenAIRequestValidator.validate(
            request,
            modelID: "gemma-4-26b-a4b-it")
    }
}

/// `ServerModelSession` needs the real checkpoint, so the start of a request is
/// only reachable here, in the pure resolver the session routes through.
@Suite("Server prompt cache decision")
struct ServerPromptCacheDecisionTests {
    private let promptIDs: [Int32] = [11, 12, 13, 14]
    private let effectivePromptIDs: [Int32] = [11, 12, 13, 14, 15, 16]

    @Test func hitResumesOnTheEffectiveIDsWhenTheKVCanReachThem() {
        var asked: [Int] = []

        let resolution = ServerPromptCacheDecision.resolve(
            match: .hit(effectivePromptIDs: effectivePromptIDs, cachedPromptTokens: 5),
            canResume: {
                asked.append($0)
                return true
            },
            promptIDs: promptIDs)

        // The probe must be asked about the count the resume will actually use.
        #expect(asked == [5])
        #expect(resolution == ServerPromptCacheDecision.Resolution(
            effectivePromptIDs: effectivePromptIDs,
            start: .resume(cachedPromptTokens: 5, allowingRewind: true),
            invalidatesCache: false))
    }

    @Test func hitTheKVCannotReachFallsBackToAFullReset() {
        let resolution = ServerPromptCacheDecision.resolve(
            match: .hit(effectivePromptIDs: effectivePromptIDs, cachedPromptTokens: 5),
            canResume: { _ in false },
            promptIDs: promptIDs)

        // A refused rewind must fall back to the raw prompt, not the effective
        // IDs: those carry cached history the reset is about to throw away.
        #expect(resolution == ServerPromptCacheDecision.Resolution(
            effectivePromptIDs: promptIDs,
            start: .reset,
            invalidatesCache: true))
    }

    /// The stubs pin the branching; this pins the probe the server actually
    /// hands in. `ServerModelSession` passes `RealForwardRunner.canResume`,
    /// which is a one-line delegation to `KVCacheManager.canResume(from:)`: the
    /// runner cannot be built without the checkpoint, a cache manager needs only
    /// a Metal device, so this is as deep as the real mapping can be driven.
    @Test func aRealKVCacheResumesInsideTheRingSlackAndResetsBeyondIt() throws {
        let window = 4
        let context = try MetalContext()
        // A scaled-down ring: production numbers (window 1024, capacity 1152)
        // run the same arithmetic but would put a thousand token IDs in every
        // failure message. `KVCacheManagerTests` covers the production sizes.
        let kv = try KVCacheManager(device: context.device,
                                    config: .gemma4_26B_A4B,
                                    maxContext: 32,
                                    fp16RingEnabled: true,
                                    slidingWindow: window,
                                    maxPrefillChunkTokens: 4,
                                    fp16RingCapacityOverride: 8)
        // A ring layer must keep a whole window resident, so only what its
        // capacity has over that window is rewindable.
        let slack = kv.capacity(layer: 0) - window
        try #require(slack > 0)
        kv.advance(by: 8)
        let rawPromptIDs: [Int32] = [11, 12, 13, 14, 15, 16, 17, 18]
        let effective = rawPromptIDs + [19, 20]
        let reachable = kv.position - slack

        func resolve(cachedPromptTokens: Int) -> ServerPromptCacheDecision.Resolution {
            ServerPromptCacheDecision.resolve(
                match: .hit(effectivePromptIDs: effective,
                            cachedPromptTokens: cachedPromptTokens),
                canResume: { kv.canResume(from: $0) },
                promptIDs: rawPromptIDs)
        }

        #expect(resolve(cachedPromptTokens: reachable)
            == ServerPromptCacheDecision.Resolution(
                effectivePromptIDs: effective,
                start: .resume(cachedPromptTokens: reachable, allowingRewind: true),
                invalidatesCache: false))
        // One token further back the ring no longer holds everything the
        // re-prefill would read, so a real cache turns the same hit into a
        // reset. A stubbed probe can only assume this branch exists.
        #expect(resolve(cachedPromptTokens: reachable - 1)
            == ServerPromptCacheDecision.Resolution(
                effectivePromptIDs: rawPromptIDs,
                start: .reset,
                invalidatesCache: true))
        // Reusing nothing is a legal rewind but not a resume.
        #expect(resolve(cachedPromptTokens: 0)
            == ServerPromptCacheDecision.Resolution(
                effectivePromptIDs: rawPromptIDs,
                start: .reset,
                invalidatesCache: true))

        // The promised resume must be one the cache will actually perform:
        // `rewind` traps rather than returning when it will not.
        kv.rewind(to: reachable)
        #expect(kv.position == reachable)
    }

    @Test func missResetsOnTheRawPromptIDs() {
        let resolution = ServerPromptCacheDecision.resolve(
            match: .miss,
            canResume: { _ in
                Issue.record("a miss has no cached count to probe")
                return true
            },
            promptIDs: promptIDs)

        #expect(resolution == ServerPromptCacheDecision.Resolution(
            effectivePromptIDs: promptIDs,
            start: .reset,
            invalidatesCache: true))
    }
}
