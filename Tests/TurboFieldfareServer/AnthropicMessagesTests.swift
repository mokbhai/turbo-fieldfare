import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Anthropic request validation")
struct AnthropicValidationTests {
    @Test func systemBlocksAndTurnsTranslateToChatMessages() throws {
        let validated = try validate(#"""
        {"model":"test-model","max_tokens":64,
         "system":[{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}],
         "messages":[{"role":"user","content":"hi"},
                     {"role":"assistant","content":[{"type":"text","text":"hello"}]},
                     {"role":"user","content":"again"}],
         "temperature":0.5,"top_k":10,"top_p":0.9,"stop_sequences":["END"],"stream":true}
        """#)
        #expect(validated.messages.map(\.role) == [.system, .user, .assistant, .user])
        #expect(validated.messages.map(\.content) == ["Be brief.", "hi", "hello", "again"])
        #expect(validated.stream)
        #expect(validated.tools.isEmpty)
        #expect(validated.maximumCompletionTokens == 64)
        #expect(validated.generationConfig.temperature == 0.5)
        #expect(validated.generationConfig.topK == 10)
        #expect(validated.generationConfig.topP == 0.9)
        #expect(validated.generationConfig.stopStrings == ["END"])
    }

    @Test func unsupportedRequestsAreRefusedNotIgnored() throws {
        let cases = [
            #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
            #"{"model":"test-model","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}"#,
            #"{"model":"test-model","max_tokens":8,"messages":[]}"#,
            #"{"model":"test-model","max_tokens":8,"tools":[{"name":"x","input_schema":{}}],"messages":[{"role":"user","content":"hi"}]}"#,
            #"{"model":"test-model","max_tokens":8,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"hi"}]}"#,
            #"{"model":"test-model","max_tokens":8,"messages":[{"role":"user","content":[{"type":"image","source":{}}]}]}"#,
            #"{"model":"test-model","max_tokens":8,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"prefill"}]}"#,
            #"{"model":"test-model","max_tokens":8,"temperature":3,"messages":[{"role":"user","content":"hi"}]}"#,
        ]
        for body in cases {
            #expect(throws: ServerRequestError.self, "\(body)") { try validate(body) }
        }
    }

    @Test func wrongModelIsUnknown() {
        #expect(throws: ServerRequestError.unknownModel) {
            try validate(#"{"model":"other","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#)
        }
    }

    @Test func disabledThinkingIsAccepted() throws {
        _ = try validate(#"""
        {"model":"test-model","max_tokens":8,"thinking":{"type":"disabled"},
         "messages":[{"role":"user","content":"hi"}]}
        """#)
    }

    @Test func stopMatcherReportsWhichStopFired() {
        var matcher = StreamingStopMatcher(stops: ["STOP", "END"])
        #expect(matcher.push("abc EN") == "abc ")
        #expect(matcher.push("D tail") == "")
        #expect(matcher.isStopped)
        #expect(matcher.matchedStop == "END")
    }

    private func validate(_ json: String) throws -> ValidatedChatRequest {
        let request = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(json.utf8))
        return try AnthropicRequestValidator.validate(request, modelID: "test-model")
    }
}

private actor AnthropicScriptedBackend: ServerInferenceBackend {
    let stopSequence: String?
    let fails: Bool

    init(stopSequence: String? = nil, fails: Bool = false) {
        self.stopSequence = stopSequence
        self.fails = fails
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("hel"))
        if fails { throw CocoaError(.fileReadUnknown) }
        onEvent(.content("lo"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 10, completionTokens: 2, totalTokens: 12,
                               cachedTokens: 4),
            stopSequence: stopSequence)
    }
}

@Suite("Anthropic HTTP endpoint", .serialized)
struct AnthropicHTTPTests {
    @Test func nonStreamingMessageHasAnthropicShape() async throws {
        try await withServer(AnthropicScriptedBackend()) { port in
            let (data, status) = try await post(port, #"""
            {"model":"test-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}
            """#)
            #expect(status == 200)
            let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["type"] as? String == "message")
            #expect((object["id"] as? String)?.hasPrefix("msg_") == true)
            #expect(object["stop_reason"] as? String == "end_turn")
            #expect(object["stop_sequence"] is NSNull)
            let content = try #require(object["content"] as? [[String: Any]])
            #expect(content.first?["text"] as? String == "hello")
            let usage = try #require(object["usage"] as? [String: Any])
            #expect(usage["input_tokens"] as? Int == 6)
            #expect(usage["cache_read_input_tokens"] as? Int == 4)
            #expect(usage["output_tokens"] as? Int == 2)
        }
    }

    @Test func streamingEmitsTheFullEventSequence() async throws {
        try await withServer(AnthropicScriptedBackend(stopSequence: "END")) { port in
            let (data, status) = try await post(port, #"""
            {"model":"test-model","max_tokens":16,"stream":true,"stop_sequences":["END"],
             "messages":[{"role":"user","content":"hi"}]}
            """#)
            #expect(status == 200)
            let text = String(decoding: data, as: UTF8.self)
            let events = text.split(separator: "\n")
                .filter { $0.hasPrefix("event: ") }
                .map { String($0.dropFirst(7)) }
            #expect(events == ["message_start", "content_block_start",
                               "content_block_delta", "content_block_delta",
                               "content_block_stop", "message_delta", "message_stop"])
            #expect(text.contains(#""text":"hel""#))
            #expect(text.contains(#""stop_reason":"stop_sequence""#))
            #expect(text.contains(#""stop_sequence":"END""#))
        }
    }

    @Test func streamingFailureSendsAnErrorEvent() async throws {
        try await withServer(AnthropicScriptedBackend(fails: true)) { port in
            let (data, _) = try await post(port, #"""
            {"model":"test-model","max_tokens":16,"stream":true,
             "messages":[{"role":"user","content":"hi"}]}
            """#)
            let text = String(decoding: data, as: UTF8.self)
            #expect(text.contains("event: error"))
            #expect(text.contains(#""type":"api_error""#))
            #expect(!text.contains("message_stop"))
        }
    }

    @Test func errorsUseTheAnthropicEnvelope() async throws {
        try await withServer(AnthropicScriptedBackend()) { port in
            let (missing, missingStatus) = try await post(port, #"""
            {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
            """#)
            #expect(missingStatus == 400)
            let object = try #require(JSONSerialization.jsonObject(with: missing) as? [String: Any])
            #expect(object["type"] as? String == "error")
            #expect((object["error"] as? [String: Any])?["type"] as? String == "invalid_request_error")

            let (_, wrongModel) = try await post(port, #"""
            {"model":"other","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}
            """#)
            #expect(wrongModel == 404)

            let (_, malformed) = try await post(port, "{")
            #expect(malformed == 400)
        }
    }

    private func withServer(_ backend: some ServerInferenceBackend,
                            _ body: (Int) async throws -> Void) async throws {
        let server = TurboFieldfareHTTPServer(modelID: "test-model",
                                              queueLimit: 1,
                                              backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            try await body(port)
        } catch {
            try await server.shutdown()
            throw error
        }
        try await server.shutdown()
    }

    private func post(_ port: Int, _ body: String) async throws -> (Data, Int) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(body.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
}
