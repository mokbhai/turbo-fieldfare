import Foundation
import TurboFieldfare

/// Anthropic Messages API (`POST /v1/messages`), text only.
///
/// Requests are translated into the same `ValidatedChatRequest` the OpenAI
/// surface produces, so both share one backend, one queue, and one prompt
/// cache. Anything this translation cannot express faithfully — tools, images,
/// extended thinking, assistant prefill — is refused with a 400 rather than
/// silently dropped.

public struct AnthropicErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let type: String
        public let message: String
    }

    public let type: String
    public let error: Detail

    public init(type errorType: String, message: String) {
        type = "error"
        error = Detail(type: errorType, message: message)
    }

    /// The Anthropic rendering of a request failure. Status codes match the
    /// OpenAI surface so clients see the same outcome on either API.
    static func from(_ error: ServerRequestError) -> AnthropicErrorEnvelope {
        switch error {
        case .invalid(let message, _, _):
            AnthropicErrorEnvelope(type: "invalid_request_error", message: message)
        case .unknownModel:
            AnthropicErrorEnvelope(type: "not_found_error",
                                   message: "requested model is not available")
        case .queueFull:
            AnthropicErrorEnvelope(type: "rate_limit_error",
                                   message: "generation queue is full")
        }
    }
}

/// A content block as sent by a client. Only `text` is accepted; the other
/// fields a client may attach (such as `cache_control`) are ignored.
public struct AnthropicContentBlock: Decodable, Equatable, Sendable {
    public let type: String
    public let text: String?
}

/// `content` and `system` accept a plain string or an array of blocks.
public enum AnthropicContent: Decodable, Equatable, Sendable {
    case text(String)
    case blocks([AnthropicContentBlock])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .blocks(try container.decode([AnthropicContentBlock].self))
        }
    }

    func textValue(param: String) throws -> String {
        switch self {
        case .text(let text):
            return text
        case .blocks(let blocks):
            guard blocks.allSatisfy({ $0.type == "text" && $0.text != nil }) else {
                throw ServerRequestError.invalid(
                    message: "only text content blocks are supported",
                    param: param,
                    code: "unsupported_content")
            }
            return blocks.compactMap(\.text).joined()
        }
    }
}

public struct AnthropicMessage: Decodable, Equatable, Sendable {
    public let role: String
    public let content: AnthropicContent
}

public struct AnthropicThinking: Decodable, Equatable, Sendable {
    public let type: String
}

public struct AnthropicMessagesRequest: Decodable, Sendable {
    public let model: String
    public let maxTokens: Int?
    public let messages: [AnthropicMessage]
    public let system: AnthropicContent?
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let stopSequences: [String]?
    public let stream: Bool?
    public let tools: [JSONValue]?
    public let toolChoice: JSONValue?
    public let thinking: AnthropicThinking?

    enum CodingKeys: String, CodingKey {
        case model, messages, system, temperature, stream, tools, thinking
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case stopSequences = "stop_sequences"
        case toolChoice = "tool_choice"
    }
}

public enum AnthropicRequestValidator {
    public static func validate(_ request: AnthropicMessagesRequest,
                                modelID: String) throws -> ValidatedChatRequest {
        guard request.model == modelID else { throw ServerRequestError.unknownModel }
        guard request.tools?.isEmpty ?? true, request.toolChoice == nil else {
            throw invalid("tools are not supported on /v1/messages; use /v1/chat/completions",
                          "tools", "unsupported_value")
        }
        guard request.thinking == nil || request.thinking?.type == "disabled" else {
            throw invalid("extended thinking is not supported", "thinking", "unsupported_value")
        }
        guard let maximum = request.maxTokens else {
            throw invalid("max_tokens is required", "max_tokens", "missing_value")
        }
        guard maximum > 0 else {
            throw invalid("max_tokens must be positive", "max_tokens", "invalid_value")
        }

        let temperature = request.temperature ?? 0.2
        guard temperature.isFinite, temperature >= 0, temperature <= 2 else {
            throw invalid("temperature must be between 0 and 2", "temperature", "invalid_value")
        }
        let topP = request.topP ?? 0.95
        guard topP.isFinite, topP > 0, topP <= 1 else {
            throw invalid("top_p must be greater than 0 and at most 1", "top_p", "invalid_value")
        }
        let topK = request.topK ?? 64
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }

        var messages: [GFTokenizer.Message] = []
        if let system = try request.system?.textValue(param: "system"),
           !system.isEmpty {
            messages.append(GFTokenizer.Message(role: .system, content: system))
        }
        messages += try validateMessages(request.messages)

        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      stopStrings: request.stopSequences ?? [])
        return ValidatedChatRequest(messages: messages,
                                    tools: [],
                                    stream: request.stream ?? false,
                                    includeUsage: false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum)
    }

    private static func validateMessages(
        _ input: [AnthropicMessage]
    ) throws -> [GFTokenizer.Message] {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        // A trailing assistant message is an Anthropic prefill: the reply is
        // meant to continue it. The chat template can only render it as a
        // finished turn, so accepting it would quietly answer something else.
        guard input.last?.role == "user" else {
            throw invalid("the final message must have role user; assistant prefill is not supported",
                          "messages", "invalid_message")
        }
        return try input.map { message in
            let role: GFTokenizer.Role
            switch message.role {
            case "user": role = .user
            case "assistant": role = .assistant
            default:
                throw invalid("message role must be user or assistant",
                              "messages", "invalid_message")
            }
            let content = try message.content.textValue(param: "messages")
            guard !content.isEmpty else {
                throw invalid("message content must not be empty",
                              "messages", "invalid_message")
            }
            return GFTokenizer.Message(role: role, content: content)
        }
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}

/// Response shapes, as JSON objects so the handler can write them with the
/// same serializer it uses for OpenAI chunks.
enum AnthropicResponse {
    static func stopReason(for completion: ServerCompletion) -> String {
        if completion.stopSequence != nil { return "stop_sequence" }
        return completion.finishReason == "length" ? "max_tokens" : "end_turn"
    }

    /// Anthropic counts cache reads separately from `input_tokens`.
    static func usage(_ usage: OpenAIUsage, outputTokens: Int) -> [String: Any] {
        let cached = usage.promptTokensDetails.cachedTokens
        return [
            "input_tokens": max(0, usage.promptTokens - cached),
            "output_tokens": outputTokens,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        ]
    }

    static func message(id: String,
                        model: String,
                        content: String?,
                        stopReason: String?,
                        stopSequence: String?,
                        usage: [String: Any]) -> [String: Any] {
        [
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content.map { [["type": "text", "text": $0]] } ?? [],
            "stop_reason": stopReason.map { $0 as Any } ?? NSNull(),
            "stop_sequence": stopSequence.map { $0 as Any } ?? NSNull(),
            "usage": usage,
        ]
    }

    static func messageID() -> String {
        "msg_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    }
}
