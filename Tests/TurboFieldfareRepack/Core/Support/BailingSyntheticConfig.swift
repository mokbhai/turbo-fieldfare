import Foundation

/// A realistic flat `config.json` for `inclusionAI/Ling-mini-2.0`, consumed as
/// `mlx-community/Ling-mini-2.0-4bit`.
///
/// Values are transcribed from the published config, not invented, because the
/// point of the tests using this fixture is that `ArchInfo.load` reads the keys
/// Ling really has rather than inheriting Gemma's defaults. A fixture written
/// to match the parser would prove nothing.
///
/// Unlike `SyntheticSnapshot` this writes **only** `config.json`: `ArchInfo.load`
/// reads nothing else, and no weights exist for this architecture yet — the
/// milestone this belongs to is plumbing, not execution.
enum BailingSyntheticConfig {

    /// The published config, trimmed to the architecture keys. `rms_norm_eps`
    /// and `max_position_embeddings` are carried even though nothing parses
    /// them: they are published facts, and their presence is the standing
    /// evidence that ignoring them is a decision rather than an oversight —
    /// `max_position_embeddings` in particular belongs to the context-cap
    /// policy, not to the arch description.
    static var dictionary: [String: Any] {
        [
            "model_type": "bailing_moe",
            "architectures": ["BailingMoeV2ForCausalLM"],

            "num_hidden_layers": 20,
            "hidden_size": 2048,
            // The DENSE layer-0 FFN width. NOT the shared-expert width — see
            // `moe_shared_expert_intermediate_size` below. Both keys exist and
            // both are plain integers, which is exactly why this fixture keeps
            // them ten times apart in magnitude.
            "intermediate_size": 5120,
            "first_k_dense_replace": 1,

            "moe_intermediate_size": 512,
            "moe_shared_expert_intermediate_size": 512,
            "num_shared_experts": 1,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "n_group": 8,
            "topk_group": 4,
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
            "score_function": "sigmoid",
            "moe_router_enable_expert_bias": true,

            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "use_qk_norm": true,
            "partial_rotary_factor": 0.5,
            "rope_theta": 600_000,

            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "vocab_size": 157_184,
            "max_position_embeddings": 32_768,
        ]
    }

    /// Write `config.json` into `directory`, optionally dropping or replacing
    /// keys. `dropping` is how the anti-inheritance tests ask "does removing
    /// the key Ling carries produce an error, or a silent Gemma default?".
    @discardableResult
    static func write(to directory: String,
                      dropping: [String] = [],
                      overriding: [String: Any] = [:]) throws -> String {
        try FileManager.default.createDirectory(atPath: directory,
                                                withIntermediateDirectories: true)
        var config = dictionary
        for key in dropping {
            precondition(config[key] != nil,
                         "dropping \(key), which the fixture does not carry — the test would pass vacuously")
            config.removeValue(forKey: key)
        }
        for (key, value) in overriding { config[key] = value }
        let path = (directory as NSString).appendingPathComponent("config.json")
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }
}
