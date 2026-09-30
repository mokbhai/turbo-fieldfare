import Foundation

/// One-click sampling settings. Each sets temperature, Top-K and Top-P
/// together; repetition penalty and response length are left as they are.
public enum AppSamplingPreset: String, CaseIterable, Sendable, Identifiable {
    case precise
    case balanced
    case creative

    public var id: String { rawValue }

    public var label: String { rawValue.capitalized }

    public var detail: String {
        switch self {
        case .precise: "Greedy decoding. The same prompt gives the same answer."
        case .balanced: "The shipped defaults. Steady, with a little variety."
        case .creative: "More varied wording and ideas, less predictable."
        }
    }

    public var temperature: Double {
        switch self {
        case .precise: 0
        case .balanced: MacAppSettings().temperature
        case .creative: 0.9
        }
    }

    public var topK: Int { MacAppSettings().topK }

    public var topP: Double { MacAppSettings().topP }
}
