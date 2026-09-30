import Foundation

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 1

    var version: Int = currentVersion
    var contextTokens: Int = AppContextLengthOption.defaultTokens
    var expertCacheSlots: Int = 16
    var temperature: Double = 0.2
    var topKEnabled: Bool = true
    var topK: Int = 64
    var topPEnabled: Bool = true
    var topP: Double = 0.95
    var prefillEnabled: Bool = true
    var repetitionPenalty: Double = 1.0
    var maxResponseTokensEnabled: Bool = false
    var maxResponseTokens: Int = 1_024
    /// Seeded into each new conversation. Existing conversations keep their own.
    var defaultSystemPrompt: String = ""

    static let repetitionPenaltyRange: ClosedRange<Double> = 1...2
    static let maxResponseTokensRange: ClosedRange<Int> = 16...262_144

    func isValid() -> Bool {
        version == Self.currentVersion
            && AppContextLengthOption.candidateTokens.contains(contextTokens)
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
            && repetitionPenalty.isFinite
            && Self.repetitionPenaltyRange.contains(repetitionPenalty)
            && Self.maxResponseTokensRange.contains(maxResponseTokens)
    }
}

extension MacAppSettings {
    private enum CodingKeys: String, CodingKey {
        case version, contextTokens, expertCacheSlots, temperature, topKEnabled,
             topK, topPEnabled, topP, prefillEnabled, repetitionPenalty,
             maxResponseTokensEnabled, maxResponseTokens, defaultSystemPrompt
    }

    /// Keys added after version 1 shipped decode as their defaults, so a file
    /// written by an older build keeps the choices it holds instead of being
    /// treated as corrupt and replaced.
    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = MacAppSettings()
        version = try container.decode(Int.self, forKey: .version)
        contextTokens = try container.decode(Int.self, forKey: .contextTokens)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        temperature = try container.decode(Double.self, forKey: .temperature)
        topKEnabled = try container.decode(Bool.self, forKey: .topKEnabled)
        topK = try container.decode(Int.self, forKey: .topK)
        topPEnabled = try container.decode(Bool.self, forKey: .topPEnabled)
        topP = try container.decode(Double.self, forKey: .topP)
        prefillEnabled = try container.decode(Bool.self, forKey: .prefillEnabled)
        repetitionPenalty = try container.decodeIfPresent(
            Double.self, forKey: .repetitionPenalty) ?? defaults.repetitionPenalty
        maxResponseTokensEnabled = try container.decodeIfPresent(
            Bool.self, forKey: .maxResponseTokensEnabled) ?? defaults.maxResponseTokensEnabled
        maxResponseTokens = try container.decodeIfPresent(
            Int.self, forKey: .maxResponseTokens) ?? defaults.maxResponseTokens
        defaultSystemPrompt = try container.decodeIfPresent(
            String.self, forKey: .defaultSystemPrompt) ?? defaults.defaultSystemPrompt
    }
}

enum MacAppSettingsFileStore {
    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(MacAppSettings.fileName, isDirectory: false)
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             fileManager: FileManager = .default) -> MacAppSettings {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
                guard settings.isValid() else { throw InvalidSettings() }
                return settings
            } catch {
                // Renamed, not deleted: settings are cheap to recreate but the
                // user did choose them, and nothing here justifies destroying
                // a file behind their back.
                let backup = fileURL.deletingLastPathComponent()
                    .appendingPathComponent("mac-app-settings.corrupt.json", isDirectory: false)
                try? fileManager.removeItem(at: backup)
                if (try? fileManager.moveItem(at: fileURL, to: backup)) == nil {
                    try? fileManager.removeItem(at: fileURL)
                }
            }
        }

        let settings = MacAppSettings()
        try? save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
        return settings
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw InvalidSettings() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private struct InvalidSettings: Error {}
}
