

/// LoRA compatibility information
public struct LoRACompatibility: Codable, Sendable {
    public let isCompatible: Bool
    public let warnings: [String]
    public let baseModelSHA: String
    public let adapterConfig: String

    public init(isCompatible: Bool, warnings: [String], baseModelSHA: String, adapterConfig: String) {
        self.isCompatible = isCompatible
        self.warnings = warnings
        self.baseModelSHA = baseModelSHA
        self.adapterConfig = adapterConfig
    }
}
