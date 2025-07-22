import Foundation

/// LoRA adapter information
public struct LoRAAdapter: Codable, Equatable, Sendable {
    public let path: String
    public let scale: Float
    public let metadata: LoRAMetadata
    public let appliedAt: Date

    public init(path: String, scale: Float, metadata: LoRAMetadata, appliedAt: Date) {
        self.path = path
        self.scale = scale
        self.metadata = metadata
        self.appliedAt = appliedAt
    }
}
