import Foundation

/// LoRA metadata
public struct LoRAMetadata: Codable, Equatable, Sendable {
    public let baseModelSHA: String
    public let loraConfig: String
    public let trainingRunHash: String
    public let trainingDate: Date
    public let rank: Int
    public let alpha: Float

    public init(
        baseModelSHA: String = "unknown",
        loraConfig: String = "unknown",
        trainingRunHash: String = "unknown",
        trainingDate: Date = Date(),
        rank: Int = 8,
        alpha: Float = 16.0
    ) {
        self.baseModelSHA = baseModelSHA
        self.loraConfig = loraConfig
        self.trainingRunHash = trainingRunHash
        self.trainingDate = trainingDate
        self.rank = rank
        self.alpha = alpha
    }
}
