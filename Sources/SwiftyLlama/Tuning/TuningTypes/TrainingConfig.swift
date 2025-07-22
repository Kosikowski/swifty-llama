
/// Training configuration
public struct TrainingConfig: Codable, Sendable {
    public let loraRank: Int
    public let learningRate: Float
    public let epochs: Int
    public let batchSize: Int
    public let useQLoRA: Bool
    public let qLoRAConfig: QLoRAConfig?

    public init(
        loraRank: Int = 8,
        learningRate: Float = 2e-5,
        epochs: Int = 3,
        batchSize: Int = 1,
        useQLoRA: Bool = false,
        qLoRAConfig: QLoRAConfig? = nil
    ) {
        self.loraRank = loraRank
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
        self.useQLoRA = useQLoRA
        self.qLoRAConfig = qLoRAConfig
    }
}
