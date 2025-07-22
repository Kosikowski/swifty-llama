

/// Parameters for text generation
public struct GenerationParams: Hashable, Sendable {
    public let seed: UInt32
    public let topK: Int32
    public let topP: Float
    public let temperature: Float
    public let repeatPenalty: Float
    public let repetitionLookback: Int32
    public let maxTokens: Int32
    public let threads: Int32
    public let batchThreads: Int32
    public let enableEmbeddings: Bool
    public let enableCausalAttention: Bool

    public init(
        seed: UInt32 = 42,
        topK: Int32 = 40,
        topP: Float = 0.9,
        temperature: Float = 0.7,
        repeatPenalty: Float = 1.1,
        repetitionLookback: Int32 = 64,
        maxTokens: Int32 = 100,
        threads: Int32 = 4,
        batchThreads: Int32 = 4,
        enableEmbeddings: Bool = false,
        enableCausalAttention: Bool = true
    ) {
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temperature = temperature
        self.repeatPenalty = repeatPenalty
        self.repetitionLookback = repetitionLookback
        self.maxTokens = maxTokens
        self.threads = threads
        self.batchThreads = batchThreads
        self.enableEmbeddings = enableEmbeddings
        self.enableCausalAttention = enableCausalAttention
    }
}
