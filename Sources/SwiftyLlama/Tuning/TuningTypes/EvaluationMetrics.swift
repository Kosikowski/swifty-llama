

/// Evaluation metrics
public struct EvaluationMetrics: Codable, Sendable {
    public let perplexity: Float
    public let averageLoss: Float
    public let totalExamples: Int
    public let totalTokens: Int

    public init(perplexity: Float, averageLoss: Float, totalExamples: Int, totalTokens: Int) {
        self.perplexity = perplexity
        self.averageLoss = averageLoss
        self.totalExamples = totalExamples
        self.totalTokens = totalTokens
    }
}
