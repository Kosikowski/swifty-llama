

/// Training metrics
public struct TrainingMetrics: Codable, Sendable {
    public var currentEpoch: Int = 0
    public var currentStep: Int = 0
    public var trainingLoss: Float = 0.0
    public var validationLoss: Float = 0.0
    public var learningRate: Float = 0.0
    public var epoch: Int = 0
    public var loss: Float = 0.0

    public init() {}

    public init(epoch: Int, loss: Float, learningRate: Float) {
        self.epoch = epoch
        self.loss = loss
        self.learningRate = learningRate
        currentEpoch = epoch
        trainingLoss = loss
        self.learningRate = learningRate
    }
}
