

/// Training dataset
public struct TrainingDataset: Codable, Sendable {
    public let training: [TrainingExample]
    public let validation: [TrainingExample]

    public init(training: [TrainingExample], validation: [TrainingExample]) {
        self.training = training
        self.validation = validation
    }
}
