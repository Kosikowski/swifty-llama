
import Foundation

/// Training session
public struct TrainingSession: Codable, Sendable {
    public let id: String
    public let dataset: TrainingDataset
    public let config: TrainingConfig
    public let startTime: Date
    public var endTime: Date?
    public var status: TrainingStatus

    public init(
        id: String,
        dataset: TrainingDataset,
        config: TrainingConfig,
        startTime: Date,
        status: TrainingStatus,
        endTime: Date? = nil
    ) {
        self.id = id
        self.dataset = dataset
        self.config = config
        self.startTime = startTime
        self.status = status
        self.endTime = endTime
    }
}
