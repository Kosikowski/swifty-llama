

/// Training status
public enum TrainingStatus: String, Codable, Sendable {
    case running
    case completed
    case stopped
    case failed
}
