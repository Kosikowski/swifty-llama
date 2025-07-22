
/// Training conversation
public struct TrainingConversation: Codable, Sendable {
    public let id: String
    public let messages: [TrainingMessage]

    public init(id: String, messages: [TrainingMessage]) {
        self.id = id
        self.messages = messages
    }
}
