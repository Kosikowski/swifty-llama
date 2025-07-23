
import Foundation

// Conversation management
public struct Conversation: Codable {
    public let id: ConversationID
    public var messages: [ConversationMessage]
    public var totalTokens: Int32
    public let createdAt: Date

    public init(id: ConversationID, messages: [ConversationMessage], totalTokens: Int32, createdAt: Date) {
        self.id = id
        self.messages = messages
        self.totalTokens = totalTokens
        self.createdAt = createdAt
    }
}
