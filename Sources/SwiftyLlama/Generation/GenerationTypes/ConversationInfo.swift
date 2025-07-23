import Foundation

/// Information about a conversation
public struct ConversationInfo {
    public let id: SwiftyLlamaID
    public let messageCount: Int
    public let totalTokens: Int32
    public let createdAt: Date
}
