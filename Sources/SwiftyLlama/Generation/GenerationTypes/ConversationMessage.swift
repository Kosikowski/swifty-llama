
import Foundation

public struct ConversationMessage: Codable {
    public let role: String
    public let content: String
    public let tokens: [SwiftyLlamaToken] // SLlamaToken == Int32 is Codable
    public let timestamp: Date

    public init(role: String, content: String, tokens: [SwiftyLlamaToken], timestamp: Date) {
        self.role = role
        self.content = content
        self.tokens = tokens
        self.timestamp = timestamp
    }
}
