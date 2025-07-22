import Foundation

/// Training example
public struct TrainingExample: Codable, Sendable {
    public let conversation: TrainingConversation
    public let formattedText: String
    public let tokens: [SwiftyLlamaToken]

    public init(conversation: TrainingConversation, formattedText: String, tokens: [SwiftyLlamaToken]) {
        self.conversation = conversation
        self.formattedText = formattedText
        self.tokens = tokens
    }
}
