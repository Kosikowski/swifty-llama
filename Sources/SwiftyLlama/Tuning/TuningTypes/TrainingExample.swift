import Foundation

/// Training example
public struct TrainingExample: Codable, Sendable {
    public let conversation: TrainingConversation
    public let formattedText: String
    public let tokens: [SwiftyLlamaToken]
    public let targetTokens: [SwiftyLlamaToken]

    public init(
        conversation: TrainingConversation,
        formattedText: String,
        tokens: [SwiftyLlamaToken],
        targetTokens: [SwiftyLlamaToken]
    ) {
        self.conversation = conversation
        self.formattedText = formattedText
        self.tokens = tokens
        self.targetTokens = targetTokens
    }
}
