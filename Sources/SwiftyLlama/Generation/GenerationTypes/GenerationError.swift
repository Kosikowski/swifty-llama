import Foundation

// Errors that can occur during generation
public enum GenerationError: Error, LocalizedError {
    case abortedByUser(generationId: GenerationID)
    case modelNotLoaded
    case contextNotInitialized
    case conversationNotFound(conversationId: ConversationID)
    case contextPreparationFailed(conversationId: ConversationID)
    case tokenizationFailed(conversationId: ConversationID)
    case generationFailed(generationId: GenerationID)
    case invalidState(conversationId: ConversationID?)

    public var errorDescription: String? {
        switch self {
            case let .abortedByUser(generationId):
                "Generation was aborted by user (GenerationID: \(generationId))"
            case .modelNotLoaded:
                "Model is not loaded"
            case .contextNotInitialized:
                "Context not initialized"
            case let .conversationNotFound(conversationId):
                "Conversation not found (ConversationID: \(conversationId))"
            case let .contextPreparationFailed(conversationId):
                "Failed to prepare context (ConversationID: \(conversationId))"
            case let .tokenizationFailed(conversationId):
                "Failed to tokenize input (ConversationID: \(conversationId))"
            case let .generationFailed(generationId):
                "Generation failed (GenerationID: \(generationId))"
            case let .invalidState(conversationId):
                if let conversationId {
                    "Invalid state (ConversationID: \(conversationId))"
                } else {
                    "Invalid state"
                }
        }
    }
}
