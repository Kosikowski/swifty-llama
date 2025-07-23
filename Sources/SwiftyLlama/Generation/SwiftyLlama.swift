import Foundation
import SLlama

/// Public protocol defining the core interface for SwiftyLlama
/// This protocol allows for different implementations and better testability
@SwiftyLlamaActor
public protocol SwiftyLlama: AnyObject {
    // MARK: - Conversation Management

    /// Start a new conversation
    /// - Returns: The ID of the new conversation
    func startNewConversation() -> ConversationID

    /// Continue an existing conversation
    /// - Parameter id: The conversation ID to continue
    /// - Throws: GenerationError.conversationNotFound if conversation doesn't exist
    func continueConversation(_ id: ConversationID) throws

    /// Get the current conversation ID
    /// - Returns: The current conversation ID, or nil if no conversation is active
    func getCurrentConversationId() -> ConversationID?

    /// Get information about a conversation
    /// - Parameter id: The conversation ID
    /// - Returns: Conversation information, or nil if conversation doesn't exist
    func getConversationInfo(_ id: ConversationID) -> ConversationInfo?

    /// Clear a conversation and reset context
    /// - Parameter id: The conversation ID to clear
    func clearConversation(_ id: ConversationID)

    /// Continue a conversation with warm-up (for restored conversations)
    /// - Parameter id: The conversation ID to continue
    /// - Throws: GenerationError.conversationNotFound if conversation doesn't exist
    func continueConversationWithWarmUp(_ id: ConversationID) throws

    // MARK: - Generation

    /// Begin a new generation and immediately obtain a token stream
    /// - Parameters:
    ///   - prompt: The text prompt to generate from
    ///   - params: Generation parameters (temperature, max tokens, etc.)
    ///   - conversationId: Optional conversation ID to use (if nil, uses current or creates new)
    /// - Returns: A GenerationStream that provides tokens as they're generated
    @discardableResult
    func start(
        prompt: String,
        params: GenerationParams,
        conversationId: ConversationID?
    ) async
        -> GenerationStream

    /// Update generation parameters for an active generation
    /// - Parameters:
    ///   - id: The generation ID
    ///   - params: New generation parameters
    func update(id: GenerationID, _ params: GenerationParams) async

    /// Cancel an active generation
    /// - Parameter id: The generation ID to cancel
    func cancel(_ id: GenerationID) async

    /// Get information about a generation
    /// - Parameter id: The generation ID
    /// - Returns: Generation information, or nil if generation doesn't exist
    func getGenerationInfo(_ id: GenerationID) async -> GenerationInfo?

    /// Get all active generation IDs
    /// - Returns: Array of active generation IDs
    func getActiveGenerationIDs() async -> [GenerationID]

    /// Cancel all active generations
    func cancelAll() async

    // MARK: - Persistence

    /// Get the current state of all conversations for persistence
    /// - Returns: Array of all conversations
    func getConversationState() -> [Conversation]

    /// Restore conversations from persisted state
    /// - Parameter savedConversations: Array of saved conversations
    func restoreConversations(_ savedConversations: [Conversation])

    /// Save conversations to JSON data
    /// - Returns: JSON data containing all conversations
    /// - Throws: Encoding errors if serialization fails
    func saveConversationsToJSON() throws -> Data

    /// Load conversations from JSON data
    /// - Parameter data: JSON data containing conversations
    /// - Throws: Decoding errors if deserialization fails
    func loadConversationsFromJSON(_ data: Data) throws

    // MARK: - Model Information

    /// Get information about the loaded model
    var modelInfo: ModelInfo { get }

    /// Get information about the vocabulary
    var vocabInfo: VocabInfo { get }
}

// MARK: - Convenience Methods

public extension SwiftyLlama {
    /// Convenience method to start generation with current conversation
    /// - Parameters:
    ///   - prompt: The text prompt to generate from
    ///   - params: Generation parameters
    /// - Returns: A GenerationStream that provides tokens as they're generated
    @discardableResult
    func start(
        prompt: String,
        params: GenerationParams
    ) async
        -> GenerationStream
    {
        await start(prompt: prompt, params: params, conversationId: nil)
    }
}

// MARK: - Error Handling

public extension SwiftyLlama {
    /// Check if a conversation exists
    /// - Parameter id: The conversation ID to check
    /// - Returns: True if the conversation exists, false otherwise
    func conversationExists(_ id: ConversationID) -> Bool {
        getConversationInfo(id) != nil
    }

    /// Get all conversation IDs
    /// - Returns: Array of all conversation IDs
    func getAllConversationIDs() -> [ConversationID] {
        getConversationState().map(\.id)
    }

    /// Clear all conversations
    func clearAllConversations() {
        for conversation in getConversationState() {
            clearConversation(conversation.id)
        }
    }
}
