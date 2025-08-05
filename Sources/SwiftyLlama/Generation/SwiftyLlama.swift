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

    /// Clear a conversation and remove its state without resetting the model's overall context
    /// - Parameter id: The conversation ID to clear
    func clearConversation(_ id: ConversationID)

    /// Continue a conversation with context reconstruction (for restored conversations)
    /// - Parameter id: The conversation ID to continue
    /// - Throws: GenerationError.conversationNotFound if conversation doesn't exist
    func continueConversationWithContextReconstruction(_ id: ConversationID) throws

    // MARK: - Conversation Titles

    /// Set a title for a conversation
    /// - Parameters:
    ///   - id: The conversation ID
    ///   - title: The title to set (must be 200 characters or less)
    /// - Throws: GenerationError.conversationNotFound if conversation doesn't exist
    func setConversationTitle(_ id: ConversationID, title: String) throws

    /// Get the title of a conversation
    /// - Parameter id: The conversation ID
    /// - Returns: The conversation title, or nil if not set
    func getConversationTitle(_ id: ConversationID) -> String?

    /// Auto-generate a title for a conversation using the LLM
    /// - Parameter id: The conversation ID
    /// - Throws: GenerationError.conversationNotFound if conversation doesn't exist
    /// - Note: Title will only be generated if conversation has sufficient tokens (minimum threshold)
    func generateConversationTitle(_ id: ConversationID) async throws

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

    // MARK: - State Management

    /// Get all conversations
    /// - Returns: Array of all conversations
    func getAllConversations() -> [Conversation]

    /// Set conversations (replaces all existing conversations)
    /// - Parameter conversations: Array of conversations to set
    func setConversations(_ conversations: [Conversation])

    /// Export conversations to data format
    /// - Returns: Data containing all conversations
    /// - Throws: Encoding errors if serialization fails
    func exportConversations() async throws -> Data

    /// Import conversations from data format
    /// - Parameter data: Data containing conversations
    /// - Throws: Decoding errors if deserialization fails
    func importConversations(_ data: Data) throws

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
        getAllConversations().map(\.id)
    }

    /// Clear all conversations
    func clearAllConversations() {
        for conversation in getAllConversations() {
            clearConversation(conversation.id)
        }
    }
}
