import Foundation
import SLlama

// MARK: - Conversation Management Extension

public extension SwiftyLlamaCore {
    // MARK: - Conversation Lifecycle

    /// Start a new conversation
    func startNewConversation() -> ConversationID {
        let id = ConversationID()
        conversations[id] = Conversation(
            id: id,
            messages: [],
            totalTokens: 0,
            createdAt: Date(),
            title: nil
        )
        currentConversationId = id

        // Initialize conversation state
        _ = getOrCreateConversationState(for: id)

        return id
    }

    /// Continue an existing conversation
    func continueConversation(_ id: ConversationID) throws {
        guard conversations[id] != nil else {
            throw GenerationError.conversationNotFound(conversationId: id)
        }
        currentConversationId = id
    }

    /// Get current conversation ID
    func getCurrentConversationId() -> ConversationID? {
        currentConversationId
    }

    /// Get conversation info
    func getConversationInfo(_ id: ConversationID) -> ConversationInfo? {
        guard let conversation = conversations[id] else { return nil }

        return ConversationInfo(
            id: conversation.id,
            messageCount: conversation.messages.count,
            totalTokens: conversation.totalTokens,
            createdAt: conversation.createdAt
        )
    }

    /// Clear a conversation by removing its state without resetting the model context.
    /// Context memory remains untouched unless explicitly reset elsewhere.
    func clearConversation(_ id: ConversationID) {
        conversations.removeValue(forKey: id)
        clearConversationState(for: id)

        if currentConversationId == id {
            currentConversationId = nil
        }
    }

    // MARK: - Conversation Titles

    /// Set a title for a conversation
    func setConversationTitle(_ id: ConversationID, title: String) throws {
        guard conversations[id] != nil else {
            throw GenerationError.conversationNotFound(conversationId: id)
        }

        guard title.count <= 200 else {
            throw GenerationError.titleTooLong(conversationId: id, maxLength: 200)
        }

        conversations[id]?.title = title
    }

    /// Get the title of a conversation
    func getConversationTitle(_ id: ConversationID) -> String? {
        conversations[id]?.title
    }

    /// Auto-generate a title for a conversation using the LLM
    func generateConversationTitle(_ id: ConversationID) async throws {
        guard let conversation = conversations[id] else {
            throw GenerationError.conversationNotFound(conversationId: id)
        }

        // Minimum token threshold for title generation (to ensure meaningful context)
        let minimumTokensForTitle = 10

        guard conversation.totalTokens >= minimumTokensForTitle else {
            throw GenerationError.insufficientTokensForTitleGeneration(
                conversationId: id,
                requiredTokens: minimumTokensForTitle
            )
        }

        // Create a title generation prompt based on the conversation content
        let titlePrompt = createTitleGenerationPrompt(from: conversation)

        // Generate title with conservative parameters
        let titleParams = GenerationParams(
            seed: 42,
            topK: 40,
            topP: 0.9,
            temperature: 0.8,
            repeatPenalty: 1.1,
            maxTokens: 50 // Give more room for title generation
        )

        // Use a temporary conversation for title generation to avoid affecting the original conversation
        let tempConversationId = ConversationID()
        conversations[tempConversationId] = Conversation(
            id: tempConversationId,
            messages: [],
            totalTokens: 0,
            createdAt: Date(),
            title: nil
        )

        let titleStream = await start(prompt: titlePrompt, params: titleParams, conversationId: tempConversationId)

        var generatedTitle = ""
        for try await token in titleStream.stream {
            generatedTitle += token
        }

        // Clean up the generated title
        let cleanedTitle = generatedTitle.sanitizeForTitle()

        // Set the generated title
        conversations[id]?.title = cleanedTitle

        // Clean up the temporary conversation
        conversations.removeValue(forKey: tempConversationId)
        clearConversationState(for: tempConversationId)
    }

    // MARK: - Private Title Generation Helpers

    internal func createTitleGenerationPrompt(from conversation: Conversation) -> String {
        // Extract the first few messages to provide context for title generation
        let contextMessages = Array(conversation.messages.prefix(3))
        let contextText = contextMessages.map { message in
            "\(message.role): \(message.content)"
        }.joined(separator: "\n")

        return """
        Create a short, descriptive title for this conversation (max 200 characters):

        \(contextText)

        Title:
        """
    }

    /// Auto-generate titles for conversations that don't have titles
    internal func autoGenerateTitlesForConversations() async {
        for (id, conversation) in conversations {
            // Only generate titles for conversations that don't already have one
            if conversation.title == nil {
                do {
                    try await generateConversationTitle(id)
                } catch {
                    // Silently ignore title generation errors to avoid blocking save
                    // The conversation will be saved without a title
                }
            }
        }
    }

    // MARK: - Persistence Support

    /// Get all conversations
    func getAllConversations() -> [Conversation] {
        Array(conversations.values)
    }

    /// Set conversations (replaces all existing conversations)
    func setConversations(_ newConversations: [Conversation]) {
        conversations.removeAll()
        conversationStates.removeAll()

        for conversation in newConversations {
            conversations[conversation.id] = conversation
            // Initialize conversation state for restored conversations
            // Mark as not warmed up since we need to reconstruct the context
            updateConversationState(for: conversation.id) { state in
                state.isWarmedUp = false
                state.currentTokenCount = 0
                state.tokenBuffer.removeAll()
            }
        }
    }

    /// Export conversations to data format
    func exportConversations() async throws -> Data {
        // Auto-generate titles for conversations that don't have titles
        await autoGenerateTitlesForConversations()

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(getAllConversations())
    }

    /// Import conversations from data format
    func importConversations(_ data: Data) throws {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let savedConversations = try decoder.decode([Conversation].self, from: data)

        // Force a complete context reset by recreating the context entirely
        // Recreate the context to ensure complete reset with proper size
        let contextParams = SLlamaContext.createParams(
            contextSize: UInt32(maxContextSize),
            batchSize: UInt32(maxContextSize)
        )
        context = try SLlamaContext(model: model, contextParams: contextParams)

        // Reinitialize the batch to ensure no residual state
        batch = SLlamaBatch(nTokens: maxContextSize, nSeqMax: 1)
        clearBatch()

        // Reset conversation states to ensure clean slate
        conversationStates.removeAll()

        setConversations(savedConversations)
    }

    /// Continue a conversation with context reconstruction (for restored conversations)
    func continueConversationWithContextReconstruction(_ id: ConversationID) throws {
        guard let conversation = conversations[id] else {
            throw GenerationError.conversationNotFound(conversationId: id)
        }

        // Force a complete context reset by recreating the context entirely
        // Recreate the context to ensure complete reset with proper size
        let contextParams = SLlamaContext.createParams(
            contextSize: UInt32(maxContextSize),
            batchSize: UInt32(maxContextSize)
        )
        context = try SLlamaContext(model: model, contextParams: contextParams)

        // Reinitialize the batch to ensure no residual state
        batch = SLlamaBatch(nTokens: maxContextSize, nSeqMax: 1)
        clearBatch()

        // Reset conversation state to ensure clean slate
        updateConversationState(for: id) { state in
            state.currentTokenCount = 0
            state.tokenBuffer.removeAll()
            state.isWarmedUp = false
        }

        // Reconstruct the context with conversation history
        try reconstructContext(with: conversation)
        currentConversationId = id
    }

    // MARK: - Conversation Storage

    internal func storeConversationMessage(
        conversationId: ConversationID,
        role: String,
        content: String,
        tokens: [SLlamaToken]
    ) async {
        // Ensure conversation exists
        if conversations[conversationId] == nil {
            conversations[conversationId] = Conversation(
                id: conversationId,
                messages: [],
                totalTokens: 0,
                createdAt: Date()
            )
        }

        guard var conversation = conversations[conversationId] else { return }

        let message = ConversationMessage(
            role: role,
            content: content,
            tokens: tokens,
            timestamp: Date()
        )

        conversation.messages.append(message)
        conversation.totalTokens += Int32(tokens.count)
        conversations[conversationId] = conversation
    }
}
