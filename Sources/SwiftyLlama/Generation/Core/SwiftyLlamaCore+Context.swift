import Foundation
import SLlama

// MARK: - Context Management Extension

extension SwiftyLlamaCore {
    // MARK: - Context Initialization

    func ensureContextInitialized() throws {
        guard !isContextInitialized else { return }

        // Create context with the configured context size
        let contextParams = SLlamaContext.createParams(
            contextSize: UInt32(maxContextSize),
            batchSize: UInt32(maxContextSize)
        )
        context = try SLlamaContext(model: model, contextParams: contextParams)

        // Create single batch instance
        batch = SLlamaBatch(nTokens: maxContextSize, nSeqMax: 1)

        isContextInitialized = true
    }

    func configureContext(with params: GenerationParams) throws {
        guard let context else { throw GenerationError.contextNotInitialized }

        // Configure the context using parameters
        context.setThreads(nThreads: params.threads, nThreadsBatch: params.batchThreads)
        context.setEmbeddings(params.enableEmbeddings)
        context.setCausalAttention(params.enableCausalAttention)
    }

    // MARK: - Batch Management

    func clearBatch() {
        guard let batch else { return }
        batch.clear()
    }

    func addToBatch(token: SLlamaToken, position: Int32, isLogit: Bool = true) {
        guard let batch else { return }
        // Safe to use try! since batch is created with maxContextSize capacity
        // and we ensure we never exceed that limit
        try! batch.addToken(token, position: position, sequenceIds: [0], logits: isLogit)
    }

    // MARK: - Context Reconstruction

    /// Reconstruct context with conversation history (lazy reconstruction of KV cache)
    func reconstructContext(with conversation: Conversation) throws {
        guard let ctx = context else { return }
        let hist = conversation.messages.flatMap(\.tokens)
        guard !hist.isEmpty else { return }

        // 1) Start from a clean slate
        ctx.clearMemory(data: true)

        // Update conversation state
        updateConversationState(for: conversation.id) { state in
            state.currentTokenCount = 0
            state.tokenBuffer.removeAll()
            state.isWarmedUp = true
        }

        clearBatch()

        // 2) Configure context for warm-up (use default params)
        let defaultParams = GenerationParams(temperature: 0.7, maxTokens: 100)
        try configureContext(with: defaultParams)

        // 3) Feed every historical token
        for (i, tok) in hist.enumerated() {
            addToBatch(
                token: tok,
                position: Int32(i),
                isLogit: i == hist.count - 1
            ) // last token gets logits
        }
        try ctx.core().decode(batch!)

        // Update conversation state with historical token count
        updateConversationState(for: conversation.id) { state in
            state.currentTokenCount = Int32(hist.count)
            state.tokenBuffer = hist
        }
    }

    /// Check if a conversation needs warm-up (has history but context is cold)
    func needsContextReconstruction(for conversationId: ConversationID) -> Bool {
        guard let conversation = conversations[conversationId] else { return false }
        let state = getOrCreateConversationState(for: conversationId)

        // Need warm-up if conversation has history but our context is empty
        // AND we haven't already warmed up this conversation
        return !conversation.messages.isEmpty && (state.currentTokenCount == 0 || state.tokenBuffer.isEmpty)
    }

    // MARK: - Context Preparation with Conversation History

    func prepareContextWithConversation(
        for promptTokens: [SLlamaToken],
        conversationId: ConversationID,
        llamaContext: SLlamaContext
    )
        -> Bool
    {
        guard !promptTokens.isEmpty else {
            return false
        }

        // Check if conversation exists
        if let conversation = conversations[conversationId] {
            // Existing conversation - continue from history
            let historyTokens = conversation.messages.flatMap(\.tokens)
            let totalTokens = historyTokens.count + promptTokens.count

            // Robust fallback: truncate oldest messages if context is too long
            if totalTokens >= maxContextSize {
                let allowed = max(0, Int(maxContextSize) - promptTokens.count)
                let trimmedHistory = Array(historyTokens.suffix(allowed))
                let adjustedTotalTokens = trimmedHistory.count + promptTokens.count

                // Validate that our truncation worked correctly
                guard adjustedTotalTokens <= maxContextSize else {
                    return false
                }

                // Continue with trimmed history
                let needsContextClear = trimmedHistory.isEmpty ||
                    !getOrCreateConversationState(for: conversationId).isWarmedUp ||
                    (getOrCreateConversationState(for: conversationId).currentTokenCount == 0 && !trimmedHistory
                        .isEmpty
                    ) ||
                    (getOrCreateConversationState(for: conversationId)
                        .currentTokenCount > 0 && getOrCreateConversationState(for: conversationId).tokenBuffer.isEmpty
                    ) ||
                    (getOrCreateConversationState(for: conversationId)
                        .currentTokenCount > 0 && getOrCreateConversationState(for: conversationId).tokenBuffer
                        .count != getOrCreateConversationState(for: conversationId).currentTokenCount
                    )

                if needsContextClear {
                    // Clear the KV cache completely for a fresh start
                    llamaContext.clearMemory(data: true)

                    updateConversationState(for: conversationId) { state in
                        state.currentTokenCount = 0
                        state.tokenBuffer.removeAll()
                    }

                    clearBatch()

                    // If we have trimmed history, we need to warm up the context first
                    if !trimmedHistory.isEmpty {
                        // Warm up with trimmed historical tokens
                        for (i, token) in trimmedHistory.enumerated() {
                            addToBatch(token: token, position: Int32(i), isLogit: false)
                        }

                        do {
                            try llamaContext.core().decode(batch!)
                        } catch {
                            return false
                        }

                        // Update state with trimmed historical tokens
                        updateConversationState(for: conversationId) { state in
                            state.currentTokenCount = Int32(trimmedHistory.count)
                            state.tokenBuffer = trimmedHistory
                        }
                    }
                }

                // Now add the new prompt tokens
                clearBatch()

                let startPosition = getOrCreateConversationState(for: conversationId).currentTokenCount
                for (i, token) in promptTokens.enumerated() {
                    let position = startPosition + Int32(i)
                    addToBatch(token: token, position: position, isLogit: i == promptTokens.count - 1)
                }

                do {
                    try llamaContext.core().decode(batch!)
                } catch {
                    return false
                }

                // Update token count
                updateConversationState(for: conversationId) { state in
                    state.currentTokenCount = startPosition + Int32(promptTokens.count)
                    state.shouldContinuePredicting = true
                }

                return true
            }

            // Get current conversation state to check if we need to clear context
            let currentState = getOrCreateConversationState(for: conversationId)

            // Only clear context if:
            // 1. This is a new conversation (no history)
            // 2. This conversation hasn't been warmed up yet
            // 3. We're switching from a different conversation that wasn't properly warmed up
            // 4. The conversation state is inconsistent (has history but no token count)
            // 5. We're continuing a conversation that was loaded from JSON (needs fresh context)
            let needsContextClear = historyTokens.isEmpty ||
                !currentState.isWarmedUp ||
                (currentState.currentTokenCount == 0 && !historyTokens.isEmpty) ||
                (currentState.currentTokenCount > 0 && currentState.tokenBuffer.isEmpty) ||
                (currentState.currentTokenCount > 0 && currentState.tokenBuffer.count != currentState.currentTokenCount)

            if needsContextClear {
                // Clear the KV cache completely for a fresh start
                llamaContext.clearMemory(data: true)

                updateConversationState(for: conversationId) { state in
                    state.currentTokenCount = 0
                    state.tokenBuffer.removeAll()
                }

                clearBatch()

                // If we have history, we need to warm up the context first
                if !historyTokens.isEmpty {
                    // Warm up with historical tokens
                    for (i, token) in historyTokens.enumerated() {
                        addToBatch(token: token, position: Int32(i), isLogit: false)
                    }

                    do {
                        try llamaContext.core().decode(batch!)
                    } catch {
                        return false
                    }

                    // Update state with historical tokens
                    updateConversationState(for: conversationId) { state in
                        state.currentTokenCount = Int32(historyTokens.count)
                        state.tokenBuffer = historyTokens
                    }
                }
            }

            // Now add the new prompt tokens
            clearBatch()

            let startPosition = currentState.currentTokenCount
            for (i, token) in promptTokens.enumerated() {
                let position = startPosition + Int32(i)
                addToBatch(token: token, position: position, isLogit: i == promptTokens.count - 1)
            }

            do {
                try llamaContext.core().decode(batch!)
            } catch {
                return false
            }

            // Update token count
            updateConversationState(for: conversationId) { state in
                state.currentTokenCount = startPosition + Int32(promptTokens.count)
                state.shouldContinuePredicting = true
            }

            return true
        } else {
            // New conversation - start fresh
            // Clear the KV cache completely for a brand-new conversation
            llamaContext.clearMemory(data: true)

            // Initialize conversation state
            updateConversationState(for: conversationId) { state in
                state.currentTokenCount = 0
                state.tokenBuffer.removeAll()
            }

            clearBatch()

            // For new conversation, start from position 0 since we cleared the cache
            for (i, token) in promptTokens.enumerated() {
                addToBatch(token: token, position: Int32(i), isLogit: i == promptTokens.count - 1)
            }

            do {
                try llamaContext.core().decode(batch!)
            } catch {
                return false
            }

            updateConversationState(for: conversationId) { state in
                state.currentTokenCount = Int32(promptTokens.count)
                state.shouldContinuePredicting = true
            }

            return true
        }
    }

    // MARK: - Context Preparation - for new conversations

    func prepareContext(for promptTokens: [SLlamaToken], llamaContext _: SLlamaContext) -> Bool {
        guard !promptTokens.isEmpty else { return false }

        let initialCount = promptTokens.count
        guard maxContextSize > initialCount else { return false }

        clearBatch()

        for (i, token) in promptTokens.enumerated() {
            addToBatch(token: token, position: Int32(i), isLogit: i == initialCount - 1)
        }

        do {
            try context!.core().decode(batch!)
        } catch {
            return false
        }

        return true
    }
}
