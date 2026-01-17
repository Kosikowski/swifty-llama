import Foundation
import SLlama

// MARK: - Generation Extension

public extension SwiftyLlamaCore {
    // MARK: - Public Generation API

    /// Begin a new generation and immediately obtain a token stream
    /// If no conversation is active, starts a new one
    @discardableResult
    func start(
        prompt: String,
        params: GenerationParams,
        conversationId: ConversationID?
    ) async
        -> GenerationStream
    {
        let id = GenerationID()

        // Determine conversation to use
        let targetConversationId: ConversationID = if let conversationId {
            conversationId
        } else if let currentId = currentConversationId {
            currentId
        } else {
            startNewConversation()
        }

        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream(
            bufferingPolicy: .unbounded
        )

        // Start generation task within the same actor context
        let task = Task { @SwiftyLlamaActor in
            await self.performGeneration(
                id: id,
                prompt: prompt,
                params: params,
                conversationId: targetConversationId,
                continuation: continuation
            )
            return ()
        }

        // Store generation info with the actual task
        activeGenerations[id] = LiveGeneration(
            id: id,
            conversationId: targetConversationId,
            params: params,
            startTime: Date(),
            task: task
        )

        // Set up termination callback
        continuation.onTermination = { @Sendable _ in
            Task { await self.cancel(id) }
        }

        return GenerationStream(id: id, stream: stream)
    }

    // MARK: - Private Generation Logic

    internal func performGeneration(
        id: GenerationID,
        prompt: String,
        params: GenerationParams,
        conversationId: ConversationID,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            // Ensure context is initialized (only once per instance)
            try ensureContextInitialized()

            // Configure context for this generation (only if not already warmed up)
            let state = getOrCreateConversationState(for: conversationId)
            if state.currentTokenCount == 0 {
                try configureContext(with: params)
            }

            // Tokenize prompt
            let promptTokens = try vocab.tokenize(text: prompt)

            // Prepare context with conversation history
            guard prepareContextWithConversation(
                for: promptTokens,
                conversationId: conversationId,
                llamaContext: context!
            ) else {
                continuation.finish(throwing: GenerationError.contextPreparationFailed(conversationId: conversationId))
                return
            }

            // Recreate sampler chain with updated context
            guard let sampler = createSampler(with: params, llamaContext: context!) else {
                continuation.finish(throwing: GenerationError.samplerInitializationFailed)
                return
            }
            // Note: No need to reset a freshly created chain

            // Generate tokens
            var generatedTokens: [SLlamaToken] = []
            // maxTokens limits the response length, not the total context
            let maxNewTokens = Int(params.maxTokens)

            // Update conversation state for generation
            updateConversationState(for: conversationId) { state in
                state.shouldContinuePredicting = true
                state.lastUsedParams = params
                state.isActive = true
            }

            // Buffer for incomplete UTF-8 sequences (byte-level BPE tokenizers may
            // produce tokens that represent partial multi-byte UTF-8 characters)
            var utf8Buffer: [UInt8] = []

            while getOrCreateConversationState(for: conversationId).shouldContinuePredicting,
                  generatedTokens.count < maxNewTokens
            {
                let token = predictNextToken(sampler: sampler, llamaContext: context!, conversationId: conversationId)

                if token == vocab.eosToken {
                    break
                }

                generatedTokens.append(token)

                // Get raw bytes for this token and add to buffer
                let tokenBytes = try vocab.tokenToBytes(token: token)
                utf8Buffer.append(contentsOf: tokenBytes)

                // Try to decode complete UTF-8 characters from the buffer
                let (decodedText, remainingBytes) = decodeCompleteUTF8(from: utf8Buffer)
                utf8Buffer = remainingBytes

                if !decodedText.isEmpty {
                    continuation.yield(decodedText)
                }
            }

            // Flush any remaining bytes in the buffer (use replacement character for invalid sequences)
            if !utf8Buffer.isEmpty {
                let finalText = String(decoding: utf8Buffer, as: UTF8.self)
                if !finalText.isEmpty {
                    continuation.yield(finalText)
                }
            }

            // Store the conversation
            await storeConversationMessage(
                conversationId: conversationId,
                role: "user",
                content: prompt,
                tokens: promptTokens
            )

            if !generatedTokens.isEmpty {
                // Use batch detokenization which properly handles multi-byte UTF-8 sequences
                let responseContent = try vocab.detokenize(tokens: generatedTokens)
                await storeConversationMessage(
                    conversationId: conversationId,
                    role: "assistant",
                    content: responseContent,
                    tokens: generatedTokens
                )
            }

            continuation.finish()

        } catch {
            // Convert specific errors to GenerationError
            let generationError: GenerationError = if error is GenerationError {
                error as! GenerationError
            } else {
                // Map underlying errors to appropriate GenerationError cases
                switch error {
                    case _ where error.localizedDescription.contains("tokenize"):
                        .tokenizationFailed(conversationId: conversationId)
                    case _ where error.localizedDescription.contains("context"):
                        .contextPreparationFailed(conversationId: conversationId)
                    default:
                        .generationFailed(generationId: id)
                }
            }
            continuation.finish(throwing: generationError)
        }
    }

    // MARK: - Token Prediction

    internal func predictNextToken(
        sampler: SLlamaSamplerChain,
        llamaContext _: SLlamaContext,
        conversationId: ConversationID
    )
        -> SLlamaToken
    {
        let state = getOrCreateConversationState(for: conversationId)

        guard state.shouldContinuePredicting, state.currentTokenCount < maxContextSize else {
            return vocab.eosToken
        }

        // Sample next token
        guard let token = sampler.sample() else {
            updateConversationState(for: conversationId) { state in
                state.shouldContinuePredicting = false
            }
            return vocab.eosToken
        }

        // Accept the token
        sampler.accept(token)

        updateConversationState(for: conversationId) { state in
            state.tokenBuffer.append(token)
        }

        // Check for stop conditions
        if token == vocab.eosToken {
            updateConversationState(for: conversationId) { state in
                state.shouldContinuePredicting = false
            }
            return vocab.eosToken
        }

        // Process the token for next generation
        clearBatch()
        addToBatch(token: token, position: state.currentTokenCount)

        do {
            try context!.core().decode(batch!)
        } catch {
            updateConversationState(for: conversationId) { state in
                state.shouldContinuePredicting = false
            }
            return vocab.eosToken
        }

        updateConversationState(for: conversationId) { state in
            state.currentTokenCount += 1
        }

        return token
    }

    // MARK: - Sampler Creation

    internal func createSampler(with params: GenerationParams, llamaContext: SLlamaContext) -> SLlamaSamplerChain? {
        // If grammar constraint is specified, use grammar-enabled chain
        if let grammar = params.grammar {
            return SLlamaSamplerChain.withGrammar(
                context: llamaContext,
                grammarRules: grammar.rules,
                grammarRoot: grammar.rootSymbol,
                temperature: params.temperature,
                topK: params.topK,
                topP: params.topP,
                repetitionPenalty: params.repeatPenalty,
                seed: params.seed
            )
        }

        // Create a proper sampler chain with all sampling stages:
        // 1. Repetition penalty
        // 2. Top-K filtering
        // 3. Top-P filtering
        // 4. Temperature scaling
        // 5. Distribution sampler (REQUIRED to actually select a token)
        return SLlamaSamplerChain.custom(
            context: llamaContext,
            temperature: params.temperature,
            topK: params.topK,
            topP: params.topP,
            repetitionPenalty: params.repeatPenalty,
            seed: params.seed
        )
    }

    // MARK: - UTF-8 Streaming Decoder

    /// Decode complete UTF-8 characters from a byte buffer.
    /// Returns the decoded string and any remaining incomplete bytes.
    /// This handles byte-level BPE tokenizers where tokens may represent
    /// partial multi-byte UTF-8 sequences.
    internal func decodeCompleteUTF8(from bytes: [UInt8]) -> (decoded: String, remaining: [UInt8]) {
        guard !bytes.isEmpty else {
            return ("", [])
        }

        // Find the last position where we can safely decode UTF-8
        // UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
        // UTF-8 leading bytes start with 0xxxxxxx (ASCII), 110xxxxx, 1110xxxx, or 11110xxx
        var safeEndIndex = bytes.count

        // Check the last few bytes to see if they're part of an incomplete sequence
        for i in stride(from: bytes.count - 1, through: max(0, bytes.count - 4), by: -1) {
            let byte = bytes[i]

            // If this is a leading byte (not a continuation byte), check if the sequence is complete
            if byte & 0b1100_0000 != 0b1000_0000 {
                // This is a leading byte
                let expectedLength: Int
                if byte & 0b1000_0000 == 0 {
                    expectedLength = 1 // ASCII
                } else if byte & 0b1110_0000 == 0b1100_0000 {
                    expectedLength = 2
                } else if byte & 0b1111_0000 == 0b1110_0000 {
                    expectedLength = 3
                } else if byte & 0b1111_1000 == 0b1111_0000 {
                    expectedLength = 4
                } else {
                    // Invalid leading byte - include it and let String handle it
                    break
                }

                let actualLength = bytes.count - i
                if actualLength < expectedLength {
                    // Incomplete sequence - exclude it from decoding
                    safeEndIndex = i
                }
                break
            }
        }

        let safeToDecode = Array(bytes.prefix(safeEndIndex))
        let remaining = Array(bytes.suffix(from: safeEndIndex))

        let decoded = String(decoding: safeToDecode, as: UTF8.self)
        return (decoded, remaining)
    }

    // MARK: - Generation Management

    func update(id: GenerationID, _ params: GenerationParams) async {
        if var generation = activeGenerations[id] {
            generation.params = params
            activeGenerations[id] = generation
        }
    }

    func cancel(_ id: GenerationID) async {
        if let generation = activeGenerations[id] {
            generation.task?.cancel()
            activeGenerations.removeValue(forKey: id)
        }
    }

    func getGenerationInfo(_ id: GenerationID) async -> GenerationInfo? {
        guard let generation = activeGenerations[id] else { return nil }

        return GenerationInfo(
            id: generation.id,
            conversationId: generation.conversationId,
            params: generation.params,
            startTime: generation.startTime,
            isActive: generation.task?.isCancelled == false
        )
    }

    func getActiveGenerationIDs() async -> [GenerationID] {
        Array(activeGenerations.keys)
    }

    func cancelAll() async {
        for (_, generation) in activeGenerations {
            generation.task?.cancel()
        }
        activeGenerations.removeAll()
    }
}
