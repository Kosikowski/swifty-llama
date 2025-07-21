import Foundation
import Atomics
import SLlama

// MARK: - Core actor

public actor LlamaCoreActor {

    // MARK: - inner "streaming session" object
    private struct Session {
        let id: GenerationID
        let task: Task<Void, Never>
    }

    // MARK: - state

    private let model: SLlamaModel
    private let context: SLlamaContext
    private let core: PLlamaCore
    private let vocab: SLlamaVocab
    private var sessions: [GenerationID: Session] = [:]

    // MARK: - init / deinit

    /// Pass in the *already loaded* model context (one per actor).
    public init(modelPath: String, maxCtx: Int32 = 2048) throws {
        // Initialize SLlama backend
        SLlama.initialize()
        
        // Load model
        self.model = try SLlamaModel(modelPath: modelPath)
        
        // Create context with custom parameters using the Swift API
        let contextParams = SLlamaContext.createParams(
            contextSize: UInt32(maxCtx),
            batchSize: 512,
            physicalBatchSize: 512,
            maxSequences: 1,
            threads: 8,
            batchThreads: 8
        )
        
        self.context = try SLlamaContext(model: model, contextParams: contextParams)
        self.core = context.core()
        self.vocab = SLlamaVocab(vocab: model.vocab)
        
        // Configure context for generation
        context.setThreads(nThreads: 8, nThreadsBatch: 8)
        context.setEmbeddings(false)
        context.setCausalAttention(true)
    }

    deinit { 
        // SLlama cleanup is handled automatically by the context and model deinit
    }

    // MARK: - public surface

    /// Start generating tokens *immediately*.
    ///
    /// * The returned stream never throws directly – errors are reported by its yield.
    public func generate(id: GenerationID,
                         prompt: String,
                         params: GenerationParams)
        async throws -> AsyncThrowingStream<String, Error>
    {
        let (stream, cont) = AsyncThrowingStream<String, Error>.makeStream(
            bufferingPolicy: .unbounded
        )

        // Create sampler with current parameters
        let sampler = createSampler(with: params)

        // Start generation in a detached task
        let t = Task.detached(priority: .userInitiated) {
            await self.performGeneration(
                id: id,
                prompt: prompt,
                sampler: sampler,
                continuation: cont
            )
        }

        sessions[id] = Session(id: id, task: t)
        return stream
    }

    /// Co-operative cancellation.
    public func cancel(id: GenerationID) {
        sessions[id]?.task.cancel()
        sessions[id] = nil
    }

    // MARK: - private helpers

    /// Perform the actual generation within the actor context
    private func performGeneration(
        id: GenerationID,
        prompt: String,
        sampler: SLlamaSampler,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            // Tokenize the prompt
            let promptTokens = try SLlamaTokenizer.tokenize(
                text: prompt,
                vocab: vocab.pointer,
                addSpecial: true,
                parseSpecial: true
            )
            
            // Create batch for initial prompt processing
            let batch = SLlamaBatch(nTokens: Int32(promptTokens.count), nSeqMax: 1)
            
            // Add prompt tokens to batch
            for (index, token) in promptTokens.enumerated() {
                batch.addToken(
                    token,
                    position: Int32(index),
                    sequenceIds: [0],
                    logits: index == promptTokens.count - 1 // Only last token needs logits
                )
            }
            
            // Process the prompt
            try core.decode(batch)
            
            // Generate tokens
            var generatedTokens: [SLlamaToken] = []
            let maxTokens = 1000 // Safety limit
            
            for position in promptTokens.count..<(promptTokens.count + maxTokens) {
                // Check for cancellation
                if Task.isCancelled { break }
                
                // Sample next token
                guard let nextToken = sampler.sample() else {
                    continuation.finish(throwing: GenerationError.internalFailure("Sampling failed"))
                    break
                }
                
                // Check for end of generation
                if nextToken == vocab.eosToken {
                    break
                }
                
                // Convert token to text and yield
                if let tokenText = try? SLlamaTokenizer.tokenToPiece(
                    token: nextToken,
                    vocab: vocab.pointer,
                    lstrip: 0,
                    special: false
                ) {
                    continuation.yield(tokenText)
                }
                
                // Accept token for sampler state
                sampler.accept(nextToken)
                
                // Prepare next batch with single token
                batch.clear()
                batch.addToken(nextToken, position: Int32(position), sequenceIds: [0], logits: true)
                
                // Decode the new token
                try core.decode(batch)
                
                generatedTokens.append(nextToken)
            }
            
            continuation.finish()

        } catch is CancellationError {
            continuation.finish(throwing: GenerationError.abortedByUser)

        } catch {
            continuation.finish(throwing: error)
        }
    }

    /// Create a sampler with the given generation parameters
    private func createSampler(with params: GenerationParams) -> SLlamaSampler {
        // Create a simple temperature sampler for now
        // In a real implementation, you'd want to create a more sophisticated sampler chain
        guard let sampler = SLlamaSampler.temperature(
            context: context,
            temperature: params.temperature
        ) else {
            // Fallback to greedy sampler
            return SLlamaSampler.greedy(context: context) ?? SLlamaSampler(context: context)
        }
        
        return sampler
    }
}
