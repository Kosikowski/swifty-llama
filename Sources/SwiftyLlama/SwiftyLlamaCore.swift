import Atomics
import Foundation
import SLlama

@SLlamaActor
public class SwiftyLlamaCore: GenerationCore {
    private struct Session {
        let id: GenerationID
        let task: Task<Void, Never>
    }

    // MARK: - state

    private let model: SLlamaModel
    private let vocab: SLlamaVocab
    private let maxContextSize: Int32
    private var sessions: [GenerationID: Session] = [:]

    // MARK: - init / deinit

    /// Pass in the *already loaded* model context (one per actor).
    public init(modelPath: String, maxCtx: Int32 = 2048) throws {
        // Initialize SLlama backend
        SLlama.initialize()

        // Load model
        model = try SLlamaModel(modelPath: modelPath)
        vocab = SLlamaVocab(vocab: model.vocab)
        maxContextSize = maxCtx
    }

    deinit {
        // SLlama cleanup is handled automatically by the context and model deinit
    }

    // MARK: - public surface

    /// Start generating tokens *immediately*.
    ///
    /// * The returned stream never throws directly – errors are reported by its yield.
    public func generate(
        id: GenerationID,
        prompt: String,
        params: GenerationParams
    )
        async throws -> AsyncThrowingStream<String, Error>
    {
        let (stream, cont) = AsyncThrowingStream<String, Error>.makeStream(
            bufferingPolicy: .unbounded
        )

        // Start generation in a detached task
        let t = Task.detached(priority: .userInitiated) {
            await self.performGeneration(
                id: id,
                prompt: prompt,
                params: params,
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
        id _: GenerationID,
        prompt: String,
        params: GenerationParams,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            // Create a fresh context for this generation to avoid state conflicts
            let contextParams = SLlamaContext.createParams(
                contextSize: UInt32(maxContextSize), // Use the configured context size
                batchSize: 256, // Reduced batch size for better memory management
                physicalBatchSize: 256,
                maxSequences: 1,
                threads: 8,
                batchThreads: 8
            )

            let context = try SLlamaContext(model: model, contextParams: contextParams)
            let core = context.core()

            // Configure the context
            context.setThreads(nThreads: 8, nThreadsBatch: 8)
            context.setEmbeddings(false)
            context.setCausalAttention(true)

            // Create sampler for this context
            let sampler = createSampler(with: params, context: context)

            // Tokenize the prompt
            let promptTokens = try vocab.tokenize(text: prompt)
            
            // Check for empty prompt
            guard !promptTokens.isEmpty else {
                continuation.finish()
                return
            }

            // Process prompt tokens in chunks to avoid exceeding batch size
            let maxBatchSize = 256 // Reduced to match context batch size
            var position = 0

            for batchStart in stride(from: 0, to: promptTokens.count, by: maxBatchSize) {
                let batchEnd = min(batchStart + maxBatchSize, promptTokens.count)
                let currentBatchSize = batchEnd - batchStart

                let batch = SLlamaBatch(nTokens: Int32(currentBatchSize), nSeqMax: 1)

                // Add tokens for this batch
                for (localIndex, token) in promptTokens[batchStart ..< batchEnd].enumerated() {
                    let globalIndex = batchStart + localIndex
                    batch.addToken(
                        token,
                        position: Int32(position + localIndex),
                        sequenceIds: [0],
                        logits: globalIndex == promptTokens.count - 1 // Only last token needs logits
                    )
                }

                // Process this batch
                try core.decode(batch)
                position += currentBatchSize
            }

            // Generate tokens
            var generatedTokens: [SLlamaToken] = []
            let maxTokens = 1000 // Safety limit
            var currentPosition = promptTokens.count // Start from where we left off

            for _ in 0 ..< maxTokens {
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
                let generationBatch = SLlamaBatch(nTokens: 1, nSeqMax: 1)
                generationBatch.addToken(nextToken, position: Int32(currentPosition), sequenceIds: [0], logits: true)

                // Decode the new token
                try core.decode(generationBatch)

                generatedTokens.append(nextToken)
                currentPosition += 1
            }

            continuation.finish()

        } catch is CancellationError {
            continuation.finish(throwing: GenerationError.abortedByUser)

        } catch {
            continuation.finish(throwing: error)
        }
    }

    /// Create a sampler with the given generation parameters
    private func createSampler(with params: GenerationParams, context: SLlamaContext) -> SLlamaSampler {
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
