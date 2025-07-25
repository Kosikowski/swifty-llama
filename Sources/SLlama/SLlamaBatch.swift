import Foundation
import llama

/// A wrapper for llama batch operations
public class SLlamaBatch: @unchecked Sendable, PLlamaBatch {
    // MARK: Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    var batch: llama_batch
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    let maxTokens: Int32

    // MARK: Computed Properties

    /// Get the underlying C batch structure for direct API access
    public var cBatch: llama_batch {
        batch
    }

    /// Number of tokens in the batch
    public var tokenCount: Int32 {
        batch.n_tokens
    }

    /// Token array (allocated if embd == 0 in init)
    public var tokens: SLlamaTokenPointer? {
        batch.token
    }

    /// Embeddings array (allocated if embd != 0 in init)
    public var embeddings: SLlamaFloatPointer? {
        batch.embd
    }

    /// Position array for each token
    public var positions: SLlamaPositionPointer? {
        batch.pos
    }

    /// Number of sequence IDs for each token
    public var sequenceIdCounts: SLlamaInt32Pointer? {
        batch.n_seq_id
    }

    /// Sequence ID arrays for each token
    public var sequenceIds: SLlamaSeqIdPointerPointer? {
        batch.seq_id
    }

    /// Logits output flags for each token
    public var logits: SLlamaInt8Pointer? {
        batch.logits
    }

    // MARK: Lifecycle

    /// Initialize a batch with the specified parameters
    /// - Parameters:
    ///   - nTokens: Maximum number of tokens the batch can hold
    ///   - embd: If non-zero, allocates embeddings array of size nTokens * embd * sizeof(float)
    ///   - nSeqMax: Maximum number of sequence IDs per token
    public init(nTokens: Int32, embd: Int32 = 0, nSeqMax: Int32) {
        maxTokens = nTokens
        batch = llama_batch_init(nTokens, embd, nSeqMax)
    }

    deinit {
        llama_batch_free(batch)
    }

    /// Private initializer for creating from C batch
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    init(batch: llama_batch, maxTokens: Int32) {
        self.batch = batch
        self.maxTokens = maxTokens
    }

    // MARK: Static Functions

    /// Create a batch with a single token
    /// - Parameter token: The token ID
    /// - Returns: A batch containing the single token
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func single(token: SLlamaToken) -> SLlamaBatch {
        var tokenCopy = token
        let batch = llama_batch_get_one(&tokenCopy, 1)
        return SLlamaBatch(batch: batch, maxTokens: 1)
    }

    /// Create a batch with multiple tokens
    /// - Parameter tokens: Array of token IDs
    /// - Returns: A batch containing the tokens
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func fromTokens(_ tokens: [SLlamaToken]) -> SLlamaBatch {
        let batch = SLlamaBatch(nTokens: Int32(tokens.count), nSeqMax: 1)
        batch.setTokens(tokens)
        return batch
    }

    /// Create a batch with embeddings
    /// - Parameters:
    ///   - embeddings: Array of embedding values
    ///   - embeddingSize: Size of each embedding vector
    /// - Returns: A batch containing the embeddings
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func fromEmbeddings(_ embeddings: [Float], embeddingSize: Int32) -> SLlamaBatch {
        let batch = SLlamaBatch(nTokens: Int32(embeddings.count / Int(embeddingSize)), embd: embeddingSize, nSeqMax: 1)
        batch.setEmbeddings(embeddings)
        return batch
    }

    // MARK: Functions

    /// Set tokens for the batch
    /// - Parameter tokens: Array of token IDs
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setTokens(_ tokens: [SLlamaToken]) {
        guard tokens.count <= tokenCount else {
            fatalError("Too many tokens for batch capacity")
        }

        batch.n_tokens = Int32(tokens.count)
        tokens.withUnsafeBufferPointer { buffer in
            batch.token?.update(from: buffer.baseAddress!, count: tokens.count)
        }
    }

    /// Set embeddings for the batch
    /// - Parameter embeddings: Array of embedding values
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setEmbeddings(_ embeddings: [Float]) {
        guard embeddings.count <= tokenCount * (batch.embd != nil ? 1 : 0) else {
            fatalError("Too many embeddings for batch capacity")
        }

        embeddings.withUnsafeBufferPointer { buffer in
            batch.embd?.update(from: buffer.baseAddress!, count: embeddings.count)
        }
    }

    /// Set positions for the batch
    /// - Parameter positions: Array of position values
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setPositions(_ positions: [SLlamaPosition]) {
        guard positions.count <= tokenCount else {
            fatalError("Too many positions for batch capacity")
        }

        positions.withUnsafeBufferPointer { buffer in
            batch.pos?.update(from: buffer.baseAddress!, count: positions.count)
        }
    }

    /// Set logits output flags for the batch
    /// - Parameter logits: Array of logits flags (0 = no output, 1 = output)
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setLogits(_ logits: [Int8]) {
        guard logits.count <= tokenCount else {
            fatalError("Too many logits flags for batch capacity")
        }

        logits.withUnsafeBufferPointer { buffer in
            batch.logits?.update(from: buffer.baseAddress!, count: logits.count)
        }
    }

    /// Set sequence IDs for a specific token
    /// - Parameters:
    ///   - tokenIndex: Index of the token
    ///   - sequenceIds: Array of sequence IDs for this token
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setSequenceIds(for tokenIndex: Int, sequenceIds: [SLlamaSeqId]) {
        guard tokenIndex < tokenCount else {
            fatalError("Token index out of bounds")
        }

        batch.n_seq_id?[tokenIndex] = Int32(sequenceIds.count)
        batch.seq_id?[tokenIndex]?.update(from: sequenceIds, count: sequenceIds.count)
    }

    // MARK: - PLlamaBatch Protocol Methods

    /// Add a token to the batch
    /// - Parameters:
    ///   - token: The token to add
    ///   - position: Position of the token in the sequence
    ///   - sequenceIds: Array of sequence IDs for this token
    ///   - logits: Whether to output logits for this token
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func addToken(
        _ token: SLlamaToken,
        position: SLlamaPosition,
        sequenceIds: [SLlamaSequenceId],
        logits: Bool
    ) {
        let currentIndex = Int(batch.n_tokens)

        // Ensure we don't exceed capacity
        guard currentIndex < maxTokens else {
            fatalError("Batch capacity exceeded")
        }

        // Add token
        batch.token?[currentIndex] = token

        // Add position
        batch.pos?[currentIndex] = position

        // Add sequence IDs
        batch.n_seq_id?[currentIndex] = Int32(sequenceIds.count)
        for (i, seqId) in sequenceIds.enumerated() {
            batch.seq_id?[currentIndex]?[i] = seqId
        }

        // Add logits flag
        batch.logits?[currentIndex] = logits ? 1 : 0

        // Increment token count
        batch.n_tokens += 1
    }

    /// Clear the batch (reset token count to 0)
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func clear() {
        batch.n_tokens = 0
    }

    /// Get batch with single token
    /// - Parameter token: The token to create batch for
    /// - Returns: A batch containing the single token
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getSingleTokenBatch(_ token: SLlamaToken) -> PLlamaBatch {
        single(token: token)
    }
}
