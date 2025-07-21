import Foundation
import llama

// MARK: - SLlamaLogits

/// A wrapper for llama logits and embeddings access
public class SLlamaLogits: @unchecked Sendable {
    // MARK: Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    let context: SLlamaContext

    // MARK: Lifecycle

    /// Initialize with a context
    /// - Parameter context: The llama context to use for logits access
    public init(context: SLlamaContext) {
        self.context = context
    }

    // MARK: Functions

    /// Get logits from the last decode call
    /// - Returns: Pointer to logits array, or nil if not available
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getLogits() -> SLlamaFloatPointer? {
        guard let ctx = context.pointer else { return nil }
        return llama_get_logits(ctx)
    }

    /// Get logits for a specific token index
    /// - Parameter index: Token index (negative for reverse order, -1 is last)
    /// - Returns: Pointer to logits for the token, or nil if invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getLogits(for index: Int32) -> SLlamaFloatPointer? {
        guard let ctx = context.pointer else { return nil }
        return llama_get_logits_ith(ctx, index)
    }

    /// Get embeddings from the last decode call
    /// - Returns: Pointer to embeddings array, or nil if not available
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getEmbeddings() -> SLlamaFloatPointer? {
        guard let ctx = context.pointer else { return nil }
        return llama_get_embeddings(ctx)
    }

    /// Get embeddings for a specific token index
    /// - Parameter index: Token index (negative for reverse order, -1 is last)
    /// - Returns: Pointer to embeddings for the token, or nil if invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getEmbeddings(for index: Int32) -> SLlamaFloatPointer? {
        guard let ctx = context.pointer else { return nil }
        return llama_get_embeddings_ith(ctx, index)
    }

    /// Get embeddings for a specific sequence
    /// - Parameter sequenceId: The sequence ID
    /// - Returns: Pointer to embeddings for the sequence, or nil if not available
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getEmbeddingsForSequence(_ sequenceId: SLlamaSeqId) -> SLlamaFloatPointer? {
        guard let ctx = context.pointer else { return nil }
        return llama_get_embeddings_seq(ctx, sequenceId)
    }

    /// Get logits as an array for a specific token
    /// - Parameter index: Token index (negative for reverse order, -1 is last)
    /// - Returns: Array of logits for the token, or nil if invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getLogitsArray(for index: Int32) -> [Float]? {
        guard let logitsPtr = getLogits(for: index) else { return nil }

        // Get vocabulary size from the model
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0

        if vocabSize > 0 {
            return Array(UnsafeBufferPointer(start: logitsPtr, count: Int(vocabSize)))
        }

        return nil
    }

    /// Get embeddings as an array for a specific token
    /// - Parameter index: Token index (negative for reverse order, -1 is last)
    /// - Returns: Array of embeddings for the token, or nil if invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getEmbeddingsArray(for index: Int32) -> [Float]? {
        guard let embeddingsPtr = getEmbeddings(for: index) else { return nil }

        // Get embedding dimensions from the model
        guard let model = context.associatedModel else { return nil }
        let embeddingDim = model.embeddingDimensions

        if embeddingDim > 0 {
            return Array(UnsafeBufferPointer(start: embeddingsPtr, count: Int(embeddingDim)))
        }

        return nil
    }

    /// Get embeddings as an array for a specific sequence
    /// - Parameter sequenceId: The sequence ID
    /// - Returns: Array of embeddings for the sequence, or nil if not available
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getEmbeddingsArrayForSequence(_ sequenceId: SLlamaSeqId) -> [Float]? {
        guard let embeddingsPtr = getEmbeddingsForSequence(sequenceId) else { return nil }

        // Get embedding dimensions from the model
        guard let model = context.associatedModel else { return nil }
        let embeddingDim = model.embeddingDimensions

        if embeddingDim > 0 {
            return Array(UnsafeBufferPointer(start: embeddingsPtr, count: Int(embeddingDim)))
        }

        return nil
    }

    /// Get the token with highest logit for a specific token index
    /// - Parameter index: Token index (negative for reverse order, -1 is last)
    /// - Returns: The token ID with highest logit, or nil if invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getMostLikelyToken(for index: Int32) -> SLlamaToken? {
        guard let logitsArray = getLogitsArray(for: index) else { return nil }
        guard let maxIndex = logitsArray.enumerated().max(by: { $0.element < $1.element })?.offset else { return nil }
        return SLlamaToken(maxIndex)
    }

    /// Get the top-k tokens with highest logits for a specific token index
    /// - Parameters:
    ///   - index: Token index (negative for reverse order, -1 is last)
    ///   - k: Number of top tokens to return
    /// - Returns: Array of (token, logit) pairs sorted by logit value, or nil if invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getTopKTokens(for index: Int32, k: Int) -> [(token: SLlamaToken, logit: Float)]? {
        guard let logitsArray = getLogitsArray(for: index) else { return nil }
        let sortedIndices = logitsArray.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(k)
            .map { (SLlamaToken($0.offset), $0.element) }
        return Array(sortedIndices)
    }
}

/// Extension to SLlamaContext for logits access
public extension SLlamaContext {
    /// Create a logits wrapper for this context
    /// - Returns: A SLlamaLogits instance
    func logits() -> SLlamaLogits {
        SLlamaLogits(context: self)
    }

    /// Get logits from the last decode call
    /// - Returns: Pointer to logits array, or nil if not available
    func getLogits() -> SLlamaFloatPointer? {
        logits().getLogits()
    }

    /// Get logits for a specific token index
    /// - Parameter index: Token index (negative for reverse order, -1 is last)
    /// - Returns: Pointer to logits for the token, or nil if invalid
    func getLogits(for index: Int32) -> SLlamaFloatPointer? {
        logits().getLogits(for: index)
    }

    /// Get embeddings from the last decode call
    /// - Returns: Pointer to embeddings array, or nil if not available
    func getEmbeddings() -> SLlamaFloatPointer? {
        logits().getEmbeddings()
    }

    /// Get embeddings for a specific token index
    /// - Parameter index: Token index (negative for reverse order, -1 is last)
    /// - Returns: Pointer to embeddings for the token, or nil if invalid
    func getEmbeddings(for index: Int32) -> SLlamaFloatPointer? {
        logits().getEmbeddings(for: index)
    }

    /// Get embeddings for a specific sequence
    /// - Parameter sequenceId: The sequence ID
    /// - Returns: Pointer to embeddings for the sequence, or nil if not available
    func getEmbeddingsForSequence(_ sequenceId: SLlamaSeqId) -> SLlamaFloatPointer? {
        logits().getEmbeddingsForSequence(sequenceId)
    }

    /// Get the token with highest logit for a specific token index
    /// - Parameter index: Token index (negative for reverse order, -1 is last)
    /// - Returns: The token ID with highest logit, or nil if invalid
    func getMostLikelyToken(for index: Int32) -> SLlamaToken? {
        logits().getMostLikelyToken(for: index)
    }

    /// Get the top-k tokens with highest logits for a specific token index
    /// - Parameters:
    ///   - index: Token index (negative for reverse order, -1 is last)
    ///   - k: Number of top tokens to return
    /// - Returns: Array of (token, logit) pairs sorted by logit value, or nil if invalid
    func getTopKTokens(for index: Int32, k: Int) -> [(token: SLlamaToken, logit: Float)]? {
        logits().getTopKTokens(for: index, k: k)
    }
}
