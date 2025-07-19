import Foundation
import llama

// MARK: - SLlamaInference

/// A wrapper for llama inference operations
public class SLlamaInference {
    // MARK: Properties

    private let context: SLlamaContext

    // MARK: Computed Properties

    /// Get the model from inference context
    public var inferenceModel: SLlamaModel? {
        guard let ctx = context.pointer else { return nil }
        let modelPtr = llama_get_model(ctx)
        return try? SLlamaModel(modelPointer: modelPtr)
    }

    // MARK: Lifecycle

    /// Initialize with a context
    /// - Parameter context: The llama context to use for inference
    public init(context: SLlamaContext) {
        self.context = context
    }

    // MARK: Functions

    /// Encode a batch of tokens (does not use KV cache)
    /// - Parameter batch: The batch to encode
    /// - Throws: SLlamaError if encoding fails
    public func encode(_ batch: SLlamaBatch) throws {
        guard let ctx = context.pointer else {
            throw SLlamaError.contextNotInitialized
        }

        let result = llama_encode(ctx, batch.cBatch)

        guard result == 0 else {
            switch result {
                case -1:
                    throw SLlamaError.invalidBatch("Invalid batch for encoding")
                case -2:
                    throw SLlamaError.contextFull
                case -3:
                    throw SLlamaError.outOfMemory
                default:
                    throw SLlamaError.batchOperationFailed("Encoding failed with error code: \(result)")
            }
        }
    }

    /// Decode a batch of tokens (uses KV cache)
    /// - Parameter batch: The batch to decode
    /// - Throws: SLlamaError if decoding fails
    public func decode(_ batch: SLlamaBatch) throws {
        guard let ctx = context.pointer else {
            throw SLlamaError.contextNotInitialized
        }

        let result = llama_decode(ctx, batch.cBatch)

        // Positive values are warnings, negative values are errors
        if result < 0 {
            switch result {
                case -1:
                    throw SLlamaError.invalidBatch("Invalid batch for decoding")
                case -2:
                    throw SLlamaError.contextFull
                case -3:
                    throw SLlamaError.outOfMemory
                default:
                    throw SLlamaError.batchOperationFailed("Decoding failed with error code: \(result)")
            }
        }
        // Note: Positive values are warnings and are not treated as errors
    }

    /// Legacy encode method that returns error code (deprecated)
    /// - Parameter batch: The batch to encode
    /// - Returns: 0 on success, negative value on error
    @available(*, deprecated, message: "Use encode(_:) throws instead")
    public func _encode(_ batch: SLlamaBatch) -> Int32 {
        do {
            try encode(batch)
            return 0
        } catch {
            return -1 // Generic error code for legacy compatibility
        }
    }

    /// Legacy decode method that returns error code (deprecated)
    /// - Parameter batch: The batch to decode
    /// - Returns: 0 on success, positive values are warnings, negative values are errors
    @available(*, deprecated, message: "Use decode(_:) throws instead")
    public func _decode(_ batch: SLlamaBatch) -> Int32 {
        do {
            try decode(batch)
            return 0
        } catch {
            return -1 // Generic error code for legacy compatibility
        }
    }

    /// Set the number of threads used for decoding
    /// - Parameters:
    ///   - nThreads: Number of threads for generation (single token)
    ///   - nThreadsBatch: Number of threads for prompt and batch processing (multiple tokens)
    public func setThreads(nThreads: Int32, nThreadsBatch: Int32) {
        guard let ctx = context.pointer else { return }
        llama_set_n_threads(ctx, nThreads, nThreadsBatch)
    }

    /// Get the number of threads used for generation
    /// - Returns: Number of threads for single token generation
    public func getThreads() -> Int32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_threads(ctx)
    }

    /// Get the number of threads used for batch processing
    /// - Returns: Number of threads for multiple token processing
    public func getThreadsBatch() -> Int32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_threads_batch(ctx)
    }

    /// Set whether the context outputs embeddings
    /// - Parameter embeddings: Whether to output embeddings
    public func setEmbeddings(_ embeddings: Bool) {
        guard let ctx = context.pointer else { return }
        llama_set_embeddings(ctx, embeddings)
    }

    /// Set whether to use causal attention
    /// - Parameter causalAttn: Whether to use causal attention
    public func setCausalAttention(_ causalAttn: Bool) {
        guard let ctx = context.pointer else { return }
        llama_set_causal_attn(ctx, causalAttn)
    }

    /// Set whether the model is in warmup mode
    /// - Parameter warmup: Whether to enable warmup mode
    public func setWarmup(_ warmup: Bool) {
        guard let ctx = context.pointer else { return }
        llama_set_warmup(ctx, warmup)
    }

    /// Set abort callback
    /// - Parameters:
    ///   - callback: The abort callback function
    ///   - userData: User data passed to the callback
    public func setAbortCallback(_: @escaping () -> Bool, userData: SLlamaRawPointer? = nil) {
        guard let ctx = context.pointer else { return }

        // Note: This is a simplified implementation that doesn't capture the callback
        // In a production implementation, you would need a more sophisticated callback management system
        // For now, we'll just pass nil to avoid concurrency issues
        llama_set_abort_callback(ctx, nil, userData)
    }

    /// Wait until all computations are finished
    public func synchronize() {
        guard let ctx = context.pointer else { return }
        llama_synchronize(ctx)
    }

    /// Get context size
    /// - Returns: The context size
    public func getContextSize() -> UInt32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_ctx(ctx)
    }

    /// Get batch size
    /// - Returns: The logical maximum batch size
    public func getBatchSize() -> UInt32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_batch(ctx)
    }

    /// Get unified batch size
    /// - Returns: The physical maximum batch size
    public func getUnifiedBatchSize() -> UInt32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_ubatch(ctx)
    }

    /// Get maximum sequences
    /// - Returns: The maximum number of sequences
    public func getMaxSequences() -> UInt32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_seq_max(ctx)
    }

    /// Get memory
    /// - Returns: The memory associated with this context
    public func getMemory() -> SLlamaMemory? {
        guard let ctx = context.pointer else { return nil }
        return llama_get_memory(ctx)
    }

    /// Get pooling type
    /// - Returns: The pooling type used by this context
    public func getPoolingType() -> SLlamaPoolingType {
        guard let ctx = context.pointer else { return .unspecified }
        return llama_pooling_type(ctx)
    }
}

/// Extension to SLlamaContext for inference operations
public extension SLlamaContext {
    /// Create an inference wrapper for this context
    /// - Returns: A SLlamaInference instance
    func inference() -> SLlamaInference {
        SLlamaInference(context: self)
    }

    /// Encode a batch of tokens (does not use KV cache)
    /// - Parameter batch: The batch to encode
    /// - Throws: SLlamaError if encoding fails
    func encode(_ batch: SLlamaBatch) throws {
        try inference().encode(batch)
    }

    /// Decode a batch of tokens (uses KV cache)
    /// - Parameter batch: The batch to decode
    /// - Throws: SLlamaError if decoding fails
    func decode(_ batch: SLlamaBatch) throws {
        try inference().decode(batch)
    }

    /// Set the number of threads
    /// - Parameters:
    ///   - nThreads: Number of threads for generation
    ///   - nThreadsBatch: Number of threads for batch processing
    func setThreads(nThreads: Int32, nThreadsBatch: Int32) {
        inference().setThreads(nThreads: nThreads, nThreadsBatch: nThreadsBatch)
    }

    /// Set embeddings output
    /// - Parameter embeddings: Whether to output embeddings
    func setEmbeddings(_ embeddings: Bool) {
        inference().setEmbeddings(embeddings)
    }

    /// Set causal attention
    /// - Parameter causalAttn: Whether to use causal attention
    func setCausalAttention(_ causalAttn: Bool) {
        inference().setCausalAttention(causalAttn)
    }

    /// Set warmup mode
    /// - Parameter warmup: Whether to enable warmup mode
    func setWarmup(_ warmup: Bool) {
        inference().setWarmup(warmup)
    }

    /// Synchronize computations
    func synchronize() {
        inference().synchronize()
    }

    // MARK: - Legacy Methods (Deprecated)

    /// Legacy encode method that returns error code (deprecated)
    /// - Parameter batch: The batch to encode
    /// - Returns: 0 on success, negative value on error
    @available(*, deprecated, message: "Use encode(_:) throws instead")
    func _encode(_ batch: SLlamaBatch) -> Int32 {
        inference()._encode(batch)
    }

    /// Legacy decode method that returns error code (deprecated)
    /// - Parameter batch: The batch to decode
    /// - Returns: 0 on success, positive values are warnings, negative values are errors
    @available(*, deprecated, message: "Use decode(_:) throws instead")
    func _decode(_ batch: SLlamaBatch) -> Int32 {
        inference()._decode(batch)
    }
}
