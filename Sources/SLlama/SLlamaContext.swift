import Foundation
import llama

/// A wrapper for llama context
public class SLlamaContext: @unchecked Sendable, PLlamaContext {
    // MARK: Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    var context: SLlamaContextPointer?
    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    var model: PLlamaModel?

    // MARK: Computed Properties

    /// Get the context pointer for direct C API access
    public var pointer: SLlamaContextPointer? {
        context
    }

    /// Get the associated model
    public var associatedModel: PLlamaModel? {
        model
    }

    /// Get context size
    public var contextSize: UInt32 {
        guard let context else { return 0 }
        return llama_n_ctx(context)
    }

    /// Get batch size
    public var batchSize: UInt32 {
        guard let context else { return 0 }
        return llama_n_batch(context)
    }

    /// Get maximum batch size
    public var maxBatchSize: UInt32 {
        guard let context else { return 0 }
        return llama_n_ubatch(context)
    }

    /// Get maximum sequence count
    public var maxSequences: UInt32 {
        guard let context else { return 0 }
        return llama_n_seq_max(context)
    }

    /// Get the model from context
    public var contextModel: PLlamaModel? {
        guard let context else { return nil }
        let modelPtr = llama_get_model(context)
        return try? SLlamaModel(modelPointer: modelPtr)
    }

    /// Get memory from context
    public var contextMemory: SLlamaMemory? {
        guard let context else { return nil }
        return llama_get_memory(context)
    }

    /// Get pooling type from context
    public var poolingType: SLlamaPoolingType {
        guard let context else { return .unspecified }
        return llama_pooling_type(context)
    }

    // MARK: Lifecycle

    /// Initialize a new llama context from a model
    /// - Parameters:
    ///   - model: The model to create context from
    ///   - contextParams: Optional context parameters (uses defaults if nil)
    /// - Throws: SLlamaError if context creation fails
    public init(model: PLlamaModel, contextParams: SLlamaContextParams? = nil) throws {
        // Validate model
        guard let modelPtr = model.pointer else {
            throw SLlamaError.invalidModel("Model pointer is null")
        }

        let params = contextParams ?? llama_context_default_params()

        // Validate context parameters
        if params.n_ctx == 0 {
            throw SLlamaError.invalidParameters("Context size cannot be zero")
        }

        if params.n_batch == 0 {
            throw SLlamaError.invalidParameters("Batch size cannot be zero")
        }

        // Create context first, before setting model reference
        context = llama_init_from_model(modelPtr, params)

        guard context != nil else {
            // Try to provide more specific error information
            let modelSize = model.size
            let requestedContextSize = params.n_ctx
            let availableMemory = ProcessInfo.processInfo.physicalMemory

            if modelSize > availableMemory / 2 {
                throw SLlamaError.outOfMemory
            } else if requestedContextSize > 32768 {
                throw SLlamaError
                    .invalidParameters("Context size (\(requestedContextSize)) may be too large for this model")
            } else {
                throw SLlamaError
                    .contextCreationFailed("Failed to create context with size \(requestedContextSize) for model")
            }
        }

        // Only set model reference after successful context creation
        // This prevents retain cycles when initialization fails
        self.model = model
    }

    deinit {
        if let context {
            llama_free(context)
        }
    }

    // MARK: Static Functions

    /// Legacy method that returns nil on failure (deprecated)
    /// - Parameters:
    ///   - model: The model to create context from
    ///   - contextParams: Optional context parameters
    /// - Returns: SLlamaContext instance or nil if creation fails
    @available(*, deprecated, message: "Use init(model:contextParams:) throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _createContext(model: PLlamaModel, contextParams: SLlamaContextParams? = nil) -> SLlamaContext? {
        do {
            return try SLlamaContext(model: model, contextParams: contextParams)
        } catch {
            return nil
        }
    }

    /// Create default context parameters
    /// - Returns: Default context parameters
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func defaultParams() -> SLlamaContextParams {
        llama_context_default_params()
    }

    /// Create custom context parameters
    /// - Parameters:
    ///   - contextSize: Context size (default: 4096)
    ///   - batchSize: Batch size (default: 2048)
    ///   - physicalBatchSize: Physical batch size (default: 512)
    ///   - maxSequences: Maximum number of sequences (default: 1)
    ///   - threads: Number of threads for inference (default: 0 = auto)
    ///   - batchThreads: Number of threads for batch processing (default: 0 = auto)
    ///   - enableEmbeddings: Whether to enable embeddings mode (default: false)
    ///   - enableOffloading: Whether to enable GPU offloading (default: true)
    /// - Returns: Configured context parameters
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func createParams(
        contextSize: UInt32 = 4096,
        batchSize: UInt32 = 2048,
        physicalBatchSize: UInt32 = 512,
        maxSequences: UInt32 = 1,
        threads: Int32 = 0,
        batchThreads: Int32 = 0,
        enableEmbeddings: Bool = false,
        enableOffloading: Bool = true
    )
        -> SLlamaContextParams
    {
        var params = llama_context_default_params()
        params.n_ctx = contextSize
        params.n_batch = batchSize
        params.n_ubatch = physicalBatchSize
        params.n_seq_max = maxSequences
        params.n_threads = threads
        params.n_threads_batch = batchThreads
        params.embeddings = enableEmbeddings
        params.offload_kqv = enableOffloading
        return params
    }

    // MARK: Functions

    // MARK: - System Configuration

    /// Configure context for optimal performance
    /// - Parameters:
    ///   - useOptimalThreads: Whether to use optimal thread count
    ///   - enableEmbeddings: Whether to enable embeddings mode
    ///   - enableCausalAttention: Whether to enable causal attention
    ///   - enableWarmup: Whether to enable warmup
    func configureForOptimalPerformance(
        useOptimalThreads: Bool = true,
        enableEmbeddings: Bool = false,
        enableCausalAttention: Bool = true,
        enableWarmup: Bool = true
    ) {
        if useOptimalThreads {
            let threadCount = Self.optimalThreadCount()
            setThreadCount(threadCount)
        }

        setEmbeddings(enableEmbeddings)
        setCausalAttention(enableCausalAttention)
        setWarmup(enableWarmup)
    }
}
