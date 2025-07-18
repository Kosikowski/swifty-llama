import Foundation
import llama

/// A wrapper for llama context
public class SLlamaContext {
    // MARK: Properties

    private var context: SLlamaContextPointer?
    private var model: SLlamaModel?

    // MARK: Computed Properties

    /// Get the context pointer for direct C API access
    public var pointer: SLlamaContextPointer? {
        context
    }

    /// Get the associated model
    public var associatedModel: SLlamaModel? {
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

    // MARK: Lifecycle

    public init?(model: SLlamaModel, contextParams: SLlamaContextParams? = nil) {
        self.model = model

        let params = contextParams ?? llama_context_default_params()
        context = llama_init_from_model(model.pointer, params)

        if context == nil {
            return nil
        }
    }

    deinit {
        if let context {
            llama_free(context)
        }
    }
    
    // MARK: System Configuration
    
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
