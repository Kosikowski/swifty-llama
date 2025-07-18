import Foundation
import llama

/// A wrapper for llama context
public class LlamaContext {
    // MARK: Properties

    private var context: LlamaContextPointer?
    private var model: LlamaModel?

    // MARK: Computed Properties

    /// Get the context pointer for direct C API access
    public var pointer: LlamaContextPointer? {
        context
    }

    /// Get the associated model
    public var associatedModel: LlamaModel? {
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

    public init?(model: LlamaModel, contextParams: LlamaContextParams? = nil) {
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
}
