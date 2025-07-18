import Foundation
import llama

/// A wrapper for llama model
public class LlamaModel {
    // MARK: Properties

    private var model: LlamaModelPointer?

    // MARK: Computed Properties

    /// Get the model pointer for direct C API access
    public var pointer: LlamaModelPointer? {
        model
    }

    /// Get model vocabulary
    public var vocab: LlamaVocabPointer? {
        guard let model else { return nil }
        return llama_model_get_vocab(model)
    }

    /// Get number of embedding dimensions
    public var embeddingDimensions: Int32 {
        guard let model else { return 0 }
        return llama_model_n_embd(model)
    }

    /// Get number of layers
    public var layers: Int32 {
        guard let model else { return 0 }
        return llama_model_n_layer(model)
    }

    /// Get number of attention heads
    public var attentionHeads: Int32 {
        guard let model else { return 0 }
        return llama_model_n_head(model)
    }

    /// Get number of parameters
    public var parameters: UInt64 {
        guard let model else { return 0 }
        return llama_model_n_params(model)
    }

    /// Get model size in bytes
    public var size: UInt64 {
        guard let model else { return 0 }
        return llama_model_size(model)
    }

    // MARK: Lifecycle

    public init?(modelPath: String) {
        let params = llama_model_default_params()
        model = llama_model_load_from_file(modelPath, params)

        if model == nil {
            return nil
        }
    }

    deinit {
        if let model {
            llama_model_free(model)
        }
    }
}
