import Foundation
import llama

/// A wrapper for llama model
public class SLlamaModel {
    // MARK: Properties

    private var model: SLlamaModelPointer?

    // MARK: Computed Properties

    /// Get the model pointer for direct C API access
    public var pointer: SLlamaModelPointer? {
        model
    }

    /// Get model vocabulary
    public var vocab: SLlamaVocabPointer? {
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

    /// Get number of KV attention heads
    public var kvAttentionHeads: Int32 {
        guard let model else { return 0 }
        return llama_model_n_head_kv(model)
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

    /// Get training context length
    public var trainingContextLength: Int32 {
        guard let model else { return 0 }
        return llama_model_n_ctx_train(model)
    }

    /// Get RoPE type
    public var ropeType: SLlamaRopeType {
        guard let model else { return .none }
        return llama_model_rope_type(model)
    }

    /// Get RoPE frequency scale for training
    public var ropeFreqScaleTrain: Float {
        guard let model else { return 0.0 }
        return llama_model_rope_freq_scale_train(model)
    }

    /// Get number of sliding window attention (SWA)
    public var slidingWindowAttention: Int32 {
        guard let model else { return 0 }
        return llama_model_n_swa(model)
    }

    /// Check if model has encoder
    public var hasEncoder: Bool {
        guard let model else { return false }
        return llama_model_has_encoder(model)
    }

    /// Check if model has decoder
    public var hasDecoder: Bool {
        guard let model else { return false }
        return llama_model_has_decoder(model)
    }

    /// Check if model is recurrent
    public var isRecurrent: Bool {
        guard let model else { return false }
        return llama_model_is_recurrent(model)
    }

    /// Get decoder start token
    public var decoderStartToken: SLlamaToken {
        guard let model else { return SLlamaTokenNull }
        return llama_model_decoder_start_token(model)
    }

    /// Get number of metadata entries
    public var metadataCount: Int32 {
        guard let model else { return 0 }
        return llama_model_meta_count(model)
    }

    // MARK: Lifecycle

    public init?(modelPath: String) {
        let params = llama_model_default_params()
        model = llama_model_load_from_file(modelPath, params)

        if model == nil {
            return nil
        }
    }

    /// Initialize with an existing model pointer (does not take ownership)
    /// - Parameter modelPointer: The model pointer
    public init?(modelPointer: SLlamaModelPointer?) {
        guard let modelPointer else { return nil }
        model = modelPointer
    }

    deinit {
        if let model {
            llama_model_free(model)
        }
    }

    // MARK: - Metadata Methods

    /// Get metadata value by key
    /// - Parameters:
    ///   - key: The metadata key
    ///   - bufferSize: Size of the buffer for the value (default: 256)
    /// - Returns: The metadata value as string, or nil if not found
    public func metadataValue(for key: String, bufferSize: Int = 256) -> String? {
        guard let model else { return nil }

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let result = llama_model_meta_val_str(model, key, &buffer, bufferSize)

        if result > 0 {
            // Convert CChar array to UInt8 array for UTF8 decoding
            let uint8Buffer = buffer.map { UInt8(bitPattern: $0) }
            return String(decoding: uint8Buffer.prefix(Int(result)), as: UTF8.self)
        }
        return nil
    }

    /// Get metadata key by index
    /// - Parameters:
    ///   - index: The metadata index
    ///   - bufferSize: Size of the buffer for the key (default: 256)
    /// - Returns: The metadata key as string, or nil if index is invalid
    public func metadataKey(at index: Int32, bufferSize: Int = 256) -> String? {
        guard let model else { return nil }

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let result = llama_model_meta_key_by_index(model, index, &buffer, bufferSize)

        if result > 0 {
            // Convert CChar array to UInt8 array for UTF8 decoding
            let uint8Buffer = buffer.map { UInt8(bitPattern: $0) }
            return String(decoding: uint8Buffer.prefix(Int(result)), as: UTF8.self)
        }
        return nil
    }

    /// Get metadata value by index
    /// - Parameters:
    ///   - index: The metadata index
    ///   - bufferSize: Size of the buffer for the value (default: 256)
    /// - Returns: The metadata value as string, or nil if index is invalid
    public func metadataValue(at index: Int32, bufferSize: Int = 256) -> String? {
        guard let model else { return nil }

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let result = llama_model_meta_val_str_by_index(model, index, &buffer, bufferSize)

        if result > 0 {
            // Convert CChar array to UInt8 array for UTF8 decoding
            let uint8Buffer = buffer.map { UInt8(bitPattern: $0) }
            return String(decoding: uint8Buffer.prefix(Int(result)), as: UTF8.self)
        }
        return nil
    }

    /// Get model description
    /// - Parameter bufferSize: Size of the buffer for the description (default: 1024)
    /// - Returns: The model description as string, or nil if failed
    public func description(bufferSize: Int = 1024) -> String? {
        guard let model else { return nil }

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let result = llama_model_desc(model, &buffer, bufferSize)

        if result > 0 {
            // Convert CChar array to UInt8 array for UTF8 decoding
            let uint8Buffer = buffer.map { UInt8(bitPattern: $0) }
            return String(decoding: uint8Buffer.prefix(Int(result)), as: UTF8.self)
        }
        return nil
    }

    /// Get chat template by name
    /// - Parameter name: The template name
    /// - Returns: The chat template as string, or nil if not found
    public func chatTemplate(named name: String) -> String? {
        guard let model else { return nil }
        let template = llama_model_chat_template(model, name)
        return template != nil ? String(cString: template!) : nil
    }
}
