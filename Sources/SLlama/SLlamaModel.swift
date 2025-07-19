import Foundation
import llama

/// A wrapper for llama model
public class SLlamaModel: @unchecked Sendable, PLlamaModel {
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

    /// Initialize with a model file path
    /// - Parameter modelPath: Path to the model file
    /// - Throws: SLlamaError if model loading fails
    public init(modelPath: String) throws {
        // Check if file exists
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw SLlamaError.fileNotFound(modelPath)
        }

        // Check if file is readable
        guard FileManager.default.isReadableFile(atPath: modelPath) else {
            throw SLlamaError.permissionDenied(modelPath)
        }

        let params = llama_model_default_params()
        model = llama_model_load_from_file(modelPath, params)

        guard model != nil else {
            // Try to determine the specific error
            let fileURL = URL(fileURLWithPath: modelPath)
            if !fileURL.hasDirectoryPath {
                // Check file size to detect potential issues
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: modelPath)
                    let fileSize = attributes[.size] as? Int64 ?? 0
                    if fileSize == 0 {
                        throw SLlamaError.corruptedFile("Model file is empty: '\(modelPath)'")
                    } else if fileSize < 1024 {
                        throw SLlamaError
                            .invalidFormat(
                                "Model file too small (\(fileSize) bytes) to be valid: '\(modelPath)' (minimum 1KB expected)"
                            )
                    }
                } catch {
                    throw SLlamaError
                        .fileAccessError(
                            "Could not read model file attributes for '\(modelPath)': \(error.localizedDescription)"
                        )
                }
            }
            let fileSize = (try? FileManager.default.attributesOfItem(atPath: modelPath)[.size] as? Int64) ?? nil
            let sizeInfo = fileSize.map { " (file size: \($0) bytes)" } ?? ""
            throw SLlamaError.modelLoadingFailed("Model could not be loaded from '\(modelPath)'\(sizeInfo)")
        }
    }

    /// Initialize with an existing model pointer (does not take ownership)
    /// - Parameter modelPointer: The model pointer
    /// - Throws: SLlamaError if the model pointer is invalid
    public init(modelPointer: SLlamaModelPointer?) throws {
        guard let modelPointer else {
            throw SLlamaError.invalidModel("Model pointer is null")
        }
        model = modelPointer
    }

    deinit {
        if let model {
            llama_model_free(model)
        }
    }

    // MARK: Static Functions

    /// Legacy initializer that returns nil on failure (deprecated)
    /// - Parameter modelPath: Path to the model file
    /// - Returns: SLlamaModel instance or nil if loading fails
    @available(*, deprecated, message: "Use init(modelPath:) throws instead")
    public static func _createModel(modelPath: String) -> SLlamaModel? {
        do {
            return try SLlamaModel(modelPath: modelPath)
        } catch {
            return nil
        }
    }

    /// Legacy initializer that returns nil on failure (deprecated)
    /// - Parameter modelPointer: The model pointer
    /// - Returns: SLlamaModel instance or nil if pointer is invalid
    @available(*, deprecated, message: "Use init(modelPointer:) throws instead")
    public static func _createModel(modelPointer: SLlamaModelPointer?) -> SLlamaModel? {
        do {
            return try SLlamaModel(modelPointer: modelPointer)
        } catch {
            return nil
        }
    }

    // MARK: Functions

    // MARK: - Metadata Methods

    /// Get metadata value by key
    /// - Parameters:
    ///   - key: The metadata key
    ///   - bufferSize: Size of the buffer for the value (default: 256)
    /// - Returns: The metadata value as string
    /// - Throws: SLlamaError if the key is not found or buffer is too small
    public func metadataValue(for key: String, bufferSize: Int = 256) throws -> String {
        guard let model else {
            throw SLlamaError.contextNotInitialized
        }

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let result = llama_model_meta_val_str(model, key, &buffer, bufferSize)

        if result > 0 {
            // Convert CChar array to UInt8 array for UTF8 decoding
            let uint8Buffer = buffer.map { UInt8(bitPattern: $0) }
            guard let string = String(decoding: uint8Buffer.prefix(Int(result)), as: UTF8.self) as String? else {
                throw SLlamaError.encodingFailure
            }
            return string
        } else if result == 0 {
            throw SLlamaError.keyNotFound("Metadata key '\(key)' not found in model")
        } else {
            throw SLlamaError.bufferTooSmall
        }
    }

    /// Get metadata key by index
    /// - Parameters:
    ///   - index: The metadata index
    ///   - bufferSize: Size of the buffer for the key (default: 256)
    /// - Returns: The metadata key as string
    /// - Throws: SLlamaError if the index is invalid or buffer is too small
    public func metadataKey(at index: Int32, bufferSize: Int = 256) throws -> String {
        guard let model else {
            throw SLlamaError.contextNotInitialized
        }

        guard index >= 0, index < metadataCount else {
            throw SLlamaError.invalidIndex("Metadata index \(index) out of range [0..\(metadataCount - 1)]")
        }

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let result = llama_model_meta_key_by_index(model, index, &buffer, bufferSize)

        if result > 0 {
            // Convert CChar array to UInt8 array for UTF8 decoding
            let uint8Buffer = buffer.map { UInt8(bitPattern: $0) }
            guard let string = String(decoding: uint8Buffer.prefix(Int(result)), as: UTF8.self) as String? else {
                throw SLlamaError.encodingFailure
            }
            return string
        } else if result == 0 {
            throw SLlamaError.invalidIndex("Metadata index \(index) is invalid or out of range")
        } else {
            throw SLlamaError.bufferTooSmall
        }
    }

    /// Get metadata value by index
    /// - Parameters:
    ///   - index: The metadata index
    ///   - bufferSize: Size of the buffer for the value (default: 256)
    /// - Returns: The metadata value as string
    /// - Throws: SLlamaError if the index is invalid or buffer is too small
    public func metadataValue(at index: Int32, bufferSize: Int = 256) throws -> String {
        guard let model else {
            throw SLlamaError.contextNotInitialized
        }

        guard index >= 0, index < metadataCount else {
            throw SLlamaError.invalidIndex("Metadata index \(index) out of range [0..\(metadataCount - 1)]")
        }

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let result = llama_model_meta_val_str_by_index(model, index, &buffer, bufferSize)

        if result > 0 {
            // Convert CChar array to UInt8 array for UTF8 decoding
            let uint8Buffer = buffer.map { UInt8(bitPattern: $0) }
            guard let string = String(decoding: uint8Buffer.prefix(Int(result)), as: UTF8.self) as String? else {
                throw SLlamaError.encodingFailure
            }
            return string
        } else if result == 0 {
            throw SLlamaError.invalidIndex("Metadata index \(index) is invalid or out of range")
        } else {
            throw SLlamaError.bufferTooSmall
        }
    }

    /// Get model description
    /// - Parameter bufferSize: Size of the buffer for the description (default: 1024)
    /// - Returns: The model description as string
    /// - Throws: SLlamaError if buffer is too small or model is not initialized
    public func description(bufferSize: Int = 1024) throws -> String {
        guard let model else {
            throw SLlamaError.contextNotInitialized
        }

        var buffer = [CChar](repeating: 0, count: bufferSize)
        let result = llama_model_desc(model, &buffer, bufferSize)

        if result > 0 {
            // Convert CChar array to UInt8 array for UTF8 decoding
            let uint8Buffer = buffer.map { UInt8(bitPattern: $0) }
            guard let string = String(decoding: uint8Buffer.prefix(Int(result)), as: UTF8.self) as String? else {
                throw SLlamaError.encodingFailure
            }
            return string
        } else if result == 0 {
            throw SLlamaError.metadataAccessFailed("Model description not available")
        } else {
            throw SLlamaError.bufferTooSmall
        }
    }

    /// Get chat template by name
    /// - Parameter name: The template name
    /// - Returns: The chat template as string
    /// - Throws: SLlamaError if template is not found or model is not initialized
    public func chatTemplate(named name: String) throws -> String {
        guard let model else {
            throw SLlamaError.contextNotInitialized
        }

        let template = llama_model_chat_template(model, name)
        guard let template, let templateString = String(cString: template, encoding: .utf8) else {
            throw SLlamaError.keyNotFound("Chat template '\(name)' not found")
        }
        return templateString
    }

    // MARK: - Legacy Methods (Deprecated)

    /// Legacy method that returns nil on failure (deprecated)
    @available(*, deprecated, message: "Use metadataValue(for:bufferSize:) throws instead")
    public func _metadataValue(for key: String, bufferSize: Int = 256) -> String? {
        try? metadataValue(for: key, bufferSize: bufferSize)
    }

    /// Legacy method that returns nil on failure (deprecated)
    @available(*, deprecated, message: "Use metadataKey(at:bufferSize:) throws instead")
    public func _metadataKey(at index: Int32, bufferSize: Int = 256) -> String? {
        try? metadataKey(at: index, bufferSize: bufferSize)
    }

    /// Legacy method that returns nil on failure (deprecated)
    @available(*, deprecated, message: "Use metadataValue(at:bufferSize:) throws instead")
    public func _metadataValue(at index: Int32, bufferSize: Int = 256) -> String? {
        try? metadataValue(at: index, bufferSize: bufferSize)
    }

    /// Legacy method that returns nil on failure (deprecated)
    @available(*, deprecated, message: "Use description(bufferSize:) throws instead")
    public func _description(bufferSize: Int = 1024) -> String? {
        try? description(bufferSize: bufferSize)
    }

    /// Legacy method that returns nil on failure (deprecated)
    @available(*, deprecated, message: "Use chatTemplate(named:) throws instead")
    public func _chatTemplate(named name: String) -> String? {
        try? chatTemplate(named: name)
    }

    /// Check if model has embeddings
    public func hasEmbeddings() -> Bool {
        embeddingDimensions > 0
    }

    /// Get model description
    public func getDescription(bufferSize: Int) -> String? {
        try? description(bufferSize: bufferSize)
    }

    /// Get metadata value by key (protocol requirement)
    public func getMetadata(key: String, bufferSize: Int) throws -> String {
        try metadataValue(for: key, bufferSize: bufferSize)
    }
}
