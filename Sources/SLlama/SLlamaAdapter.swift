import Foundation
import llama

// MARK: - SLlamaAdapter

/// A wrapper for llama LoRA adapter operations
public class SLlamaAdapter: @unchecked Sendable {
    // MARK: Properties

    var adapter: SLlamaAdapterLoraPointer?

    // MARK: Computed Properties

    /// Get the underlying C adapter pointer for direct API access
    public var cAdapter: SLlamaAdapterLoraPointer? {
        adapter
    }

    /// Check if the adapter is valid
    public var isValid: Bool {
        adapter != nil
    }

    // MARK: Lifecycle

    /// Initialize with a LoRA adapter file path
    /// - Parameters:
    ///   - model: The model to attach the adapter to
    ///   - path: Path to the LoRA adapter file
    /// - Throws: SLlamaError if adapter loading fails
    public init(model: PLlamaModel, path: String) throws {
        guard let modelPtr = model.pointer else {
            throw SLlamaError.invalidModel("Model pointer is null")
        }

        // Validate adapter file
        guard FileManager.default.fileExists(atPath: path) else {
            throw SLlamaError.fileNotFound(path)
        }

        guard FileManager.default.isReadableFile(atPath: path) else {
            throw SLlamaError.permissionDenied(path)
        }

        // Check file size and format
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: path)
            let fileSize = attributes[.size] as? Int64 ?? 0
            if fileSize == 0 {
                throw SLlamaError.corruptedFile("Adapter file is empty: '\(path)'")
            } else if fileSize < 100 {
                throw SLlamaError
                    .invalidFormat(
                        "Adapter file too small (\(fileSize) bytes) to be valid: '\(path)' (minimum 100 bytes expected)"
                    )
            }
        } catch {
            throw SLlamaError
                .fileAccessError("Could not read adapter file attributes for '\(path)': \(error.localizedDescription)")
        }

        let adapterPtr = llama_adapter_lora_init(modelPtr, path)
        guard adapterPtr != nil else {
            throw SLlamaError
                .adapterLoadingFailed(
                    "Could not load LoRA adapter from '\(path)' (file may be corrupted or incompatible)"
                )
        }

        adapter = adapterPtr
    }

    deinit {
        if let adapter {
            llama_adapter_lora_free(adapter)
        }
    }

    // MARK: Static Functions

    /// Legacy initializer that returns nil on failure (deprecated)
    /// - Parameters:
    ///   - model: The model to attach the adapter to
    ///   - path: Path to the LoRA adapter file
    /// - Returns: SLlamaAdapter instance or nil if loading fails
    @available(*, deprecated, message: "Use init(model:path:) throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _createAdapter(model: PLlamaModel, path: String) -> SLlamaAdapter? {
        do {
            return try SLlamaAdapter(model: model, path: path)
        } catch {
            return nil
        }
    }
}

// MARK: - Extension to SLlamaContext for LoRA Support

public extension SLlamaContext {
    /// Add a LoRA adapter to this context
    /// - Parameter adapter: The LoRA adapter to add
    /// - Throws: SLlamaError if the adapter operation fails
    func addLoRAAdapter(_ adapter: SLlamaAdapter) throws {
        guard let context = pointer else {
            throw SLlamaError.contextNotInitialized
        }

        guard let adapterPtr = adapter.cAdapter else {
            throw SLlamaError.invalidAdapter("Adapter pointer is null")
        }

        let result = llama_set_adapter_lora(context, adapterPtr, 0)
        guard result == 0 else {
            throw SLlamaError.adapterOperationFailed("Failed to add LoRA adapter with error code: \(result)")
        }
    }

    /// Remove a specific LoRA adapter from this context
    /// - Parameter adapter: The LoRA adapter to remove
    /// - Throws: SLlamaError if the adapter operation fails
    func removeLoRAAdapter(_ adapter: SLlamaAdapter) throws {
        guard let context = pointer else {
            throw SLlamaError.contextNotInitialized
        }

        guard let adapterPtr = adapter.cAdapter else {
            throw SLlamaError.invalidAdapter("Adapter pointer is null")
        }

        let result = llama_rm_adapter_lora(context, adapterPtr)
        guard result == 0 else {
            throw SLlamaError.adapterOperationFailed("Failed to remove LoRA adapter with error code: \(result)")
        }
    }

    /// Remove all LoRA adapters from this context
    func clearLoRAAdapters() {
        guard let context = pointer else { return }
        llama_clear_adapter_lora(context)
    }

    /// Load and add a LoRA adapter from file
    /// - Parameter path: Path to the LoRA adapter file
    /// - Returns: The loaded adapter
    /// - Throws: SLlamaError if loading or adding the adapter fails
    func loadLoRAAdapter(from path: String) throws -> SLlamaAdapter {
        guard let model = associatedModel else {
            throw SLlamaError.contextNotInitialized
        }

        let adapter = try SLlamaAdapter(model: model, path: path)
        try addLoRAAdapter(adapter)
        return adapter
    }

    /// Apply a control vector to the context
    /// - Parameters:
    ///   - data: Pointer to the control vector data (float array)
    ///   - length: Length of the control vector data
    ///   - embeddingDimensions: Number of embedding dimensions
    ///   - layerStart: Starting layer index (inclusive)
    ///   - layerEnd: Ending layer index (inclusive)
    /// - Throws: SLlamaError if the control vector operation fails
    func applyControlVector(
        data: UnsafePointer<Float>,
        length: Int,
        embeddingDimensions: Int32,
        layerStart: Int32,
        layerEnd: Int32
    ) throws {
        guard let context = pointer else {
            throw SLlamaError.contextNotInitialized
        }

        let result = llama_apply_adapter_cvec(
            context,
            data,
            size_t(length),
            embeddingDimensions,
            layerStart,
            layerEnd
        )

        guard result == 0 else {
            throw SLlamaError.adapterOperationFailed("Failed to apply control vector with error code: \(result)")
        }
    }

    /// Clear the currently loaded control vector
    /// - Throws: SLlamaError if the operation fails
    func clearControlVector() throws {
        guard let context = pointer else {
            throw SLlamaError.contextNotInitialized
        }

        let result = llama_apply_adapter_cvec(context, nil, 0, 0, 0, 0)
        guard result == 0 else {
            throw SLlamaError.adapterOperationFailed("Failed to clear control vector with error code: \(result)")
        }
    }

    // MARK: - Legacy Methods (Deprecated)

    /// Legacy method that returns bool (deprecated)
    /// - Parameter adapter: The LoRA adapter to add
    /// - Returns: true if the adapter was successfully added, false otherwise
    @available(*, deprecated, message: "Use addLoRAAdapter(_:) throws instead")
    func _addLoRAAdapter(_ adapter: SLlamaAdapter) -> Bool {
        do {
            try addLoRAAdapter(adapter)
            return true
        } catch {
            return false
        }
    }

    /// Legacy method that returns bool (deprecated)
    /// - Parameter adapter: The LoRA adapter to remove
    /// - Returns: true if the adapter was successfully removed, false otherwise
    @available(*, deprecated, message: "Use removeLoRAAdapter(_:) throws instead")
    func _removeLoRAAdapter(_ adapter: SLlamaAdapter) -> Bool {
        do {
            try removeLoRAAdapter(adapter)
            return true
        } catch {
            return false
        }
    }

    /// Legacy method that returns optional adapter (deprecated)
    /// - Parameter path: Path to the LoRA adapter file
    /// - Returns: The loaded adapter, or nil if loading failed
    @available(*, deprecated, message: "Use loadLoRAAdapter(from:) throws instead")
    func _loadLoRAAdapter(from path: String) -> SLlamaAdapter? {
        try? loadLoRAAdapter(from: path)
    }

    /// Legacy method that returns error code (deprecated)
    /// - Parameters:
    ///   - data: Pointer to the control vector data (float array)
    ///   - length: Length of the control vector data
    ///   - embeddingDimensions: Number of embedding dimensions
    ///   - layerStart: Starting layer index (inclusive)
    ///   - layerEnd: Ending layer index (inclusive)
    /// - Returns: 0 on success, negative value on error
    @available(
        *,
        deprecated,
        message: "Use applyControlVector(data:length:embeddingDimensions:layerStart:layerEnd:) throws instead"
    )
    func _applyControlVector(
        data: UnsafePointer<Float>,
        length: Int,
        embeddingDimensions: Int32,
        layerStart: Int32,
        layerEnd: Int32
    )
        -> Int32
    {
        do {
            try applyControlVector(
                data: data,
                length: length,
                embeddingDimensions: embeddingDimensions,
                layerStart: layerStart,
                layerEnd: layerEnd
            )
            return 0
        } catch {
            return -1
        }
    }

    /// Legacy method that returns error code (deprecated)
    /// - Returns: 0 on success, negative value on error
    @available(*, deprecated, message: "Use clearControlVector() throws instead")
    func _clearControlVector() -> Int32 {
        do {
            try clearControlVector()
            return 0
        } catch {
            return -1
        }
    }
}
