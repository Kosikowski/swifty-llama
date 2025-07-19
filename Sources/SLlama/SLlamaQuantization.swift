import Foundation
import llama

// MARK: - SLlamaQuantization

/// A wrapper for llama.cpp model quantization functionality
public class SLlamaQuantization {
    /// Quantize a model to reduce its size and memory usage
    /// - Parameters:
    ///   - inputPath: Path to the input model file
    ///   - outputPath: Path where the quantized model will be saved
    ///   - params: Quantization parameters (optional, uses defaults if nil)
    /// - Throws: SLlamaError if quantization fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func quantizeModel(
        inputPath: String,
        outputPath: String,
        params: SLlamaModelQuantizeParams? = nil
    ) throws {
        // Validate input file
        guard FileManager.default.fileExists(atPath: inputPath) else {
            throw SLlamaError.fileNotFound(inputPath)
        }

        guard FileManager.default.isReadableFile(atPath: inputPath) else {
            throw SLlamaError.permissionDenied(inputPath)
        }

        // Validate output directory exists and is writable
        let outputURL = URL(fileURLWithPath: outputPath)
        let outputDirectory = outputURL.deletingLastPathComponent().path

        guard FileManager.default.fileExists(atPath: outputDirectory) else {
            throw SLlamaError.fileNotFound("Output directory does not exist: \(outputDirectory)")
        }

        guard FileManager.default.isWritableFile(atPath: outputDirectory) else {
            throw SLlamaError.permissionDenied("Cannot write to output directory: \(outputDirectory)")
        }

        // Check available disk space
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: inputPath)
            let inputSize = attributes[.size] as? Int64 ?? 0

            let outputAttributes = try FileManager.default.attributesOfItem(atPath: outputDirectory)
            let availableSpace = outputAttributes[.systemFreeSize] as? Int64 ?? 0

            // Rough estimate: quantization might need up to 1.5x input size temporarily
            if availableSpace < Int64(Double(inputSize) * 1.5) {
                throw SLlamaError.insufficientSpace
            }
        } catch {
            throw SLlamaError.fileAccessError("Could not check disk space: \(error.localizedDescription)")
        }

        var quantizeParams = params ?? llama_model_quantize_default_params()
        let result = llama_model_quantize(inputPath, outputPath, &quantizeParams)

        guard result == 0 else {
            // Map specific error codes to meaningful errors
            switch result {
                case 1:
                    throw SLlamaError.invalidFormat("Invalid input model format for '\(inputPath)'")
                case 2:
                    throw SLlamaError.unsupportedQuantization
                case 3:
                    throw SLlamaError.outOfMemory
                case 4:
                    throw SLlamaError.fileAccessError("Could not write output file '\(outputPath)'")
                default:
                    throw SLlamaError
                        .operationFailed(
                            "Quantization failed from '\(inputPath)' to '\(outputPath)' with error code: \(result)"
                        )
            }
        }
    }

    /// Quantize a model with custom parameters
    /// - Parameters:
    ///   - inputPath: Path to the input model file
    ///   - outputPath: Path where the quantized model will be saved
    ///   - fileType: Target quantization type (e.g., Q4_0, Q4_1, etc.)
    ///   - threads: Number of threads to use (0 = auto)
    ///   - allowRequantize: Whether to allow requantizing already quantized tensors
    ///   - quantizeOutputTensor: Whether to quantize the output tensor
    ///   - onlyCopy: Only copy tensors without quantization
    ///   - pure: Quantize all tensors to the default type
    ///   - keepSplit: Keep the same number of shards
    /// - Throws: SLlamaError if quantization fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func quantizeModel(
        inputPath: String,
        outputPath: String,
        fileType: SLlamaFileType,
        threads: Int32 = 0,
        allowRequantize: Bool = false,
        quantizeOutputTensor: Bool = true,
        onlyCopy: Bool = false,
        pure: Bool = false,
        keepSplit: Bool = false
    ) throws {
        var params = llama_model_quantize_default_params()
        params.ftype = fileType
        params.nthread = threads
        params.allow_requantize = allowRequantize
        params.quantize_output_tensor = quantizeOutputTensor
        params.only_copy = onlyCopy
        params.pure = pure
        params.keep_split = keepSplit

        try quantizeModel(
            inputPath: inputPath,
            outputPath: outputPath,
            params: params
        )
    }

    /// Legacy method that returns error code (deprecated)
    /// - Parameters:
    ///   - inputPath: Path to the input model file
    ///   - outputPath: Path where the quantized model will be saved
    ///   - params: Quantization parameters (optional, uses defaults if nil)
    /// - Returns: 0 on success, non-zero on error
    @available(*, deprecated, message: "Use quantizeModel(inputPath:outputPath:params:) throws instead")
    @discardableResult
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _quantizeModel(
        inputPath: String,
        outputPath: String,
        params: SLlamaModelQuantizeParams? = nil
    )
        -> UInt32
    {
        do {
            try quantizeModel(inputPath: inputPath, outputPath: outputPath, params: params)
            return 0
        } catch {
            return 1 // Generic error code for legacy compatibility
        }
    }

    /// Legacy method that returns error code (deprecated)
    @available(
        *,
        deprecated,
        message: "Use quantizeModel(inputPath:outputPath:fileType:threads:allowRequantize:quantizeOutputTensor:onlyCopy:pure:keepSplit:) throws instead"
    )
    @discardableResult
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _quantizeModel(
        inputPath: String,
        outputPath: String,
        fileType: SLlamaFileType,
        threads: Int32 = 0,
        allowRequantize: Bool = false,
        quantizeOutputTensor: Bool = true,
        onlyCopy: Bool = false,
        pure: Bool = false,
        keepSplit: Bool = false
    )
        -> UInt32
    {
        do {
            try quantizeModel(
                inputPath: inputPath,
                outputPath: outputPath,
                fileType: fileType,
                threads: threads,
                allowRequantize: allowRequantize,
                quantizeOutputTensor: quantizeOutputTensor,
                onlyCopy: onlyCopy,
                pure: pure,
                keepSplit: keepSplit
            )
            return 0
        } catch {
            return 1 // Generic error code for legacy compatibility
        }
    }

    /// Get default quantization parameters
    /// - Returns: Default quantization parameters
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func defaultParams() -> SLlamaModelQuantizeParams {
        llama_model_quantize_default_params()
    }

    /// Create quantization parameters with custom settings
    /// - Parameters:
    ///   - fileType: Target quantization type
    ///   - threads: Number of threads to use
    ///   - allowRequantize: Whether to allow requantizing already quantized tensors
    ///   - quantizeOutputTensor: Whether to quantize the output tensor
    ///   - onlyCopy: Only copy tensors without quantization
    ///   - pure: Quantize all tensors to the default type
    ///   - keepSplit: Keep the same number of shards
    /// - Returns: Custom quantization parameters
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func createParams(
        fileType: SLlamaFileType,
        threads: Int32 = 0,
        allowRequantize: Bool = false,
        quantizeOutputTensor: Bool = true,
        onlyCopy: Bool = false,
        pure: Bool = false,
        keepSplit: Bool = false
    )
        -> SLlamaModelQuantizeParams
    {
        var params = llama_model_quantize_default_params()
        params.ftype = fileType
        params.nthread = threads
        params.allow_requantize = allowRequantize
        params.quantize_output_tensor = quantizeOutputTensor
        params.only_copy = onlyCopy
        params.pure = pure
        params.keep_split = keepSplit
        return params
    }
}

// MARK: - Convenience Extensions

public extension SLlamaModel {
    /// Quantize this model to a new file
    ///
    /// ⚠️ **ARCHITECTURAL DECISION**: This method is deprecated because model instances
    /// don't retain their original file paths after loading. Modern llama.cpp quantization
    /// works directly on model files, not loaded model instances.
    ///
    /// **Why this approach was abandoned**:
    /// - Model instances only contain the loaded weights, not file metadata
    /// - Quantization requires direct file I/O access to the original model file
    /// - Adding file path storage would increase memory overhead for all models
    /// - The static method approach provides better separation of concerns
    ///
    /// **Migration Path**:
    /// ```swift
    /// // ❌ Old way (doesn't work):
    /// let model = try SLlamaModel(modelPath: "input.gguf")
    /// try model.quantize(to: "output.gguf", fileType: .q4_0)
    ///
    /// // ✅ New way (works correctly):
    /// try SLlamaQuantization.quantizeModel(
    ///     inputPath: "input.gguf",
    ///     outputPath: "output.gguf",
    ///     params: SLlamaModelQuantizeParams(fileType: .q4_0)
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - outputPath: Path where the quantized model will be saved
    ///   - fileType: Target quantization type
    ///   - threads: Number of threads to use (0 = auto)
    ///   - allowRequantize: Whether to allow requantizing already quantized tensors
    ///   - quantizeOutputTensor: Whether to quantize the output tensor
    /// - Throws: SLlamaError.operationFailed with migration guidance
    @available(
        *,
        deprecated,
        message: "Model instances don't retain file paths. Use SLlamaQuantization.quantizeModel(inputPath:outputPath:params:) instead"
    )
    func quantize(
        to outputPath: String,
        fileType: SLlamaFileType,
        threads: Int32 = 0,
        allowRequantize: Bool = false,
        quantizeOutputTensor: Bool = true
    ) throws {
        // **IMPLEMENTATION DECISION**: Rather than silently fail or add complexity to store
        // file paths in every model instance, we provide a clear error with migration guidance.
        // This follows the principle of "fail fast with helpful messages" rather than
        // "fail silently" or "add unnecessary complexity".

        throw SLlamaError.operationFailed(
            """
            Model quantization from loaded instances is not supported because models don't retain their original file paths.

            Use the static method instead:
            try SLlamaQuantization.quantizeModel(
                inputPath: "path/to/original/model.gguf",
                outputPath: "\(outputPath)",
                params: SLlamaModelQuantizeParams(
                    fileType: .\(fileType),
                    threads: \(threads),
                    allowRequantize: \(allowRequantize),
                    quantizeOutputTensor: \(quantizeOutputTensor)
                )
            )
            """
        )
    }

    /// Legacy quantize method that returns error code (deprecated)
    ///
    /// **ARCHITECTURAL NOTE**: This method uses the old C-style error code pattern
    /// instead of Swift's modern error handling. Deprecated in favor of the throws version.
    @available(
        *,
        deprecated,
        message: "Use SLlamaQuantization.quantizeModel(inputPath:outputPath:params:) throws instead"
    )
    @discardableResult
    func _quantize(
        to _: String,
        fileType _: SLlamaFileType,
        threads _: Int32 = 0,
        allowRequantize _: Bool = false,
        quantizeOutputTensor _: Bool = true
    )
        -> UInt32
    {
        // Return error code 1 to indicate failure, maintaining C-style API compatibility
        1
    }
}
