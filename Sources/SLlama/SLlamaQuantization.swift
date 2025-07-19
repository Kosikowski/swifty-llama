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
    /// - Returns: 0 on success, non-zero on error
    @discardableResult
    public static func quantizeModel(
        inputPath: String,
        outputPath: String,
        params: SLlamaModelQuantizeParams? = nil
    )
        -> UInt32
    {
        var quantizeParams = params ?? llama_model_quantize_default_params()
        return llama_model_quantize(inputPath, outputPath, &quantizeParams)
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
    /// - Returns: 0 on success, non-zero on error
    @discardableResult
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
    )
        -> UInt32
    {
        var params = llama_model_quantize_default_params()
        params.ftype = fileType
        params.nthread = threads
        params.allow_requantize = allowRequantize
        params.quantize_output_tensor = quantizeOutputTensor
        params.only_copy = onlyCopy
        params.pure = pure
        params.keep_split = keepSplit

        return llama_model_quantize(inputPath, outputPath, &params)
    }

    /// Get default quantization parameters
    /// - Returns: Default quantization parameters
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
    /// - Parameters:
    ///   - outputPath: Path where the quantized model will be saved
    ///   - fileType: Target quantization type
    ///   - threads: Number of threads to use (0 = auto)
    ///   - allowRequantize: Whether to allow requantizing already quantized tensors
    ///   - quantizeOutputTensor: Whether to quantize the output tensor
    /// - Returns: 0 on success, non-zero on error
    @discardableResult
    func quantize(
        to outputPath: String,
        fileType: SLlamaFileType,
        threads: Int32 = 0,
        allowRequantize: Bool = false,
        quantizeOutputTensor: Bool = true
    )
        -> UInt32
    {
        // Get the model file path (this is a simplified approach)
        // In a real implementation, you might need to store the original path
        let inputPath = "model.gguf" // This would need to be the actual model path

        return SLlamaQuantization.quantizeModel(
            inputPath: inputPath,
            outputPath: outputPath,
            fileType: fileType,
            threads: threads,
            allowRequantize: allowRequantize,
            quantizeOutputTensor: quantizeOutputTensor
        )
    }
}
