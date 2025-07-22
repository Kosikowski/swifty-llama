import Foundation

// MARK: - Model Saving Types

/// Information about a saved LoRA adapter
public struct LoRAAdapterInfo: Codable, Sendable {
    public let path: String
    public let scale: Float
    public let metadata: LoRAMetadata
    public let baseModelPath: String
    public let trainingConfig: TrainingConfig?
    public let trainingMetrics: TrainingMetrics?
    public let createdAt: Date

    public init(
        path: String,
        scale: Float,
        metadata: LoRAMetadata,
        baseModelPath: String,
        trainingConfig: TrainingConfig?,
        trainingMetrics: TrainingMetrics?,
        createdAt: Date
    ) {
        self.path = path
        self.scale = scale
        self.metadata = metadata
        self.baseModelPath = baseModelPath
        self.trainingConfig = trainingConfig
        self.trainingMetrics = trainingMetrics
        self.createdAt = createdAt
    }
}

/// Information about a saved fine-tuned model
public struct FineTunedModelInfo: Codable, Sendable {
    public let path: String
    public let format: ModelFormat
    public let baseModelPath: String
    public let loraAdapter: LoRAAdapter?
    public let trainingConfig: TrainingConfig?
    public let trainingMetrics: TrainingMetrics?
    public let metadata: ModelMetadata
    public let createdAt: Date

    public init(
        path: String,
        format: ModelFormat,
        baseModelPath: String,
        loraAdapter: LoRAAdapter?,
        trainingConfig: TrainingConfig?,
        trainingMetrics: TrainingMetrics?,
        metadata: ModelMetadata,
        createdAt: Date
    ) {
        self.path = path
        self.format = format
        self.baseModelPath = baseModelPath
        self.loraAdapter = loraAdapter
        self.trainingConfig = trainingConfig
        self.trainingMetrics = trainingMetrics
        self.metadata = metadata
        self.createdAt = createdAt
    }
}

/// Model export information
public struct ModelExportInfo: Codable, Sendable {
    public let path: String
    public let format: ModelFormat
    public let quantization: ModelQuantization?
    public let baseModelPath: String
    public let loraAdapter: LoRAAdapter?
    public let createdAt: Date

    public init(
        path: String,
        format: ModelFormat,
        quantization: ModelQuantization?,
        baseModelPath: String,
        loraAdapter: LoRAAdapter?,
        createdAt: Date
    ) {
        self.path = path
        self.format = format
        self.quantization = quantization
        self.baseModelPath = baseModelPath
        self.loraAdapter = loraAdapter
        self.createdAt = createdAt
    }
}

/// Model saving information
public struct ModelSavingInfo: Codable, Sendable {
    public let baseModelPath: String?
    public let currentLoRA: LoRAAdapter?
    public let trainingConfig: TrainingConfig?
    public let trainingMetrics: TrainingMetrics?
    public let trainingDataset: TrainingDataset?

    public init(
        baseModelPath: String?,
        currentLoRA: LoRAAdapter?,
        trainingConfig: TrainingConfig?,
        trainingMetrics: TrainingMetrics?,
        trainingDataset: TrainingDataset?
    ) {
        self.baseModelPath = baseModelPath
        self.currentLoRA = currentLoRA
        self.trainingConfig = trainingConfig
        self.trainingMetrics = trainingMetrics
        self.trainingDataset = trainingDataset
    }
}

/// Supported model formats
public enum ModelFormat: String, Codable, Sendable, CaseIterable {
    case gguf
    case safetensors
    case pytorch
    case onnx
    case tensorrt

    public var fileExtension: String {
        switch self {
            case .gguf:
                "gguf"
            case .safetensors:
                "safetensors"
            case .pytorch:
                "pt"
            case .onnx:
                "onnx"
            case .tensorrt:
                "engine"
        }
    }

    public var description: String {
        switch self {
            case .gguf:
                "GGUF (GGML Universal Format)"
            case .safetensors:
                "SafeTensors"
            case .pytorch:
                "PyTorch"
            case .onnx:
                "ONNX"
            case .tensorrt:
                "TensorRT"
        }
    }
}

/// Model quantization settings
public struct ModelQuantization: Codable, Sendable {
    public let type: QuantizationType
    public let bits: Int
    public let groupSize: Int?
    public let scaleType: String?

    public init(
        type: QuantizationType,
        bits: Int,
        groupSize: Int? = nil,
        scaleType: String? = nil
    ) {
        self.type = type
        self.bits = bits
        self.groupSize = groupSize
        self.scaleType = scaleType
    }
}

/// Quantization types
public enum QuantizationType: String, Codable, Sendable, CaseIterable {
    case q4_0
    case q4_1
    case q5_0
    case q5_1
    case q8_0
    case f16
    case f32

    public var description: String {
        switch self {
            case .q4_0:
                "4-bit quantization (Q4_0)"
            case .q4_1:
                "4-bit quantization (Q4_1)"
            case .q5_0:
                "5-bit quantization (Q5_0)"
            case .q5_1:
                "5-bit quantization (Q5_1)"
            case .q8_0:
                "8-bit quantization (Q8_0)"
            case .f16:
                "16-bit floating point"
            case .f32:
                "32-bit floating point"
        }
    }
}

/// Model metadata
public struct ModelMetadata: Codable, Sendable {
    public let name: String
    public let description: String?
    public let version: String?
    public let author: String?
    public let license: String?
    public let tags: [String]
    public let parameters: Int?
    public let contextLength: Int?
    public let customFields: [String: String]

    public init(
        name: String,
        description: String? = nil,
        version: String? = nil,
        author: String? = nil,
        license: String? = nil,
        tags: [String] = [],
        parameters: Int? = nil,
        contextLength: Int? = nil,
        customFields: [String: String] = [:]
    ) {
        self.name = name
        self.description = description
        self.version = version
        self.author = author
        self.license = license
        self.tags = tags
        self.parameters = parameters
        self.contextLength = contextLength
        self.customFields = customFields
    }
}
