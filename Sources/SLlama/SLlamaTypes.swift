import Foundation
import llama

// MARK: - Core Type Aliases

/// Type alias for llama memory handle
public typealias SLlamaMemory = llama_memory_t

/// Type alias for llama position (context position)
public typealias SLlamaPosition = llama_pos

/// Type alias for llama token (token ID)
public typealias SLlamaToken = llama_token

/// Type alias for llama sequence ID
public typealias SLlamaSequenceId = llama_seq_id

/// Type alias for llama sampler context
public typealias SLlamaSamplerContext = llama_sampler_context_t

// MARK: - Enum Type Aliases

/// Type alias for vocabulary type
public typealias SLlamaVocabType = llama_vocab_type

/// Type alias for RoPE type
public typealias SLlamaRopeType = llama_rope_type

/// Type alias for token type
public typealias SLlamaTokenType = llama_token_type

/// Type alias for token attribute
public typealias SLlamaTokenAttribute = llama_token_attr

/// Type alias for model file type
public typealias SLlamaFileType = llama_ftype

/// Type alias for RoPE scaling type
public typealias SLlamaRopeScalingType = llama_rope_scaling_type

/// Type alias for pooling type
public typealias SLlamaPoolingType = llama_pooling_type

/// Type alias for attention type
public typealias SLlamaAttentionType = llama_attention_type

/// Type alias for split mode
public typealias SLlamaSplitMode = llama_split_mode

/// Type alias for model KV override type
public typealias SLlamaModelKvOverrideType = llama_model_kv_override_type

// MARK: - Struct Type Aliases

/// Type alias for token data
public typealias SLlamaTokenData = llama_token_data

/// Type alias for token data array
public typealias SLlamaTokenDataArray = llama_token_data_array

/// Type alias for token data array pointer
public typealias SLlamaTokenDataArrayPointer = UnsafeMutablePointer<SLlamaTokenDataArray>

/// Type alias for sequence ID
public typealias SLlamaSeqId = llama_seq_id

/// Type alias for model KV override
public typealias SLlamaModelKvOverride = llama_model_kv_override

/// Type alias for model tensor buffer type override
public typealias SLlamaModelTensorBuftOverride = llama_model_tensor_buft_override

/// Type alias for model parameters
public typealias SLlamaModelParams = llama_model_params

/// Type alias for context parameters
public typealias SLlamaContextParams = llama_context_params

/// Type alias for model quantize parameters
public typealias SLlamaModelQuantizeParams = llama_model_quantize_params

/// Type alias for logit bias
public typealias SLlamaLogitBias = llama_logit_bias

/// Type alias for sampler chain parameters
public typealias SLlamaSamplerChainParams = llama_sampler_chain_params

/// Type alias for chat message
public typealias SLlamaChatMessage = llama_chat_message

/// Type alias for performance context data
public typealias SLlamaPerfContextData = llama_perf_context_data

/// Type alias for performance sampler data
public typealias SLlamaPerfSamplerData = llama_perf_sampler_data

// MARK: - Function Pointer Type Aliases

/// Type alias for progress callback
public typealias SLlamaProgressCallback = llama_progress_callback

/// Type alias for optimization parameter filter
public typealias SLlamaOptParamFilter = llama_opt_param_filter

// MARK: - Opaque Pointer Type Aliases

/// Type alias for vocabulary pointer
public typealias SLlamaVocabPointer = OpaquePointer

/// Type alias for model pointer
public typealias SLlamaModelPointer = OpaquePointer

/// Type alias for context pointer
public typealias SLlamaContextPointer = OpaquePointer

/// Type alias for sampler pointer
public typealias SLlamaSamplerPointer = UnsafeMutablePointer<llama_sampler>

/// Type alias for adapter LoRA pointer
public typealias SLlamaAdapterLoraPointer = OpaquePointer

// MARK: - Pointer Type Aliases

/// Type alias for token pointer
public typealias SLlamaTokenPointer = UnsafeMutablePointer<SLlamaToken>

/// Type alias for float pointer
public typealias SLlamaFloatPointer = UnsafeMutablePointer<Float>

/// Type alias for position pointer
public typealias SLlamaPositionPointer = UnsafeMutablePointer<SLlamaPosition>

/// Type alias for int32 pointer
public typealias SLlamaInt32Pointer = UnsafeMutablePointer<Int32>

/// Type alias for int8 pointer
public typealias SLlamaInt8Pointer = UnsafeMutablePointer<Int8>

/// Type alias for sequence ID pointer
public typealias SLlamaSeqIdPointer = UnsafeMutablePointer<SLlamaSeqId>

/// Type alias for sequence ID pointer pointer
public typealias SLlamaSeqIdPointerPointer = UnsafeMutablePointer<SLlamaSeqIdPointer?>

/// Type alias for raw pointer
public typealias SLlamaRawPointer = UnsafeMutableRawPointer

// MARK: - Constants

/// Null token value
public let SLlamaTokenNull: SLlamaToken = LLAMA_TOKEN_NULL

/// Default seed value
public let SLlamaDefaultSeed: UInt32 = LLAMA_DEFAULT_SEED

// MARK: - File Magic Constants

/// GGLA file magic
public let SLlamaFileMagicGGLA: UInt32 = LLAMA_FILE_MAGIC_GGLA

/// GGSN file magic
public let SLlamaFileMagicGGSN: UInt32 = LLAMA_FILE_MAGIC_GGSN

/// GGSQ file magic
public let SLlamaFileMagicGGSQ: UInt32 = LLAMA_FILE_MAGIC_GGSQ

/// Session magic
public let SLlamaSessionMagic: UInt32 = LLAMA_SESSION_MAGIC

/// Session version
public let SLlamaSessionVersion: Int32 = LLAMA_SESSION_VERSION

/// State sequence magic
public let SLlamaStateSeqMagic: UInt32 = LLAMA_STATE_SEQ_MAGIC

/// State sequence version
public let SLlamaStateSeqVersion: Int32 = LLAMA_STATE_SEQ_VERSION

// MARK: - Extension for Common Constants

public extension SLlamaVocabType {
    /// None vocabulary type
    static let none = LLAMA_VOCAB_TYPE_NONE
    /// SPM vocabulary type
    static let spm = LLAMA_VOCAB_TYPE_SPM
    /// BPE vocabulary type
    static let bpe = LLAMA_VOCAB_TYPE_BPE
    /// WPM vocabulary type
    static let wpm = LLAMA_VOCAB_TYPE_WPM
    /// UGM vocabulary type
    static let ugm = LLAMA_VOCAB_TYPE_UGM
    /// RWKV vocabulary type
    static let rwkv = LLAMA_VOCAB_TYPE_RWKV
    /// PLaMo-2 vocabulary type
    static let plamo2 = LLAMA_VOCAB_TYPE_PLAMO2
}

public extension SLlamaRopeType {
    /// No RoPE
    static let none = LLAMA_ROPE_TYPE_NONE
    /// Normal RoPE
    static let norm = LLAMA_ROPE_TYPE_NORM
    /// Neox RoPE
    static let neox = LLAMA_ROPE_TYPE_NEOX
    /// MRoPE
    static let mrope = LLAMA_ROPE_TYPE_MROPE
    /// Vision RoPE
    static let vision = LLAMA_ROPE_TYPE_VISION
}

public extension SLlamaTokenType {
    /// Undefined token type
    static let undefined = LLAMA_TOKEN_TYPE_UNDEFINED
    /// Normal token type
    static let normal = LLAMA_TOKEN_TYPE_NORMAL
    /// Unknown token type
    static let unknown = LLAMA_TOKEN_TYPE_UNKNOWN
    /// Control token type
    static let control = LLAMA_TOKEN_TYPE_CONTROL
    /// User defined token type
    static let userDefined = LLAMA_TOKEN_TYPE_USER_DEFINED
    /// Unused token type
    static let unused = LLAMA_TOKEN_TYPE_UNUSED
    /// Byte token type
    static let byte = LLAMA_TOKEN_TYPE_BYTE
}

public extension SLlamaTokenAttribute {
    /// Undefined token attribute
    static let undefined = LLAMA_TOKEN_ATTR_UNDEFINED
    /// Unknown token attribute
    static let unknown = LLAMA_TOKEN_ATTR_UNKNOWN
    /// Unused token attribute
    static let unused = LLAMA_TOKEN_ATTR_UNUSED
    /// Normal token attribute
    static let normal = LLAMA_TOKEN_ATTR_NORMAL
    /// Control token attribute
    static let control = LLAMA_TOKEN_ATTR_CONTROL
    /// User defined token attribute
    static let userDefined = LLAMA_TOKEN_ATTR_USER_DEFINED
    /// Byte token attribute
    static let byte = LLAMA_TOKEN_ATTR_BYTE
    /// Normalized token attribute
    static let normalized = LLAMA_TOKEN_ATTR_NORMALIZED
    /// Left strip token attribute
    static let leftStrip = LLAMA_TOKEN_ATTR_LSTRIP
    /// Right strip token attribute
    static let rightStrip = LLAMA_TOKEN_ATTR_RSTRIP
    /// Single word token attribute
    static let singleWord = LLAMA_TOKEN_ATTR_SINGLE_WORD
}

public extension SLlamaFileType {
    /// All F32 file type
    static let allF32 = LLAMA_FTYPE_ALL_F32
    /// Mostly F16 file type
    static let mostlyF16 = LLAMA_FTYPE_MOSTLY_F16
    /// Mostly Q4_0 file type
    static let mostlyQ4_0 = LLAMA_FTYPE_MOSTLY_Q4_0
    /// Mostly Q4_1 file type
    static let mostlyQ4_1 = LLAMA_FTYPE_MOSTLY_Q4_1
    /// Mostly Q8_0 file type
    static let mostlyQ8_0 = LLAMA_FTYPE_MOSTLY_Q8_0
    /// Mostly Q5_0 file type
    static let mostlyQ5_0 = LLAMA_FTYPE_MOSTLY_Q5_0
    /// Mostly Q5_1 file type
    static let mostlyQ5_1 = LLAMA_FTYPE_MOSTLY_Q5_1
    /// Mostly Q2_K file type
    static let mostlyQ2_K = LLAMA_FTYPE_MOSTLY_Q2_K
    /// Mostly Q3_K_S file type
    static let mostlyQ3_K_S = LLAMA_FTYPE_MOSTLY_Q3_K_S
    /// Mostly Q3_K_M file type
    static let mostlyQ3_K_M = LLAMA_FTYPE_MOSTLY_Q3_K_M
    /// Mostly Q3_K_L file type
    static let mostlyQ3_K_L = LLAMA_FTYPE_MOSTLY_Q3_K_L
    /// Mostly Q4_K_S file type
    static let mostlyQ4_K_S = LLAMA_FTYPE_MOSTLY_Q4_K_S
    /// Mostly Q4_K_M file type
    static let mostlyQ4_K_M = LLAMA_FTYPE_MOSTLY_Q4_K_M
    /// Mostly Q5_K_S file type
    static let mostlyQ5_K_S = LLAMA_FTYPE_MOSTLY_Q5_K_S
    /// Mostly Q5_K_M file type
    static let mostlyQ5_K_M = LLAMA_FTYPE_MOSTLY_Q5_K_M
    /// Mostly Q6_K file type
    static let mostlyQ6_K = LLAMA_FTYPE_MOSTLY_Q6_K
    /// Mostly IQ2_XXS file type
    static let mostlyIQ2_XXS = LLAMA_FTYPE_MOSTLY_IQ2_XXS
    /// Mostly IQ2_XS file type
    static let mostlyIQ2_XS = LLAMA_FTYPE_MOSTLY_IQ2_XS
    /// Mostly Q2_K_S file type
    static let mostlyQ2_K_S = LLAMA_FTYPE_MOSTLY_Q2_K_S
    /// Mostly IQ3_XS file type
    static let mostlyIQ3_XS = LLAMA_FTYPE_MOSTLY_IQ3_XS
    /// Mostly IQ3_XXS file type
    static let mostlyIQ3_XXS = LLAMA_FTYPE_MOSTLY_IQ3_XXS
    /// Mostly IQ1_S file type
    static let mostlyIQ1_S = LLAMA_FTYPE_MOSTLY_IQ1_S
    /// Mostly IQ4_NL file type
    static let mostlyIQ4_NL = LLAMA_FTYPE_MOSTLY_IQ4_NL
    /// Mostly IQ3_S file type
    static let mostlyIQ3_S = LLAMA_FTYPE_MOSTLY_IQ3_S
    /// Mostly IQ3_M file type
    static let mostlyIQ3_M = LLAMA_FTYPE_MOSTLY_IQ3_M
    /// Mostly IQ2_S file type
    static let mostlyIQ2_S = LLAMA_FTYPE_MOSTLY_IQ2_S
    /// Mostly IQ2_M file type
    static let mostlyIQ2_M = LLAMA_FTYPE_MOSTLY_IQ2_M
    /// Mostly IQ4_XS file type
    static let mostlyIQ4_XS = LLAMA_FTYPE_MOSTLY_IQ4_XS
    /// Mostly IQ1_M file type
    static let mostlyIQ1_M = LLAMA_FTYPE_MOSTLY_IQ1_M
    /// Mostly BF16 file type
    static let mostlyBF16 = LLAMA_FTYPE_MOSTLY_BF16
    /// Mostly TQ1_0 file type
    static let mostlyTQ1_0 = LLAMA_FTYPE_MOSTLY_TQ1_0
    /// Mostly TQ2_0 file type
    static let mostlyTQ2_0 = LLAMA_FTYPE_MOSTLY_TQ2_0
    /// Guessed file type
    static let guessed = LLAMA_FTYPE_GUESSED
}

public extension SLlamaRopeScalingType {
    /// Unspecified RoPE scaling
    static let unspecified = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    /// No RoPE scaling
    static let none = LLAMA_ROPE_SCALING_TYPE_NONE
    /// Linear RoPE scaling
    static let linear = LLAMA_ROPE_SCALING_TYPE_LINEAR
    /// Yarn RoPE scaling
    static let yarn = LLAMA_ROPE_SCALING_TYPE_YARN
    /// LongRoPE scaling
    static let longRope = LLAMA_ROPE_SCALING_TYPE_LONGROPE
    /// Maximum RoPE scaling value
    static let maxValue = LLAMA_ROPE_SCALING_TYPE_MAX_VALUE
}

public extension SLlamaPoolingType {
    /// Unspecified pooling
    static let unspecified = LLAMA_POOLING_TYPE_UNSPECIFIED
    /// No pooling
    static let none = LLAMA_POOLING_TYPE_NONE
    /// Mean pooling
    static let mean = LLAMA_POOLING_TYPE_MEAN
    /// CLS pooling
    static let cls = LLAMA_POOLING_TYPE_CLS
    /// Last pooling
    static let last = LLAMA_POOLING_TYPE_LAST
    /// Rank pooling
    static let rank = LLAMA_POOLING_TYPE_RANK
}

public extension SLlamaAttentionType {
    /// Unspecified attention
    static let unspecified = LLAMA_ATTENTION_TYPE_UNSPECIFIED
    /// Causal attention
    static let causal = LLAMA_ATTENTION_TYPE_CAUSAL
    /// Non-causal attention
    static let nonCausal = LLAMA_ATTENTION_TYPE_NON_CAUSAL
}

public extension SLlamaSplitMode {
    /// No split mode
    static let none = LLAMA_SPLIT_MODE_NONE
    /// Layer split mode
    static let layer = LLAMA_SPLIT_MODE_LAYER
    /// Row split mode
    static let row = LLAMA_SPLIT_MODE_ROW
}

public extension SLlamaModelKvOverrideType {
    /// Integer override type
    static let int = LLAMA_KV_OVERRIDE_TYPE_INT
    /// Float override type
    static let float = LLAMA_KV_OVERRIDE_TYPE_FLOAT
    /// Boolean override type
    static let bool = LLAMA_KV_OVERRIDE_TYPE_BOOL
    /// String override type
    static let string = LLAMA_KV_OVERRIDE_TYPE_STR
}

// MARK: - Error Types

/// Comprehensive error types for SLlama operations
public enum SLlamaError: Error {
    // MARK: - File Operations

    case fileNotFound(String)
    case invalidFormat(String)
    case permissionDenied(String)
    case insufficientSpace
    case corruptedFile(String)
    case fileAccessError(String)

    // MARK: - Model Operations

    case invalidModel(String)
    case incompatibleModel(String)
    case unsupportedArchitecture
    case unsupportedQuantization
    case modelLoadingFailed(String)
    case modelValidationFailed(String)

    // MARK: - Memory Operations

    case outOfMemory
    case insufficientMemory
    case memoryAllocation
    case bufferTooSmall

    // MARK: - Context Operations

    case invalidParameters(String)
    case contextFull
    case inferenceFailure(String)
    case contextCreationFailed(String)
    case contextNotInitialized

    // MARK: - Tokenization Operations

    case invalidVocabulary
    case invalidToken(SLlamaToken)
    case encodingFailure
    case textTooLong
    case tokenizationFailed(String)
    case detokenizationFailed(String)

    // MARK: - Adapter Operations

    case adapterNotFound
    case invalidAdapter(String)
    case incompatibleAdapter
    case adapterLoadingFailed(String)
    case adapterOperationFailed(String)

    // MARK: - State Operations

    case corruptedState
    case incompatibleVersion
    case stateLoadingFailed(String)
    case stateSavingFailed(String)

    // MARK: - Metadata Operations

    case keyNotFound(String)
    case invalidIndex(String)
    case metadataAccessFailed(String)

    // MARK: - Backend Operations

    case backendNotInitialized
    case backendOperationFailed(String)

    // MARK: - Batch Operations

    case invalidBatch(String)
    case batchOperationFailed(String)

    // MARK: - General Operations

    case operationFailed(String)
    case unknownError(Int32)
}

// MARK: - Error Description

extension SLlamaError: LocalizedError {
    public var errorDescription: String? {
        switch self {
            // File Operations
            case let .fileNotFound(path):
                return "File not found: \(path)"
            case let .invalidFormat(details):
                return "Invalid file format: \(details)"
            case let .permissionDenied(path):
                return "Permission denied: \(path)"
            case .insufficientSpace:
                return "Insufficient disk space"
            case let .corruptedFile(path):
                return "Corrupted file: \(path)"
            case let .fileAccessError(details):
                return "File access error: \(details)"
            // Model Operations
            case let .invalidModel(details):
                return "Invalid model: \(details)"
            case let .incompatibleModel(details):
                return "Incompatible model: \(details)"
            case .unsupportedArchitecture:
                return "Unsupported model architecture"
            case .unsupportedQuantization:
                return "Unsupported quantization type"
            case let .modelLoadingFailed(details):
                return "Model loading failed: \(details)"
            case let .modelValidationFailed(details):
                return "Model validation failed: \(details)"
            // Memory Operations
            case .outOfMemory:
                return "Out of memory"
            case .insufficientMemory:
                return "Insufficient memory available"
            case .memoryAllocation:
                return "Memory allocation failed"
            case .bufferTooSmall:
                return "Buffer size too small"
            // Context Operations
            case let .invalidParameters(details):
                return "Invalid parameters: \(details)"
            case .contextFull:
                return "Context is full"
            case let .inferenceFailure(details):
                return "Inference failure: \(details)"
            case let .contextCreationFailed(details):
                return "Context creation failed: \(details)"
            case .contextNotInitialized:
                return "Context not initialized"
            // Tokenization Operations
            case .invalidVocabulary:
                return "Invalid vocabulary"
            case let .invalidToken(token):
                return "Invalid token: \(token)"
            case .encodingFailure:
                return "Text encoding failure"
            case .textTooLong:
                return "Text too long for tokenization"
            case let .tokenizationFailed(details):
                return "Tokenization failed: \(details)"
            case let .detokenizationFailed(details):
                return "Detokenization failed: \(details)"
            // Adapter Operations
            case .adapterNotFound:
                return "Adapter not found"
            case let .invalidAdapter(details):
                return "Invalid adapter: \(details)"
            case .incompatibleAdapter:
                return "Incompatible adapter"
            case let .adapterLoadingFailed(details):
                return "Adapter loading failed: \(details)"
            case let .adapterOperationFailed(details):
                return "Adapter operation failed: \(details)"
            // State Operations
            case .corruptedState:
                return "Corrupted state data"
            case .incompatibleVersion:
                return "Incompatible state version"
            case let .stateLoadingFailed(details):
                return "State loading failed: \(details)"
            case let .stateSavingFailed(details):
                return "State saving failed: \(details)"
            // Metadata Operations
            case let .keyNotFound(key):
                return "Metadata key not found: \(key)"
            case let .invalidIndex(message):
                return "Invalid index: \(message)"
            case let .metadataAccessFailed(details):
                return "Metadata access failed: \(details)"
            // Backend Operations
            case .backendNotInitialized:
                return "Backend not initialized"
            case let .backendOperationFailed(details):
                return "Backend operation failed: \(details)"
            // Batch Operations
            case let .invalidBatch(details):
                return "Invalid batch: \(details)"
            case let .batchOperationFailed(details):
                return "Batch operation failed: \(details)"
            // General Operations
            case let .operationFailed(details):
                return "Operation failed: \(details)"
            case let .unknownError(code):
                return "Unknown error with code: \(code)"
        }
    }
}
