import Foundation
import llama

// MARK: - Core Protocols for Dependency Injection

/// Protocol for SLlama model operations
public protocol PLlamaModel: AnyObject {
    // MARK: - Properties

    /// Get the model pointer for direct C API access
    var pointer: SLlamaModelPointer? { get }

    /// Get model vocabulary
    var vocab: SLlamaVocabPointer? { get }

    /// Get number of embedding dimensions
    var embeddingDimensions: Int32 { get }

    /// Get number of layers
    var layers: Int32 { get }

    /// Get number of attention heads
    var attentionHeads: Int32 { get }

    /// Get number of KV attention heads
    var kvAttentionHeads: Int32 { get }

    /// Get number of parameters
    var parameters: UInt64 { get }

    /// Get model size in bytes
    var size: UInt64 { get }

    /// Get training context length
    var trainingContextLength: Int32 { get }

    /// Get RoPE type
    var ropeType: SLlamaRopeType { get }

    /// Get RoPE frequency scale for training
    var ropeFreqScaleTrain: Float { get }

    // MARK: - Methods

    /// Get metadata value by key
    func getMetadata(key: String, bufferSize: Int) throws -> String

    /// Check if model has embeddings
    func hasEmbeddings() -> Bool

    /// Check if model has encoder
    var hasEncoder: Bool { get }

    /// Check if model has decoder
    var hasDecoder: Bool { get }

    /// Get model description
    func getDescription(bufferSize: Int) -> String?
}

/// Protocol for SLlama context operations
public protocol PLlamaContext: AnyObject {
    // MARK: - Properties

    /// Get the context pointer for direct C API access
    var pointer: SLlamaContextPointer? { get }

    /// Get the associated model
    var associatedModel: PLlamaModel? { get }

    /// Get context size
    var contextSize: UInt32 { get }

    /// Get batch size
    var batchSize: UInt32 { get }

    /// Get maximum batch size
    var maxBatchSize: UInt32 { get }

    /// Get maximum sequence count
    var maxSequences: UInt32 { get }

    /// Get the model from context
    var contextModel: PLlamaModel? { get }

    /// Get memory from context
    var contextMemory: SLlamaMemory? { get }

    /// Get pooling type from context
    var poolingType: SLlamaPoolingType { get }

    // MARK: - Methods

    /// Create a core operations wrapper for this context
    func core() -> PLlamaCore

    /// Encode a batch of tokens
    func encode(_ batch: PLlamaBatch) throws

    /// Decode a batch of tokens
    func decode(_ batch: PLlamaBatch) throws

    /// Set the number of threads
    func setThreads(nThreads: Int32, nThreadsBatch: Int32)

    /// Set embeddings output
    func setEmbeddings(_ embeddings: Bool)

    /// Set causal attention
    func setCausalAttention(_ causalAttn: Bool)

    /// Set warmup mode
    func setWarmup(_ warmup: Bool)

    /// Synchronize computations
    func synchronize()
}

/// Protocol for SLlama core operations (encoding, decoding, and configuration)
public protocol PLlamaCore: AnyObject {
    // MARK: - Properties

    /// Get the model from core context
    var coreModel: PLlamaModel? { get }

    // MARK: - Methods

    /// Encode a batch of tokens
    func encode(_ batch: PLlamaBatch) throws

    /// Decode a batch of tokens
    func decode(_ batch: PLlamaBatch) throws

    /// Set the number of threads
    func setThreads(nThreads: Int32, nThreadsBatch: Int32)

    /// Set embeddings output
    func setEmbeddings(_ embeddings: Bool)

    /// Set causal attention
    func setCausalAttention(_ causalAttn: Bool)

    /// Set warmup mode
    func setWarmup(_ warmup: Bool)

    /// Wait until all computations are finished
    func synchronize()

    /// Get context size
    func getContextSize() -> UInt32

    /// Get batch size
    func getBatchSize() -> UInt32

    /// Get unified batch size
    func getUnifiedBatchSize() -> UInt32

    /// Get maximum sequences
    func getMaxSequences() -> UInt32

    /// Get memory
    func getMemory() -> SLlamaMemory?

    /// Get pooling type
    func getPoolingType() -> SLlamaPoolingType
}

/// Protocol for SLlama batch operations
public protocol PLlamaBatch: AnyObject {
    // MARK: - Properties

    /// Get the underlying C batch structure
    var cBatch: llama_batch { get }

    /// Number of tokens in the batch
    var tokenCount: Int32 { get }

    /// Token array
    var tokens: SLlamaTokenPointer? { get }

    /// Embeddings array
    var embeddings: SLlamaFloatPointer? { get }

    /// Position array for each token
    var positions: SLlamaPositionPointer? { get }

    /// Number of sequence IDs for each token
    var sequenceIdCounts: SLlamaInt32Pointer? { get }

    /// Sequence ID arrays for each token
    var sequenceIds: SLlamaSeqIdPointerPointer? { get }

    /// Logits output flags for each token
    var logits: SLlamaInt8Pointer? { get }

    // MARK: - Methods

    /// Add a token to the batch
    func addToken(_ token: SLlamaToken, position: SLlamaPosition, sequenceIds: [SLlamaSequenceId], logits: Bool)

    /// Clear the batch
    func clear()

    /// Get batch with single token
    static func getSingleTokenBatch(_ token: SLlamaToken) -> PLlamaBatch
}

/// Protocol for SLlama sampler operations
public protocol PLlamaSampler: AnyObject {
    // MARK: - Properties

    /// Get the underlying C sampler pointer
    var cSampler: SLlamaSamplerPointer? { get }

    /// Get the sampler name
    var name: String? { get }

    // MARK: - Methods

    /// Accept a token (updates internal state)
    func accept(_ token: SLlamaToken)

    /// Apply the sampler to token data array
    func apply(to tokenDataArray: SLlamaTokenDataArrayPointer)

    /// Reset the sampler state
    func reset()

    /// Clone the sampler
    func clone() -> PLlamaSampler?

    /// Sample a token
    func sample(_ tokenDataArray: SLlamaTokenDataArrayPointer) -> SLlamaToken
}

/// Protocol for SLlama tokenizer operations
public protocol PLlamaTokenizer {
    // MARK: - Methods

    /// Tokenize text into tokens
    static func tokenize(
        text: String,
        vocab: SLlamaVocabPointer?,
        addSpecial: Bool,
        parseSpecial: Bool
    ) throws
        -> [SLlamaToken]

    /// Detokenize tokens into text
    static func detokenize(
        tokens: [SLlamaToken],
        vocab: SLlamaVocabPointer?,
        renderSpecialTokens: Bool
    ) throws
        -> String

    /// Get token text
    static func getTokenText(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?
    )
        -> String?

    /// Get token type
    static func getTokenType(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?
    )
        -> SLlamaTokenType

    /// Get token attributes
    static func getTokenAttributes(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?
    )
        -> SLlamaTokenAttribute

    /// Check if token is control token
    static func isControlToken(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?
    )
        -> Bool
}

/// Protocol for SLlama memory management
public protocol PLlamaMemoryManager: AnyObject {
    // MARK: - Properties

    /// Get the underlying C memory pointer
    var cMemory: SLlamaMemory? { get }

    // MARK: - Methods

    /// Clear all memory
    func clear(data: Bool)

    /// Remove a sequence from memory
    func removeSequence(_ seqId: SLlamaSequenceId, from pPLlamaQuantization: SLlamaPosition, to p1: SLlamaPosition)
        -> Bool

    /// Copy a sequence in memory
    func copySequence(
        from seqIdSrc: SLlamaSequenceId,
        to seqIdDst: SLlamaSequenceId,
        from pPLlamaQuantization: SLlamaPosition,
        to p1: SLlamaPosition
    )

    /// Keep a sequence in memory
    func keepSequence(_ seqId: SLlamaSequenceId, from pPLlamaQuantization: SLlamaPosition, to p1: SLlamaPosition)

    /// Shift memory by delta
    func shiftMemory(
        _ seqId: SLlamaSequenceId,
        from pPLlamaQuantization: SLlamaPosition,
        to p1: SLlamaPosition,
        delta: SLlamaPosition
    )
}

/// Protocol for SLlama vocabulary operations
public protocol PLlamaVocab: AnyObject {
    // MARK: - Properties

    /// Get the vocabulary pointer
    var pointer: SLlamaVocabPointer? { get }

    /// Get number of tokens in vocabulary
    var tokenCount: Int32 { get }

    /// Get vocabulary type
    var type: SLlamaVocabType { get }

    /// Get special tokens
    var bosToken: SLlamaToken { get }
    var eosToken: SLlamaToken { get }
    var eotToken: SLlamaToken { get }
    var sepToken: SLlamaToken { get }
    var nlToken: SLlamaToken { get }
    var padToken: SLlamaToken { get }
    var maskToken: SLlamaToken { get }
    var clsToken: SLlamaToken { get }
    var unknownToken: SLlamaToken { get }

    // MARK: - Methods

    /// Check if token is end of generation
    func isEog(_ token: SLlamaToken) -> Bool

    /// Check if token is control token
    func isControl(_ token: SLlamaToken) -> Bool

    /// Get token text
    func getTokenText(_ token: SLlamaToken) -> String?

    /// Get token type
    func getTokenType(_ token: SLlamaToken) -> SLlamaTokenType

    /// Get token attributes
    func getTokenAttributes(_ token: SLlamaToken) -> SLlamaTokenAttribute

    /// Get token score
    func getTokenScore(_ token: SLlamaToken) -> Float
}

/// Protocol for SLlama adapter operations
public protocol PLlamaAdapter: AnyObject {
    // MARK: - Properties

    /// Get the underlying C adapter pointer
    var cAdapter: SLlamaAdapterLoraPointer? { get }

    /// Check if the adapter is valid
    var isValid: Bool { get }

    // MARK: - Methods

    /// Apply adapter to context with scale
    func apply(to context: PLlamaContext, scale: Float) -> Bool

    /// Remove adapter from context
    func remove(from context: PLlamaContext) -> Bool

    /// Get adapter information
    func getInfo() -> [String: Any]?
}

/// Protocol for SLlama performance monitoring
public protocol PLlamaPerformance: AnyObject {
    // MARK: - Methods

    /// Benchmark model loading performance
    func benchmarkModelLoading(modelPath: String, iterations: Int) -> SLoadingBenchmarkResults?

    /// Benchmark inference performance
    func benchmarkInference(prompt: String, iterations: Int) -> [String: Any]?

    /// Get context performance metrics
    func getContextMetrics() -> SDetailedContextMetrics?

    /// Get sampler performance metrics
    func getSamplerMetrics() -> SDetailedSamplerMetrics?

    /// Reset performance counters
    func resetMetrics()

    /// Enable/disable performance monitoring
    func setMonitoringEnabled(_ enabled: Bool)
}

/// Protocol for SLlama system information
public protocol PLlamaSystemInfo {
    // MARK: - Methods

    /// Get system capabilities
    func getSystemCapabilities() -> SLlamaSystemCapabilities

    /// Print system information to console
    static func printSystemInfo()

    /// Get current time in microseconds
    static func getCurrentTimeMicroseconds() -> Int64

    /// Get maximum number of available devices
    static func getMaxDevices() -> Int

    /// Check if memory mapping is supported
    static func supportsMmap() -> Bool

    /// Check if memory locking is supported
    static func supportsMlock() -> Bool

    /// Check if GPU offloading is supported
    static func supportsGpuOffload() -> Bool

    /// Check if RPC is supported
    static func supportsRpc() -> Bool
}

/// Protocol for SLlama backend management
public protocol PLlamaBackend: AnyObject {
    // MARK: - Properties

    /// Check if the backend is initialized
    var isInitialized: Bool { get }

    // MARK: - Methods

    /// Initialize the llama backend
    func initialize()

    /// Free the llama backend
    func free()
}

/// Protocol for SLlama quantization operations
public protocol PLlamaQuantization {
    // MARK: - Methods

    /// Quantize a model
    static func quantizeModel(
        inputPath: String,
        outputPath: String,
        params: SLlamaModelQuantizeParams?
    ) throws

    /// Get supported quantization types
    static func getSupportedQuantizationTypes() -> [SLlamaFileType]

    /// Estimate quantized model size
    static func estimateQuantizedSize(
        originalSize: UInt64,
        quantizationType: SLlamaFileType
    )
        -> UInt64

    /// Validate quantization parameters
    static func validateQuantizationParams(_ params: SLlamaModelQuantizeParams) throws
}
