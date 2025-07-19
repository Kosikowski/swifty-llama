# SLlama API Documentation

This document provides detailed API documentation for the SLlama Swift wrapper for llama.cpp.

## Table of Contents

- [Getting Started](#getting-started)
- [Core Classes](#core-classes)
- [Protocol Layer](#protocol-layer)
- [Advanced Usage](#advanced-usage)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)

## Getting Started

### Installation and Setup

```swift
import SLlama

// 1. Initialize the backend (required before any operations)
SLlama.initialize()

// 2. Ensure cleanup when done (use defer for safety)
defer { SLlama.cleanup() }

// 3. Check system capabilities
let supportsMetal = SLlama.supportsMetal()
let maxDevices = SLlama.getMaxDevices()
```

### Basic Model Loading

```swift
// Load a model with default parameters
let model = try SLlamaModel(modelPath: "/path/to/model.gguf")

// Or with custom parameters
var params = SLlamaModelParams()
params.nGpuLayers = 32  // Offload layers to GPU
params.useMetalEnabled = true
let model = try SLlamaModel(modelPath: "/path/to/model.gguf", params: params)
```

## Core Classes

### SLlama - Main Library Interface

The main entry point for the SLlama library.

```swift
public final class SLlama {
    // Backend Management
    public static func initialize()                    // Initialize llama.cpp backend
    public static func cleanup()                      // Free backend resources
    
    // System Information
    public static func getCurrentTime() -> Int64      // Get time in microseconds
    public static func getMaxDevices() -> Int         // Maximum available devices
    public static func getMaxParallelSequences() -> Int  // Max parallel sequences
    
    // Capability Checks
    public static func supportsMmap() -> Bool         // Memory mapping support
    public static func supportsMlock() -> Bool        // Memory locking support
    public static func supportsMetal() -> Bool        // Metal acceleration support
    
    // Logging Control
    public static func disableLogging()              // Disable llama.cpp logging
}
```

**Usage:**
```swift
// Initialize once at app startup
SLlama.initialize()

// Check capabilities
if SLlama.supportsMetal() {
    print("Metal acceleration available")
}

// Disable verbose C library logging
SLlama.disableLogging()

// Cleanup at app shutdown
SLlama.cleanup()
```

### SLlamaModel - Model Management

Represents a loaded language model.

```swift
public class SLlamaModel: PLlamaModel {
    // Initialization
    public init(modelPath: String, params: SLlamaModelParams? = nil) throws
    public init(modelPointer: SLlamaModelPointer) throws  // From existing pointer
    
    // Core Properties
    public var pointer: SLlamaModelPointer? { get }       // Raw C pointer
    public var vocab: SLlamaVocabPointer? { get }         // Model vocabulary
    
    // Model Architecture
    public var embeddingDimensions: Int32 { get }         // Embedding size
    public var layers: Int32 { get }                      // Number of layers
    public var attentionHeads: Int32 { get }              // Attention heads
    public var kvAttentionHeads: Int32 { get }            // KV attention heads
    public var parameters: UInt64 { get }                 // Total parameters
    public var size: UInt64 { get }                       // Model size in bytes
    public var trainingContextLength: Int32 { get }       // Training context
    public var ropeType: SLlamaRopeType { get }           // RoPE configuration
    public var ropeFreqScaleTrain: Float { get }          // RoPE frequency scale
    
    // Capabilities
    public var hasEncoder: Bool { get }                   // Has encoder
    public var hasDecoder: Bool { get }                   // Has decoder
    
    // Methods
    public func getMetadata(key: String, bufferSize: Int) throws -> String
    public func hasEmbeddings() -> Bool
    public func getDescription(bufferSize: Int) -> String?
}
```

**Usage:**
```swift
// Basic model loading
let model = try SLlamaModel(modelPath: "/path/to/model.gguf")

// Check model properties
print("Model has \(model.parameters) parameters")
print("Embedding dimensions: \(model.embeddingDimensions)")
print("Context length: \(model.trainingContextLength)")

// Get model metadata
let architecture = try model.getMetadata(key: "general.architecture", bufferSize: 256)
print("Architecture: \(architecture)")

// Check capabilities
if model.hasEmbeddings() {
    print("Model supports embeddings")
}
```

### SLlamaContext - Inference Context

Manages the inference state and context.

```swift
public class SLlamaContext: PLlamaContext {
    // Initialization
    public init(model: PLlamaModel, contextParams: SLlamaContextParams? = nil) throws
    
    // Core Properties
    public var pointer: SLlamaContextPointer? { get }     // Raw C pointer
    public var associatedModel: PLlamaModel? { get }      // Associated model
    public var contextSize: UInt32 { get }                // Context size
    public var batchSize: UInt32 { get }                  // Batch size
    public var maxBatchSize: UInt32 { get }               // Maximum batch size
    public var maxSequences: UInt32 { get }               // Maximum sequences
    public var contextModel: PLlamaModel? { get }         // Model from context
    public var contextMemory: SLlamaMemory? { get }       // Context memory
    public var poolingType: SLlamaPoolingType { get }     // Pooling type
    
    // Core Methods
    public func inference() -> PLlamaInference            // Create inference wrapper
    public func encode(_ batch: PLlamaBatch) throws       // Encode batch (no KV cache)
    public func decode(_ batch: PLlamaBatch) throws       // Decode batch (uses KV cache)
    
    // Configuration
    public func setThreads(nThreads: Int32, nThreadsBatch: Int32)
    public func setEmbeddings(_ embeddings: Bool)
    public func setCausalAttention(_ causalAttn: Bool)
    public func setWarmup(_ warmup: Bool)
    public func synchronize()                             // Wait for completion
}
```

**Usage:**
```swift
// Create context with default parameters
let context = try SLlamaContext(model: model)

// Or with custom parameters
var contextParams = SLlamaContextParams()
contextParams.nCtx = 2048      // Context size
contextParams.nBatch = 512     // Batch size
contextParams.nThreads = 8     // Thread count
let context = try SLlamaContext(model: model, contextParams: contextParams)

// Configure context
context.setThreads(nThreads: 8, nThreadsBatch: 8)
context.setEmbeddings(true)  // Enable embeddings output

// Create inference wrapper
let inference = context.inference()
```

### SLlamaInference - Inference Operations

Handles the actual inference operations.

```swift
public class SLlamaInference: PLlamaInference {
    // Initialization
    public init(context: SLlamaContext)
    
    // Properties
    public var inferenceModel: PLlamaModel? { get }       // Get model from context
    
    // Core Operations
    public func encode(_ batch: PLlamaBatch) throws       // Encode tokens
    public func decode(_ batch: PLlamaBatch) throws       // Decode tokens
    
    // Configuration
    public func setThreads(nThreads: Int32, nThreadsBatch: Int32)
    public func setEmbeddings(_ embeddings: Bool)
    public func setCausalAttention(_ causalAttn: Bool)
    public func setWarmup(_ warmup: Bool)
    public func synchronize()                             // Wait for completion
    
    // Information
    public func getContextSize() -> UInt32
    public func getBatchSize() -> UInt32
    public func getUnifiedBatchSize() -> UInt32
    public func getMaxSequences() -> UInt32
    public func getMemory() -> SLlamaMemory?
    public func getPoolingType() -> SLlamaPoolingType
}
```

**Usage:**
```swift
let inference = context.inference()

// Configure inference
inference.setThreads(nThreads: 8, nThreadsBatch: 8)
inference.setEmbeddings(false)  // Disable embeddings for faster inference
inference.setCausalAttention(true)  // Enable causal attention

// Run inference
try inference.decode(batch)

// Wait for completion
inference.synchronize()
```

### SLlamaBatch - Token Batching

Manages batches of tokens for efficient processing.

```swift
public class SLlamaBatch: PLlamaBatch {
    // Initialization
    public init(nTokens: Int32, embd: Int32 = 0, nSeqMax: Int32)
    
    // Properties
    public var cBatch: llama_batch { get }                // Raw C batch structure
    public var tokenCount: Int32 { get }                  // Number of tokens
    public var tokens: SLlamaTokenPointer? { get }        // Token array
    public var embeddings: SLlamaFloatPointer? { get }    // Embeddings array
    public var positions: SLlamaPositionPointer? { get }  // Position array
    public var sequenceIdCounts: SLlamaInt32Pointer? { get } // Sequence ID counts
    public var sequenceIds: SLlamaSeqIdPointerPointer? { get } // Sequence ID arrays
    public var logits: SLlamaInt8Pointer? { get }         // Logits flags
    
    // Methods
    public func addToken(_ token: SLlamaToken, position: SLlamaPosition, 
                        sequenceIds: [SLlamaSequenceId], logits: Bool)
    public func clear()                                   // Clear batch
    public static func getSingleTokenBatch(_ token: SLlamaToken) -> PLlamaBatch
}
```

**Usage:**
```swift
// Create a batch for up to 512 tokens with 1 sequence max
let batch = SLlamaBatch(nTokens: 512, nSeqMax: 1)

// Add tokens to the batch
let tokens = [123, 456, 789]  // Example token IDs
for (index, token) in tokens.enumerated() {
    let isLast = index == tokens.count - 1
    batch.addToken(
        token, 
        position: Int32(index), 
        sequenceIds: [0],       // Sequence ID 0
        logits: isLast          // Only compute logits for last token
    )
}

// Process the batch
try inference.decode(batch)

// Clear for reuse
batch.clear()

// Quick single token batch
let singleBatch = SLlamaBatch.getSingleTokenBatch(123)
```

### SLlamaTokenizer - Text Processing

Handles tokenization and detokenization operations.

```swift
public class SLlamaTokenizer: PLlamaTokenizer {
    // Core Tokenization
    public static func tokenize(
        text: String,
        vocab: SLlamaVocabPointer?,
        addSpecial: Bool = true,
        parseSpecial: Bool = true
    ) throws -> [SLlamaToken]
    
    public static func detokenize(
        tokens: [SLlamaToken],
        vocab: SLlamaVocabPointer?,
        removeSpecial: Bool = false,
        unparseSpecial: Bool = false
    ) throws -> String
    
    // Protocol Methods
    public static func detokenize(
        tokens: [SLlamaToken],
        vocab: SLlamaVocabPointer?,
        renderSpecialTokens: Bool
    ) throws -> String
    
    // Token Analysis
    public static func getTokenText(token: SLlamaToken, vocab: SLlamaVocabPointer?) -> String?
    public static func getTokenType(token: SLlamaToken, vocab: SLlamaVocabPointer?) -> SLlamaTokenType
    public static func getTokenAttributes(token: SLlamaToken, vocab: SLlamaVocabPointer?) -> SLlamaTokenAttribute
    public static func isControlToken(token: SLlamaToken, vocab: SLlamaVocabPointer?) -> Bool
    
    // Utility Methods
    public static func tokenToPiece(token: SLlamaToken, vocab: SLlamaVocabPointer?, 
                                   bufferSize: Int32) throws -> String
    
    // Chat Templates
    public static func applyChatTemplate(template: String?, messages: [ChatMessage]) throws -> String
    public static func getBuiltinTemplates() throws -> [String]
}
```

**Usage:**
```swift
let text = "Hello, world! How are you today?"
let vocab = model.vocab

// Tokenize text
let tokens = try SLlamaTokenizer.tokenize(
    text: text,
    vocab: vocab,
    addSpecial: true,     // Add BOS/EOS tokens
    parseSpecial: true    // Parse special tokens
)

print("Text tokenized into \(tokens.count) tokens: \(tokens)")

// Analyze tokens
for (index, token) in tokens.enumerated() {
    let text = SLlamaTokenizer.getTokenText(token: token, vocab: vocab)
    let type = SLlamaTokenizer.getTokenType(token: token, vocab: vocab)
    let isControl = SLlamaTokenizer.isControlToken(token: token, vocab: vocab)
    
    print("Token[\(index)]: \(token) → '\(text ?? "nil")' (type: \(type), control: \(isControl))")
}

// Detokenize back to text
let reconstructed = try SLlamaTokenizer.detokenize(
    tokens: tokens,
    vocab: vocab,
    removeSpecial: true,  // Remove BOS/EOS tokens
    unparseSpecial: true  // Convert special tokens to text
)

print("Reconstructed: '\(reconstructed)'")

// Chat template usage
let messages = [
    ChatMessage(role: "user", content: "Hello!")
]
let chatText = try SLlamaTokenizer.applyChatTemplate(template: nil, messages: messages)
```

### SLlamaSampler - Sampling Strategies

Provides various sampling strategies for token generation.

```swift
public class SLlamaSampler: PLlamaSampler {
    // Initialization
    public init(context: SLlamaContext)
    
    // Properties
    public var cSampler: SLlamaSamplerPointer? { get }    // Raw C sampler
    public var name: String? { get }                      // Sampler name
    
    // Core Methods
    public func accept(_ token: SLlamaToken)              // Accept token
    public func apply(to tokenDataArray: SLlamaTokenDataArrayPointer) // Apply sampling
    public func reset()                                   // Reset state
    public func clone() -> PLlamaSampler?                 // Clone sampler
    public func sample(_ tokenDataArray: SLlamaTokenDataArrayPointer) -> SLlamaToken
    
    // Factory Methods - Basic Sampling
    public static func greedy() -> SLlamaSampler?
    public static func temperature(_ temperature: Float) -> SLlamaSampler?
    public static func topK(_ k: Int32) -> SLlamaSampler?
    public static func topP(_ p: Float) -> SLlamaSampler?
    
    // Factory Methods - Advanced Sampling
    public static func minP(_ p: Float) -> SLlamaSampler?
    public static func typical(_ p: Float) -> SLlamaSampler?
    public static func mirostat(_ tau: Float, eta: Float, m: Int32) -> SLlamaSampler?
    public static func mirostatV2(_ tau: Float, eta: Float) -> SLlamaSampler?
    public static func xtc(_ threshold: Float, probability: Float, minKeep: size_t) -> SLlamaSampler?
    public static func topNSigma(_ n: Int32, sigma: Float, minKeep: size_t) -> SLlamaSampler?
    
    // Utility Methods
    public static func getSeed(_ sampler: SLlamaSamplerPointer?) -> UInt32
    public static func sampleFromIndex(_ sampler: SLlamaSamplerPointer?, 
                                      tokenDataArray: SLlamaTokenDataArrayPointer, 
                                      index: Int32) -> SLlamaToken
}
```

**Usage:**
```swift
let context = try SLlamaContext(model: model)

// Create different sampling strategies
let greedySampler = SLlamaSampler.greedy()
let tempSampler = SLlamaSampler.temperature(0.8)
let topKSampler = SLlamaSampler.topK(40)
let topPSampler = SLlamaSampler.topP(0.95)
let minPSampler = SLlamaSampler.minP(0.05)

// Advanced samplers
let mirostatSampler = SLlamaSampler.mirostat(5.0, eta: 0.1, m: 100)
let xtcSampler = SLlamaSampler.xtc(threshold: 0.1, probability: 0.5, minKeep: 1)

// Use sampler
let sampler = tempSampler!
sampler.apply(to: tokenDataArray)
let selectedToken = sampler.sample(tokenDataArray)

// Accept the token (updates sampler state)
sampler.accept(selectedToken)

// Reset sampler state
sampler.reset()

// Clone sampler for parallel use
let clonedSampler = sampler.clone()
```

### SLlamaVocab - Vocabulary Management

Provides access to model vocabulary information.

```swift
public class SLlamaVocab: PLlamaVocab {
    // Initialization
    public init(vocab: SLlamaVocabPointer?)
    
    // Properties
    public var pointer: SLlamaVocabPointer? { get }       // Raw C vocab pointer
    public var tokenCount: Int32 { get }                  // Number of tokens
    public var type: SLlamaVocabType { get }              // Vocabulary type
    
    // Special Tokens
    public var bosToken: SLlamaToken { get }              // Beginning of sentence
    public var eosToken: SLlamaToken { get }              // End of sentence
    public var eotToken: SLlamaToken { get }              // End of turn
    public var sepToken: SLlamaToken { get }              // Separator
    public var nlToken: SLlamaToken { get }               // Newline
    public var padToken: SLlamaToken { get }              // Padding
    public var maskToken: SLlamaToken { get }             // Mask
    public var clsToken: SLlamaToken { get }              // Classification
    public var unknownToken: SLlamaToken { get }          // Unknown
    
    // Convenience Methods
    public func tokenize(text: String, addSpecial: Bool, parseSpecial: Bool) throws -> [SLlamaToken]
    public func detokenize(tokens: [SLlamaToken], removeSpecial: Bool, unparseSpecial: Bool) throws -> String
    
    // Token Information
    public func isEog(_ token: SLlamaToken) -> Bool       // Is end of generation
    public func isControl(_ token: SLlamaToken) -> Bool   // Is control token
    public func getTokenText(_ token: SLlamaToken) -> String?
    public func getTokenType(_ token: SLlamaToken) -> SLlamaTokenType
    public func getTokenAttributes(_ token: SLlamaToken) -> SLlamaTokenAttribute
    public func getTokenScore(_ token: SLlamaToken) -> Float
}
```

**Usage:**
```swift
let vocab = SLlamaVocab(vocab: model.vocab)

// Vocabulary information
print("Vocabulary has \(vocab.tokenCount) tokens")
print("Vocabulary type: \(vocab.type)")

// Special tokens
print("BOS token: \(vocab.bosToken)")
print("EOS token: \(vocab.eosToken)")
print("Newline token: \(vocab.nlToken)")

// Convenience tokenization
let tokens = try vocab.tokenize(text: "Hello world", addSpecial: true, parseSpecial: true)
let text = try vocab.detokenize(tokens: tokens, removeSpecial: false, unparseSpecial: true)

// Token analysis
let token = tokens.first!
let tokenText = vocab.getTokenText(token)
let tokenType = vocab.getTokenType(token)
let tokenScore = vocab.getTokenScore(token)
let isControl = vocab.isControl(token)
let isEog = vocab.isEog(token)

print("Token \(token): '\(tokenText ?? "nil")' (type: \(tokenType), score: \(tokenScore), control: \(isControl), eog: \(isEog))")
```

## Protocol Layer

SLlama uses protocols for dependency injection and testing:

### Core Protocols

```swift
public protocol PLlamaModel: AnyObject { /* ... */ }
public protocol PLlamaContext: AnyObject { /* ... */ }
public protocol PLlamaInference: AnyObject { /* ... */ }
public protocol PLlamaBatch: AnyObject { /* ... */ }
public protocol PLlamaSampler: AnyObject { /* ... */ }
public protocol PLlamaTokenizer { /* ... */ }
public protocol PLlamaVocab: AnyObject { /* ... */ }
```

### Usage in Testable Code

```swift
// Use protocols for dependency injection
func generateText(model: PLlamaModel, context: PLlamaContext, input: String) -> String {
    let inference = context.inference()
    let vocab = SLlamaVocab(vocab: model.vocab)
    
    // Tokenize input
    let tokens = try! vocab.tokenize(text: input, addSpecial: true, parseSpecial: true)
    
    // Create batch
    let batch = SLlamaBatch(nTokens: 512, nSeqMax: 1)
    for (index, token) in tokens.enumerated() {
        batch.addToken(token, position: Int32(index), sequenceIds: [0], 
                      logits: index == tokens.count - 1)
    }
    
    // Run inference
    try! inference.decode(batch)
    
    // Return generated text (simplified)
    return "Generated text"
}

// Easy to mock for testing
class MockModel: PLlamaModel {
    var pointer: SLlamaModelPointer? = nil
    var vocab: SLlamaVocabPointer? = nil
    var embeddingDimensions: Int32 = 512
    // ... implement other protocol methods
}

class MockContext: PLlamaContext {
    var pointer: SLlamaContextPointer? = nil
    var associatedModel: PLlamaModel? = MockModel()
    // ... implement other protocol methods
}

// Test with mocks
let mockModel = MockModel()
let mockContext = MockContext()
let result = generateText(model: mockModel, context: mockContext, input: "test")
```

## Advanced Usage

### Complete Text Generation Example

```swift
func generateText(model: PLlamaModel, prompt: String, maxTokens: Int = 100) throws -> String {
    // Create context
    let context = try SLlamaContext(model: model)
    let inference = context.inference()
    let vocab = SLlamaVocab(vocab: model.vocab)
    
    // Configure for text generation
    context.setThreads(nThreads: 8, nThreadsBatch: 8)
    context.setEmbeddings(false)
    
    // Tokenize prompt
    let promptTokens = try SLlamaTokenizer.tokenize(
        text: prompt,
        vocab: model.vocab,
        addSpecial: true,
        parseSpecial: true
    )
    
    // Process prompt
    let batch = SLlamaBatch(nTokens: 512, nSeqMax: 1)
    for (index, token) in promptTokens.enumerated() {
        batch.addToken(
            token,
            position: Int32(index),
            sequenceIds: [0],
            logits: index == promptTokens.count - 1
        )
    }
    
    try inference.decode(batch)
    
    // Generate tokens
    var generatedTokens: [SLlamaToken] = []
    let sampler = SLlamaSampler.temperature(0.8)!
    
    for position in promptTokens.count..<(promptTokens.count + maxTokens) {
        // Get logits
        let logits = SLlamaLogits(context: context)
        guard let logitsPtr = logits.getLogits() else { break }
        
        // Create token data array (simplified - in real use, create properly)
        var tokenDataArray = SLlamaTokenDataArray()
        // ... populate token data array from logits
        
        // Sample next token
        let nextToken = sampler.sample(&tokenDataArray)
        generatedTokens.append(nextToken)
        
        // Check for EOS
        if nextToken == vocab.eosToken {
            break
        }
        
        // Accept token
        sampler.accept(nextToken)
        
        // Prepare next batch
        batch.clear()
        batch.addToken(nextToken, position: Int32(position), sequenceIds: [0], logits: true)
        
        try inference.decode(batch)
    }
    
    // Detokenize result
    let allTokens = promptTokens + generatedTokens
    return try SLlamaTokenizer.detokenize(
        tokens: allTokens,
        vocab: model.vocab,
        removeSpecial: true,
        unparseSpecial: true
    )
}
```

### Memory Management Example

```swift
func manageMemory(context: SLlamaContext) {
    let memoryManager = SLlamaMemoryManager(context: context)
    
    // Clear all memory
    memoryManager.clear(data: true)
    
    // Manage sequences
    memoryManager.removeSequence(0, from: 0, to: 100)  // Remove first 100 tokens of sequence 0
    memoryManager.copySequence(from: 0, to: 1, from: 0, to: 50)  // Copy first 50 tokens to new sequence
    
    // Keep sequences up to a certain length
    memoryManager.keepSequence(0, from: 50, to: -1)  // Keep from position 50 to end
    
    // Shift sequence positions
    memoryManager.shiftSequence(0, p0: 0, p1: -1, delta: 10)  // Shift all positions by 10
}
```

### Performance Monitoring

```swift
func monitorPerformance(context: SLlamaContext) {
    let performance = SLlamaPerformance(context: context)
    
    // Enable monitoring
    performance.setMonitoringEnabled(true)
    
    // Benchmark model loading
    let loadingResults = performance.benchmarkModelLoading(
        modelPath: "/path/to/model.gguf",
        iterations: 5
    )
    
    if let results = loadingResults {
        print("Average loading time: \(results.averageTime)s")
        print("Memory usage: \(results.memoryUsage) MB")
    }
    
    // Get detailed metrics
    if let contextMetrics = performance.getContextMetrics() {
        print("Context performance:")
        print("- Eval time: \(contextMetrics.evalTime)ms")
        print("- Tokens per second: \(contextMetrics.tokensPerSecond)")
    }
    
    if let samplerMetrics = performance.getSamplerMetrics() {
        print("Sampler performance:")
        print("- Sample time: \(samplerMetrics.sampleTime)ms")
        print("- Accept time: \(samplerMetrics.acceptTime)ms")
    }
    
    // Reset metrics
    performance.resetMetrics()
}
```

## Error Handling

SLlama provides detailed error handling through the `SLlamaError` enum:

```swift
enum SLlamaError: Error {
    // File Operations
    case fileNotFound(String)
    case invalidFormat(String)
    case permissionDenied(String)
    case insufficientSpace
    case corruptedFile(String)
    case fileAccessError(String)
    
    // Model Operations
    case invalidModel(String)
    case incompatibleModel(String)
    case unsupportedArchitecture
    case unsupportedQuantization
    case modelLoadingFailed(String)
    case modelValidationFailed(String)
    
    // Memory Operations
    case outOfMemory
    case insufficientMemory
    case memoryAllocation
    case bufferTooSmall
    
    // Context Operations
    case invalidParameters(String)
    case contextFull
    case inferenceFailure(String)
    case contextCreationFailed(String)
    case contextNotInitialized
    
    // Tokenization Operations
    case invalidVocabulary
    case invalidToken(SLlamaToken)
    case encodingFailure
    case textTooLong
    case tokenizationFailed(String)
    case detokenizationFailed(String)
    
    // And many more...
}
```

### Error Handling Best Practices

```swift
do {
    let model = try SLlamaModel(modelPath: "/path/to/model.gguf")
    let context = try SLlamaContext(model: model)
    
    let tokens = try SLlamaTokenizer.tokenize(
        text: "Hello world",
        vocab: model.vocab,
        addSpecial: true,
        parseSpecial: true
    )
    
} catch SLlamaError.fileNotFound(let path) {
    print("Model file not found: \(path)")
} catch SLlamaError.invalidModel(let reason) {
    print("Invalid model: \(reason)")
} catch SLlamaError.outOfMemory {
    print("Out of memory - try reducing context size or model size")
} catch SLlamaError.tokenizationFailed(let reason) {
    print("Tokenization failed: \(reason)")
} catch {
    print("Unexpected error: \(error)")
}
```

## Performance Optimization

### Threading Configuration

```swift
// Optimal thread configuration depends on your hardware
let cpuCount = ProcessInfo.processInfo.processorCount
let optimalThreads = min(cpuCount, 8)  // Don't exceed 8 threads typically

context.setThreads(nThreads: Int32(optimalThreads), nThreadsBatch: Int32(optimalThreads))
```

### Memory Management

```swift
// Use appropriate batch sizes
let batchSize = min(512, context.maxBatchSize)  // Don't exceed context limits
let batch = SLlamaBatch(nTokens: batchSize, nSeqMax: 1)

// Clear batches when done
batch.clear()

// Manage context memory
let memoryManager = SLlamaMemoryManager(context: context)
memoryManager.clear(data: false)  // Clear metadata only, keep data
```

### GPU Acceleration

```swift
// Check Metal support
if SLlama.supportsMetal() {
    var modelParams = SLlamaModelParams()
    modelParams.nGpuLayers = 32  // Offload layers to GPU
    modelParams.useMetalEnabled = true
    
    let model = try SLlamaModel(modelPath: "/path/to/model.gguf", params: modelParams)
}
```

## Examples

### Chat Application

```swift
class ChatSession {
    private let model: PLlamaModel
    private let context: PLlamaContext
    private let inference: PLlamaInference
    private let vocab: SLlamaVocab
    private var conversationTokens: [SLlamaToken] = []
    
    init(modelPath: String) throws {
        self.model = try SLlamaModel(modelPath: modelPath)
        self.context = try SLlamaContext(model: model)
        self.inference = context.inference()
        self.vocab = SLlamaVocab(vocab: model.vocab)
        
        // Configure for chat
        context.setThreads(nThreads: 8, nThreadsBatch: 8)
        context.setEmbeddings(false)
    }
    
    func addMessage(role: String, content: String) throws -> String {
        // Format as chat message
        let message = "[\(role.uppercased())]: \(content)\n[ASSISTANT]: "
        
        // Tokenize new message
        let newTokens = try SLlamaTokenizer.tokenize(
            text: message,
            vocab: model.vocab,
            addSpecial: conversationTokens.isEmpty,  // Only add special tokens at start
            parseSpecial: true
        )
        
        conversationTokens.append(contentsOf: newTokens)
        
        // Generate response
        return try generateResponse()
    }
    
    private func generateResponse() throws -> String {
        let sampler = SLlamaSampler.temperature(0.7)!
        var responseTokens: [SLlamaToken] = []
        
        // Process conversation context
        let batch = SLlamaBatch(nTokens: 2048, nSeqMax: 1)
        for (index, token) in conversationTokens.enumerated() {
            batch.addToken(
                token,
                position: Int32(index),
                sequenceIds: [0],
                logits: index == conversationTokens.count - 1
            )
        }
        
        try inference.decode(batch)
        
        // Generate response
        for position in conversationTokens.count..<(conversationTokens.count + 200) {
            let logits = SLlamaLogits(context: context)
            guard let logitsPtr = logits.getLogits() else { break }
            
            // Sample next token (simplified)
            var tokenDataArray = SLlamaTokenDataArray()
            // ... populate from logits
            
            let nextToken = sampler.sample(&tokenDataArray)
            responseTokens.append(nextToken)
            
            // Check for stop conditions
            if nextToken == vocab.eosToken || 
               (responseTokens.count > 3 && 
                try SLlamaTokenizer.detokenize(tokens: Array(responseTokens.suffix(4)), 
                                             vocab: model.vocab, 
                                             removeSpecial: true, 
                                             unparseSpecial: true).contains("\n[")) {
                break
            }
            
            sampler.accept(nextToken)
            
            // Continue generation
            batch.clear()
            batch.addToken(nextToken, position: Int32(position), sequenceIds: [0], logits: true)
            try inference.decode(batch)
        }
        
        // Add response to conversation
        conversationTokens.append(contentsOf: responseTokens)
        
        // Convert to text
        return try SLlamaTokenizer.detokenize(
            tokens: responseTokens,
            vocab: model.vocab,
            removeSpecial: true,
            unparseSpecial: true
        ).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// Usage
let chat = try ChatSession(modelPath: "/path/to/chat-model.gguf")
let response1 = try chat.addMessage(role: "user", content: "Hello! How are you?")
let response2 = try chat.addMessage(role: "user", content: "Tell me a joke.")
```

### Embedding Generation

```swift
func generateEmbeddings(model: PLlamaModel, texts: [String]) throws -> [[Float]] {
    let context = try SLlamaContext(model: model)
    let inference = context.inference()
    
    // Enable embeddings
    context.setEmbeddings(true)
    
    var embeddings: [[Float]] = []
    
    for text in texts {
        // Tokenize text
        let tokens = try SLlamaTokenizer.tokenize(
            text: text,
            vocab: model.vocab,
            addSpecial: true,
            parseSpecial: true
        )
        
        // Create batch
        let batch = SLlamaBatch(nTokens: 512, nSeqMax: 1)
        for (index, token) in tokens.enumerated() {
            batch.addToken(
                token,
                position: Int32(index),
                sequenceIds: [0],
                logits: false  // Don't need logits for embeddings
            )
        }
        
        // Run inference
        try inference.encode(batch)  // Use encode for embeddings
        
        // Get embeddings
        let logits = SLlamaLogits(context: context)
        if let embeddingsPtr = logits.getEmbeddings() {
            let embeddingSize = Int(model.embeddingDimensions)
            let embeddingArray = Array(UnsafeBufferPointer(start: embeddingsPtr, count: embeddingSize))
            embeddings.append(embeddingArray)
        }
    }
    
    return embeddings
}

// Usage
let texts = ["Hello world", "Machine learning", "Natural language processing"]
let embeddings = try generateEmbeddings(model: model, texts: texts)
print("Generated \(embeddings.count) embeddings of size \(embeddings.first?.count ?? 0)")
```

This documentation provides a foundation for using the SLlama API effectively. For more specific use cases or advanced features, refer to the source code and tests for additional examples. 