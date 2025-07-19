# SLlama

A Swift wrapper for the [llama.cpp](https://github.com/ggerganov/llama.cpp) library, providing a native Swift interface to run Large Language Models (LLMs) on Apple platforms.

## Features

- 🚀 Native Swift interface to llama.cpp
- 📱 Support for iOS, macOS, tvOS, and visionOS
- 🔧 Metal acceleration support
- 🎯 Easy-to-use Swift API
- 📦 Swift Package Manager integration
- 🔮 Omen logging framework integration
- 🧪 Protocol-oriented design for dependency injection and testing
- ⚡ Advanced sampling strategies (Mirostat, XTC, Min-P, etc.)
- 🎛️ LoRA adapter support
- 📊 Performance monitoring and benchmarking

## Requirements

- iOS 13.0+
- macOS 11.0+
- tvOS 13.0+
- visionOS 1.0+
- Swift 5.9+

## Installation

### Swift Package Manager

Add SLlama to your project in Xcode:

1. File → Add Package Dependencies
2. Enter the repository URL
3. Select the package and add it to your target

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/your-username/swifty-llama-cpp.git", from: "1.0.0")
]
```

## Quick Start

### Basic Setup

```swift
import SLlama

// Initialize the library
SLlama.initialize()
defer { SLlama.cleanup() }

// Load a model
let model = try SLlamaModel(modelPath: "/path/to/your/model.gguf")

// Create a context
let context = try SLlamaContext(model: model)

// Create an inference wrapper
let inference = context.inference()
```

### Simple Text Generation

```swift
// Tokenize input text
let inputText = "Hello, world!"
let tokens = try SLlamaTokenizer.tokenize(
    text: inputText,
    vocab: model.vocab,
    addSpecial: true,
    parseSpecial: true
)

// Create a batch
let batch = SLlamaBatch(nTokens: 512, nSeqMax: 1)
for (index, token) in tokens.enumerated() {
    batch.addToken(token, position: Int32(index), sequenceIds: [0], logits: index == tokens.count - 1)
}

// Run inference
try inference.decode(batch)

// Get logits and sample next token
let logits = SLlamaLogits(context: context)
// ... sampling logic here
```

## Core API Reference

### 🏗️ SLlama - Main Library Interface

```swift
class SLlama {
    static func initialize()                    // Initialize backend
    static func cleanup()                      // Free backend
    static func disableLogging()              // Disable C library logging
    static func getCurrentTime() -> Int64      // Get time in microseconds
    static func getMaxDevices() -> Int         // Get max devices
    static func supportsMmap() -> Bool         // Check mmap support
    static func supportsMlock() -> Bool        // Check mlock support
    static func supportsMetal() -> Bool        // Check Metal support
}
```

### 🧠 SLlamaModel - Model Management

```swift
class SLlamaModel: PLlamaModel {
    // Initialization
    init(modelPath: String, params: SLlamaModelParams? = nil)
    
    // Properties
    var pointer: SLlamaModelPointer?          // Raw C pointer
    var vocab: SLlamaVocabPointer?            // Model vocabulary
    var embeddingDimensions: Int32            // Embedding size
    var layers: Int32                         // Number of layers
    var attentionHeads: Int32                 // Attention heads
    var parameters: UInt64                    // Parameter count
    var size: UInt64                          // Model size in bytes
    var trainingContextLength: Int32          // Training context length
    
    // Methods
    func getMetadata(key: String, bufferSize: Int) throws -> String
    func hasEmbeddings() -> Bool
    func getDescription(bufferSize: Int) -> String?
}
```

### 🎯 SLlamaContext - Inference Context

```swift
class SLlamaContext: PLlamaContext {
    // Initialization
    init(model: PLlamaModel, contextParams: SLlamaContextParams? = nil)
    
    // Properties
    var pointer: SLlamaContextPointer?        // Raw C pointer
    var associatedModel: PLlamaModel?         // Associated model
    var contextSize: UInt32                   // Context size
    var batchSize: UInt32                     // Batch size
    var maxBatchSize: UInt32                  // Max batch size
    var maxSequences: UInt32                  // Max sequences
    
    // Methods
    func inference() -> PLlamaInference       // Create inference wrapper
    func encode(_ batch: PLlamaBatch) throws  // Encode batch (no KV cache)
    func decode(_ batch: PLlamaBatch) throws  // Decode batch (uses KV cache)
    func setThreads(nThreads: Int32, nThreadsBatch: Int32)
    func setEmbeddings(_ embeddings: Bool)
    func synchronize()                        // Wait for computations
}
```

### ⚡ SLlamaInference - Inference Operations

```swift
class SLlamaInference: PLlamaInference {
    // Initialization
    init(context: SLlamaContext)
    
    // Properties
    var inferenceModel: PLlamaModel?          // Get model from context
    
    // Methods
    func encode(_ batch: PLlamaBatch) throws  // Encode tokens
    func decode(_ batch: PLlamaBatch) throws  // Decode tokens
    func setThreads(nThreads: Int32, nThreadsBatch: Int32)
    func setEmbeddings(_ embeddings: Bool)
    func setWarmup(_ warmup: Bool)
    func synchronize()                        // Wait for completion
    func getContextSize() -> UInt32
    func getBatchSize() -> UInt32
    func getMemory() -> SLlamaMemory?
}
```

### 📦 SLlamaBatch - Token Batching

```swift
class SLlamaBatch: PLlamaBatch {
    // Initialization
    init(nTokens: Int32, embd: Int32 = 0, nSeqMax: Int32)
    
    // Properties
    var cBatch: llama_batch                   // Raw C batch
    var tokenCount: Int32                     // Number of tokens
    var tokens: SLlamaTokenPointer?           // Token array
    var positions: SLlamaPositionPointer?     // Position array
    var sequenceIds: SLlamaSeqIdPointerPointer? // Sequence ID arrays
    var logits: SLlamaInt8Pointer?            // Logits flags
    
    // Methods
    func addToken(_ token: SLlamaToken, position: SLlamaPosition, 
                 sequenceIds: [SLlamaSequenceId], logits: Bool)
    func clear()                              // Clear batch
    static func getSingleTokenBatch(_ token: SLlamaToken) -> PLlamaBatch
}
```

### 📝 SLlamaTokenizer - Text Processing

```swift
class SLlamaTokenizer: PLlamaTokenizer {
    // Tokenization
    static func tokenize(text: String, vocab: SLlamaVocabPointer?, 
                        addSpecial: Bool, parseSpecial: Bool) throws -> [SLlamaToken]
    
    static func detokenize(tokens: [SLlamaToken], vocab: SLlamaVocabPointer?,
                          removeSpecial: Bool, unparseSpecial: Bool) throws -> String
    
    // Token analysis
    static func getTokenText(token: SLlamaToken, vocab: SLlamaVocabPointer?) -> String?
    static func getTokenType(token: SLlamaToken, vocab: SLlamaVocabPointer?) -> SLlamaTokenType
    static func getTokenAttributes(token: SLlamaToken, vocab: SLlamaVocabPointer?) -> SLlamaTokenAttribute
    static func isControlToken(token: SLlamaToken, vocab: SLlamaVocabPointer?) -> Bool
    
    // Chat templates
    static func applyChatTemplate(template: String?, messages: [ChatMessage]) throws -> String
    static func getBuiltinTemplates() throws -> [String]
}
```

### 🎲 SLlamaSampler - Sampling Strategies

```swift
class SLlamaSampler: PLlamaSampler {
    // Initialization
    init(context: SLlamaContext)
    
    // Properties
    var cSampler: SLlamaSamplerPointer?       // Raw C sampler
    var name: String?                         // Sampler name
    
    // Methods
    func accept(_ token: SLlamaToken)         // Accept token
    func apply(to tokenDataArray: SLlamaTokenDataArrayPointer) // Apply sampling
    func reset()                              // Reset state
    func clone() -> PLlamaSampler?            // Clone sampler
    func sample(_ tokenDataArray: SLlamaTokenDataArrayPointer) -> SLlamaToken
    
    // Static factory methods for different sampling strategies
    static func greedy() -> SLlamaSampler?
    static func temperature(_ temperature: Float) -> SLlamaSampler?
    static func topK(_ k: Int32) -> SLlamaSampler?
    static func topP(_ p: Float) -> SLlamaSampler?
    static func minP(_ p: Float) -> SLlamaSampler?
    static func typical(_ p: Float) -> SLlamaSampler?
    static func mirostat(_ tau: Float, eta: Float, m: Int32) -> SLlamaSampler?
    static func mirostatV2(_ tau: Float, eta: Float) -> SLlamaSampler?
}
```

### 📚 SLlamaVocab - Vocabulary Management

```swift
class SLlamaVocab: PLlamaVocab {
    // Initialization
    init(vocab: SLlamaVocabPointer?)
    
    // Properties
    var pointer: SLlamaVocabPointer?          // Raw C vocab
    var tokenCount: Int32                     // Number of tokens
    var type: SLlamaVocabType                 // Vocab type
    var bosToken: SLlamaToken                 // Beginning of sentence
    var eosToken: SLlamaToken                 // End of sentence
    var eotToken: SLlamaToken                 // End of turn
    var nlToken: SLlamaToken                  // Newline token
    
    // Convenience methods
    func tokenize(text: String, addSpecial: Bool, parseSpecial: Bool) throws -> [SLlamaToken]
    func detokenize(tokens: [SLlamaToken], removeSpecial: Bool, unparseSpecial: Bool) throws -> String
    func isEog(_ token: SLlamaToken) -> Bool  // Is end of generation
    func isControl(_ token: SLlamaToken) -> Bool // Is control token
    func getTokenText(_ token: SLlamaToken) -> String?
}
```

### 📊 SLlamaLogits - Output Access

```swift
class SLlamaLogits {
    // Initialization
    init(context: SLlamaContext)
    
    // Methods
    func getLogits() -> SLlamaFloatPointer?                    // Get all logits
    func getLogits(for index: Int32) -> SLlamaFloatPointer?    // Get logits for token
    func getEmbeddings() -> SLlamaFloatPointer?                // Get embeddings
    func getEmbeddings(for index: Int32) -> SLlamaFloatPointer? // Get embeddings for token
    func getEmbeddings(for sequenceId: SLlamaSequenceId) -> SLlamaFloatPointer? // Sequence embeddings
}
```

## Advanced Features

### 🔧 LoRA Adapters

```swift
// Load and apply LoRA adapter
let adapter = try SLlamaAdapter(model: model, path: "/path/to/adapter.gguf")
let success = adapter.apply(to: context, scale: 1.0)

// Remove adapter when done
adapter.remove(from: context)
```

### 🧠 Memory Management

```swift
let memoryManager = SLlamaMemoryManager(context: context)

// Clear memory
memoryManager.clear(data: true)

// Manage sequences
memoryManager.removeSequence(0, from: 0, to: 100)
memoryManager.copySequence(from: 0, to: 1, from: 0, to: 50)
```

### 📈 Performance Monitoring

```swift
let performance = SLlamaPerformance(context: context)

// Benchmark model loading
let loadingResults = performance.benchmarkModelLoading(
    modelPath: "/path/to/model.gguf", 
    iterations: 5
)

// Get performance metrics
let contextMetrics = performance.getContextMetrics()
let samplerMetrics = performance.getSamplerMetrics()
```

### 🔮 Logging with Omen

SLlama integrates with the Omen logging framework for structured, mystical-themed logging:

```swift
import Omen

// Omen categories are automatically registered
// Logs appear with themed emojis and structured output

// System info logging
SLlamaSystemInfo.printSystemInfo()

// Model operations automatically log progress
let model = try SLlamaModel(modelPath: path) // Logs: "🧠 Model loaded successfully"
```

## Protocol-Oriented Design

SLlama uses protocols for dependency injection and testing:

```swift
// Use protocols for testable code
func performInference(with model: PLlamaModel, context: PLlamaContext) {
    let inference = context.inference()
    let batch = SLlamaBatch.getSingleTokenBatch(123)
    try inference.decode(batch)
}

// Easy to mock for testing
class MockModel: PLlamaModel {
    // Implement protocol methods for testing
}
```

## Error Handling

SLlama provides detailed error handling:

```swift
enum SLlamaError: Error {
    // File operations
    case fileNotFound(String)
    case invalidFormat(String)
    case permissionDenied(String)
    
    // Model operations  
    case invalidModel(String)
    case incompatibleModel(String)
    case modelLoadingFailed(String)
    
    // Memory operations
    case outOfMemory
    case insufficientMemory
    case contextFull
    
    // Tokenization
    case invalidVocabulary
    case tokenizationFailed(String)
    case textTooLong
    
    // And many more...
}
```

## System Information

```swift
// Check system capabilities
let systemInfo = SLlamaSystemInfo()
let capabilities = systemInfo.getSystemCapabilities()

// Static utility methods
let supportsMetal = SLlamaSystemInfo.supportsGpuOffload()
let maxDevices = SLlamaSystemInfo.getMaxDevices()
let currentTime = SLlamaSystemInfo.getCurrentTimeMicroseconds()
```

## Best Practices

### Resource Management

```swift
// Always initialize and cleanup
SLlama.initialize()
defer { SLlama.cleanup() }

// Models and contexts are automatically managed
let model = try SLlamaModel(modelPath: path)
let context = try SLlamaContext(model: model)
// Resources freed automatically when objects are deallocated
```

### Threading

```swift
// Set thread counts for optimal performance
context.setThreads(nThreads: 8, nThreadsBatch: 8)

// Or through inference
let inference = context.inference()
inference.setThreads(nThreads: 8, nThreadsBatch: 8)
```

### Batch Processing

```swift
// Efficient batch processing
let batch = SLlamaBatch(nTokens: 512, nSeqMax: 1)

// Add tokens efficiently
for (index, token) in tokens.enumerated() {
    let isLast = index == tokens.count - 1
    batch.addToken(token, position: Int32(index), 
                  sequenceIds: [0], logits: isLast)
}

// Process the batch
try inference.decode(batch)
```

## Architecture

This package includes:

- **Binary Target**: The `llama.xcframework` containing the compiled llama.cpp library for multiple Apple platforms
- **Swift Wrapper**: High-level Swift classes providing a native interface to the C library
- **Protocol Layer**: Protocol abstractions for dependency injection and testing
- **Omen Integration**: Structured logging with mystical theming
- **Platform Support**: Support for iOS, macOS, tvOS, and visionOS with appropriate optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For development setup, see [PRE_COMMIT_SETUP.md](PRE_COMMIT_SETUP.md) for information about our code quality tools and pre-commit hooks. 
