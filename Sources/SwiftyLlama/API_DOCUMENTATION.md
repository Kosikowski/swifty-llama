# SwiftyLlama API Documentation

## Overview

This document provides detailed API documentation for the SwiftyLlama library, covering all public types, methods, and usage patterns.

## Table of Contents

1. [Generation API](#generation-api)
2. [Tuning API](#tuning-api)
3. [Type Definitions](#type-definitions)
4. [Error Handling](#error-handling)
5. [Examples](#examples)

## Generation API

### SwiftyLlama

The main class for text generation operations.

#### Initialization

```swift
public init(modelPath: String, maxCtx: Int32 = 2048) throws
```

**Parameters:**
- `modelPath`: Path to the GGUF model file
- `maxCtx`: Maximum context size (default: 2048)

**Throws:**
- `GenerationError.modelNotLoaded` if model loading fails

**Example:**
```swift
let llama = try SwiftyLlama(modelPath: "path/to/model.gguf")
```

#### Generation

```swift
public func start(
    prompt: String,
    params: GenerationParams = GenerationParams(),
    conversationId: ConversationID? = nil,
    continueConversation: Bool = false
) -> GenerationStream
```

**Parameters:**
- `prompt`: Input text to generate from
- `params`: Generation parameters (optional, uses defaults)
- `conversationId`: Conversation ID for continuity (optional)
- `continueConversation`: Whether to continue existing conversation

**Returns:**
- `GenerationStream`: Stream of generated tokens

**Example:**
```swift
let stream = llama.start(
    prompt: "Hello, how are you?",
    params: GenerationParams(temperature: 0.7, maxTokens: 100),
    conversationId: ConversationID(),
    continueConversation: false
)
```

#### Cancellation

```swift
public func cancelGeneration(id: GenerationID)
public func cancelAllGenerations()
```

**Parameters:**
- `id`: Generation ID to cancel

**Example:**
```swift
let stream = llama.start(prompt: "Generate...")
stream.cancel() // Cancel specific generation
llama.cancelAllGenerations() // Cancel all generations
```

#### Conversation Management

```swift
public func getConversationState() -> [Conversation]
public func saveConversationsToJSON(path: String) throws
public func loadConversationsFromJSON(path: String) throws -> [Conversation]
public func restoreConversations(_ savedConversations: [Conversation]) throws
```

**Example:**
```swift
// Save conversations
let conversations = llama.getConversationState()
try llama.saveConversationsToJSON(path: "conversations.json")

// Load conversations
let loadedConversations = try llama.loadConversationsFromJSON(path: "conversations.json")
try llama.restoreConversations(loadedConversations)
```

### GenerationParams

Configuration for text generation.

#### Properties

```swift
public struct GenerationParams: Hashable, Sendable {
    public let seed: UInt32              // Random seed (default: 42)
    public let topK: Int32               // Top-k sampling (default: 40)
    public let topP: Float               // Top-p sampling (default: 0.9)
    public let temperature: Float         // Sampling temperature (default: 0.7)
    public let repeatPenalty: Float      // Repetition penalty (default: 1.1)
    public let repetitionLookback: Int32 // Repetition lookback (default: 64)
    public let maxTokens: Int32          // Maximum tokens (default: 100)
    public let threads: Int32            // CPU threads (default: 4)
    public let batchThreads: Int32       // Batch threads (default: 4)
    public let enableEmbeddings: Bool    // Enable embeddings (default: false)
    public let enableCausalAttention: Bool // Enable causal attention (default: true)
}
```

#### Initialization

```swift
public init(
    seed: UInt32 = 42,
    topK: Int32 = 40,
    topP: Float = 0.9,
    temperature: Float = 0.7,
    repeatPenalty: Float = 1.1,
    repetitionLookback: Int32 = 64,
    maxTokens: Int32 = 100,
    threads: Int32 = 4,
    batchThreads: Int32 = 4,
    enableEmbeddings: Bool = false,
    enableCausalAttention: Bool = true
)
```

**Example:**
```swift
let params = GenerationParams(
    temperature: 0.8,
    maxTokens: 200,
    topK: 50,
    topP: 0.95
)
```

### GenerationStream

Stream of generated tokens.

#### Properties

```swift
public struct GenerationStream {
    public let id: GenerationID
    public let stream: AsyncThrowingStream<String, Error>
}
```

#### Methods

```swift
public func cancel()
```

**Example:**
```swift
let stream = llama.start(prompt: "Generate...")

// Consume tokens
for try await token in stream.stream {
    print(token, terminator: "")
}

// Cancel if needed
stream.cancel()
```

## Tuning API

### SwiftyLlamaTuning

Protocol for fine-tuning operations.

#### Model Loading

```swift
func loadModel(path: String) throws
```

**Parameters:**
- `path`: Path to the model file

**Throws:**
- `TuningError.modelNotLoaded` if model loading fails

**Example:**
```swift
try tuner.loadModel(path: "path/to/model.gguf")
```

#### LoRA Management

```swift
func applyLoRA(path: String, scale: Float, metadata: LoRAMetadata?) throws
func removeLoRA() throws
func getCurrentLoRA() -> LoRAAdapter?
func getAvailableAdapters() -> [LoRAAdapter]
```

**Parameters:**
- `path`: Path to LoRA adapter file
- `scale`: Scaling factor for adapter
- `metadata`: Optional metadata for adapter

**Example:**
```swift
// Apply LoRA
try tuner.applyLoRA(
    path: "path/to/adapter.gguf",
    scale: 1.0,
    metadata: LoRAMetadata()
)

// Remove LoRA
try tuner.removeLoRA()

// Get current LoRA
let currentLoRA = tuner.getCurrentLoRA()
```

#### Training Data Preparation

```swift
func prepareTrainingData(
    conversations: [TrainingConversation],
    validationSplit: Double
) throws -> TrainingDataset
```

**Parameters:**
- `conversations`: Array of training conversations
- `validationSplit`: Fraction for validation (0.0-1.0)

**Returns:**
- `TrainingDataset`: Prepared training and validation data

**Example:**
```swift
let dataset = try tuner.prepareTrainingData(
    conversations: conversations,
    validationSplit: 0.2
)
```

#### Training Session Management

```swift
func startTrainingSession(
    dataset: TrainingDataset,
    config: TrainingConfig
) throws -> TrainingSession

func stopTrainingSession()
func getCurrentTrainingSession() -> TrainingSession?
func getTrainingMetrics() -> TrainingMetrics?
```

**Example:**
```swift
// Start training
let session = try tuner.startTrainingSession(
    dataset: dataset,
    config: config
)

// Monitor training
let currentSession = tuner.getCurrentTrainingSession()
let metrics = tuner.getTrainingMetrics()

// Stop training
tuner.stopTrainingSession()
```

#### Evaluation

```swift
func evaluateModel(validationExamples: [TrainingExample]) throws -> EvaluationMetrics
```

**Parameters:**
- `validationExamples`: Array of validation examples

**Returns:**
- `EvaluationMetrics`: Evaluation results

**Example:**
```swift
let metrics = try tuner.evaluateModel(
    validationExamples: dataset.validation
)
print("Perplexity: \(metrics.perplexity)")
```

#### Safety Features

```swift
func validateLoRACompatibility(path: String) throws -> LoRACompatibility
func setLoRAFallbackMode(_ enabled: Bool)
```

**Example:**
```swift
// Check compatibility
let compatibility = try tuner.validateLoRACompatibility(
    path: "path/to/adapter.gguf"
)

// Enable fallback mode
tuner.setLoRAFallbackMode(true)
```

### TrainingConfig

Configuration for training sessions.

#### Properties

```swift
public struct TrainingConfig: Codable, Sendable {
    public let loraRank: Int           // LoRA rank (default: 8)
    public let learningRate: Float     // Learning rate (default: 2e-5)
    public let epochs: Int             // Number of epochs (default: 3)
    public let batchSize: Int          // Batch size (default: 1)
    public let useQLoRA: Bool          // Use QLoRA (default: false)
    public let qLoRAConfig: QLoRAConfig? // QLoRA configuration (default: nil)
}
```

#### Initialization

```swift
public init(
    loraRank: Int = 8,
    learningRate: Float = 2e-5,
    epochs: Int = 3,
    batchSize: Int = 1,
    useQLoRA: Bool = false,
    qLoRAConfig: QLoRAConfig? = nil
)
```

**Example:**
```swift
let config = TrainingConfig(
    loraRank: 16,
    learningRate: 1e-5,
    epochs: 5,
    batchSize: 2,
    useQLoRA: true,
    qLoRAConfig: QLoRAConfig()
)
```

## Type Definitions

### Generation Types

#### ConversationID
```swift
public struct ConversationID: Hashable, Codable, Sendable {
    public let id: UUID
    public init() { self.id = UUID() }
}
```

#### GenerationID
```swift
public struct GenerationID: Hashable, Sendable {
    public let id: UUID
    public init() { self.id = UUID() }
}
```

#### Conversation
```swift
public struct Conversation: Codable {
    public let id: ConversationID
    public var messages: [ConversationMessage]
    public var totalTokens: Int32
    public let createdAt: Date
}
```

#### ConversationMessage
```swift
public struct ConversationMessage: Codable {
    public let role: String
    public let content: String
    public let tokens: [SLlamaToken]
    public let timestamp: Date
}
```

### Tuning Types

#### TrainingConversation
```swift
public struct TrainingConversation: Codable, Sendable {
    public let id: String
    public let messages: [TrainingMessage]
}
```

#### TrainingMessage
```swift
public struct TrainingMessage: Codable, Sendable {
    public let role: MessageRole
    public let content: String
}
```

#### MessageRole
```swift
public enum MessageRole: String, Codable, Sendable {
    case system
    case user
    case assistant
}
```

#### TrainingExample
```swift
public struct TrainingExample: Codable, Sendable {
    public let tokens: [SLlamaToken]
    public let targetTokens: [SLlamaToken]
}
```

#### TrainingDataset
```swift
public struct TrainingDataset: Codable, Sendable {
    public let training: [TrainingExample]
    public let validation: [TrainingExample]
}
```

#### LoRAAdapter
```swift
public struct LoRAAdapter: Codable, Sendable {
    public let path: String
    public let scale: Float
    public let metadata: LoRAMetadata
    public let appliedAt: Date
}
```

#### LoRAMetadata
```swift
public struct LoRAMetadata: Codable, Sendable {
    public let rank: Int
    public let alpha: Float
    public let targetModules: [String]
}
```

#### QLoRAConfig
```swift
public struct QLoRAConfig: Codable, Sendable {
    public let quantType: String
    public let useDoubleQuant: Bool
    public let computeDtype: String
}
```

#### EvaluationMetrics
```swift
public struct EvaluationMetrics: Codable, Sendable {
    public let perplexity: Float
    public let averageLoss: Float
    public let totalExamples: Int
    public let totalTokens: Int
}
```

#### TrainingSession
```swift
public struct TrainingSession: Codable, Sendable {
    public let id: String
    public let status: TrainingStatus
    public let startTime: Date
    public let endTime: Date?
}
```

#### TrainingStatus
```swift
public enum TrainingStatus: String, Codable, Sendable {
    case running
    case stopped
    case completed
    case failed
}
```

#### TrainingMetrics
```swift
public struct TrainingMetrics: Codable, Sendable {
    public let epoch: Int
    public let loss: Float
    public let learningRate: Float
}
```

#### LoRACompatibility
```swift
public struct LoRACompatibility: Codable, Sendable {
    public let isCompatible: Bool
    public let reason: String?
}
```

## Error Handling

### GenerationError

```swift
public enum GenerationError: Error, LocalizedError {
    case abortedByUser
    case modelNotLoaded
    case contextNotInitialized
    case conversationNotFound
    case contextPreparationFailed
    case tokenizationFailed
    case generationFailed
    case invalidState
}
```

### TuningError

```swift
public enum TuningError: Error, LocalizedError, Equatable {
    case contextNotInitialized
    case modelNotLoaded
    case tokenizerNotInitialized
    case adapterFileNotFound(path: String)
    case adapterApplicationFailed(path: String, errorDescription: String)
    case invalidLoRARank(rank: Int)
    case invalidLearningRate(rate: Float)
    case invalidEpochs(epochs: Int)
    case trainingSessionNotFound
    case incompatibleAdapter
}
```

## Examples

### Complete Generation Example

```swift
import SwiftyLlama

// Initialize
let llama = try SwiftyLlama(modelPath: "model.gguf")

// Start generation with conversation
let conversationId = ConversationID()
let stream = llama.start(
    prompt: "Tell me a story about a robot",
    params: GenerationParams(
        temperature: 0.8,
        maxTokens: 200,
        topK: 50
    ),
    conversationId: conversationId,
    continueConversation: false
)

// Consume tokens
var fullResponse = ""
for try await token in stream.stream {
    fullResponse += token
    print(token, terminator: "")
}

// Continue conversation
let continuationStream = llama.start(
    prompt: "What happened next?",
    conversationId: conversationId,
    continueConversation: true
)

// Save conversation
try llama.saveConversationsToJSON(path: "conversation.json")
```

### Complete Fine-tuning Example

```swift
import SwiftyLlama

// Initialize tuner
let tuner: SwiftyLlamaTuning = SwiftyLlamaTuner()

// Load model
try tuner.loadModel(path: "base-model.gguf")

// Prepare training data
let conversations = [
    TrainingConversation(
        id: "conv1",
        messages: [
            TrainingMessage(role: .system, content: "You are a helpful assistant."),
            TrainingMessage(role: .user, content: "What is AI?"),
            TrainingMessage(role: .assistant, content: "AI is artificial intelligence...")
        ]
    )
]

let dataset = try tuner.prepareTrainingData(
    conversations: conversations,
    validationSplit: 0.2
)

// Configure training
let config = TrainingConfig(
    loraRank: 8,
    learningRate: 2e-5,
    epochs: 3,
    batchSize: 1
)

// Start training
let session = try tuner.startTrainingSession(
    dataset: dataset,
    config: config
)

// Monitor training
let currentSession = tuner.getCurrentTrainingSession()
let metrics = tuner.getTrainingMetrics()

// Stop training
tuner.stopTrainingSession()

// Evaluate model
let evaluationMetrics = try tuner.evaluateModel(
    validationExamples: dataset.validation
)

print("Final perplexity: \(evaluationMetrics.perplexity)")
```

---

This API documentation provides detailed coverage of all public interfaces in the SwiftyLlama library. For additional examples and usage patterns, refer to the test suite in `Tests/SwiftyLlamaTests/`. 