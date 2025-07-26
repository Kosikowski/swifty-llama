# SwiftyLlama

A detailed Swift library for LLM generation and fine-tuning using llama.cpp, providing a clean, type-safe interface for both text generation and model fine-tuning operations.

## 🏗️ Architecture

SwiftyLlama is built with a modular architecture that separates concerns and provides clean interfaces:

```
Sources/SwiftyLlama/
├── Generation/           # Text generation functionality
│   ├── SwiftyLlama.swift (659 lines)
│   └── GenerationTypes/  # Type definitions for generation
├── Tuning/              # Fine-tuning functionality  
│   ├── SwiftyLlamaTuner.swift (298 lines)
│   ├── SwiftyTuningLlamaProtocol.swift (121 lines)
│   └── TuningTypes/     # Type definitions for tuning
└── Common/              # Shared types
    └── SLlamaTypes.swift
```

### Core Design Principles

- **Actor Isolation**: All operations use `@SwiftyLlamaActor` for thread safety
- **Protocol-First**: Clean interfaces with `SwiftyLlamaTuning` protocol
- **Type Safety**: Detailed error handling with typed errors
- **Streaming**: Real-time token generation with `AsyncThrowingStream`
- **Persistence**: Conversation state management with JSON serialization

## 🚀 Features

### Text Generation (`SwiftyLlama`)

#### Core Capabilities
- **Unified Actor Design**: Single `@SwiftyLlamaActor` for all operations
- **Conversation Management**: Persistent conversations with context continuity
- **Stream-based Generation**: Real-time token streaming with cancellation support
- **Context Management**: Single context per instance with proper lifecycle
- **Error Handling**: Detailed error types with descriptive messages

#### Key Features
- ✅ **Conversation Persistence**: Save/restore conversation state
- ✅ **Context Continuity**: Continue conversations across app launches
- ✅ **Cancellation Support**: Full cancellation with proper cleanup
- ✅ **Parameter Configuration**: Rich generation parameters
- ✅ **Memory Management**: Efficient memory usage with Metal GPU support

### Fine-tuning (`SwiftyLlamaTuner`)

#### Core Capabilities
- **LoRA Adapter Management**: Apply, remove, and manage LoRA adapters
- **Training Data Preparation**: Chat formatting with validation splits
- **Training Session Management**: Start, stop, and monitor training
- **Evaluation**: Perplexity, loss, and token count metrics
- **QLoRA Support**: Quantized LoRA configuration

#### Key Features
- ✅ **LoRA Adapter Lifecycle**: Complete adapter management
- ✅ **Training Pipeline**: End-to-end fine-tuning workflow
- ✅ **Evaluation Metrics**: Real-time performance monitoring
- ✅ **Safety Features**: Compatibility validation and fallback modes
- ✅ **Protocol Interface**: Clean `SwiftyLlamaTuning` protocol

## 📚 Usage Examples

### Text Generation

#### Basic Generation

```swift
import SwiftyLlama

// Initialize with model
let llama = try SwiftyLlama(modelPath: "path/to/model.gguf")

// Start generation
let stream = llama.start(
    prompt: "Hello, how are you?",
    params: GenerationParams(
        temperature: 0.7,
        maxTokens: 100
    )
)

// Consume tokens
for try await token in stream.stream {
    print(token, terminator: "")
}
```

#### Conversation Management

```swift
// Start new conversation
let conversationId = ConversationID()
let stream = llama.start(
    prompt: "Tell me a story",
    conversationId: conversationId,
    continueConversation: false
)

// Continue existing conversation
let continuationStream = llama.start(
    prompt: "What happened next?",
    conversationId: conversationId,
    continueConversation: true
)

// Save conversation state
let savedState = llama.getConversationState()
try llama.saveConversationsToJSON(path: "conversations.json")

// Restore conversation state
let restoredConversations = try llama.loadConversationsFromJSON(path: "conversations.json")
try llama.restoreConversations(restoredConversations)
```

#### Advanced Conversation Management

```swift
// Get all current conversations
let allConversations = llama.getConversationState()

// Save specific conversation
let conversation = allConversations.first { $0.id == conversationId }
if let conversation = conversation {
    let conversationsToSave = [conversation]
    // Save to custom path
    try llama.saveConversationsToJSON(path: "my_conversation.json")
}

// Load and restore multiple conversations
let loadedConversations = try llama.loadConversationsFromJSON(path: "all_conversations.json")
try llama.restoreConversations(loadedConversations)

// Continue with specific conversation
let specificConversation = loadedConversations.first { $0.id == targetId }
if let conversation = specificConversation {
    let stream = llama.start(
        prompt: "Continue our discussion",
        conversationId: conversation.id,
        continueConversation: true
    )
}

// Clear conversation history
let emptyConversations: [Conversation] = []
try llama.restoreConversations(emptyConversations)
```

#### Conversation Persistence Best Practices

```swift
// Auto-save conversations periodically
Task {
    while true {
        try await Task.sleep(nanoseconds: 30_000_000_000) // 30 seconds
        try llama.saveConversationsToJSON(path: "auto_save.json")
    }
}

// Save on app termination
NotificationCenter.default.addObserver(
    forName: UIApplication.willTerminateNotification,
    object: nil,
    queue: .main
) { _ in
    try? llama.saveConversationsToJSON(path: "final_save.json")
}

// Load conversations on app start
func loadSavedConversations() {
    do {
        let conversations = try llama.loadConversationsFromJSON(path: "saved_conversations.json")
        try llama.restoreConversations(conversations)
    } catch {
        print("No saved conversations found or error loading: \(error)")
    }
}
```

#### Cancellation Support

```swift
let stream = llama.start(prompt: "Generate a long story...")

// Cancel after 5 seconds
Task {
    try await Task.sleep(nanoseconds: 5_000_000_000)
    stream.cancel()
}

// Handle cancellation
for try await token in stream.stream {
    print(token, terminator: "")
}
// Stream automatically handles cleanup
```

### Fine-tuning

#### Basic Fine-tuning Setup

```swift
import SwiftyLlama

// Initialize tuner
let tuner: SwiftyLlamaTuning = SwiftyLlamaTuner()

// Load base model
try tuner.loadModel(path: "path/to/base-model.gguf")

// Apply LoRA adapter
try tuner.applyLoRA(
    path: "path/to/adapter.gguf",
    scale: 1.0
)
```

#### Training Data Preparation

```swift
// Create training conversations
let conversations = [
    TrainingConversation(
        id: "conv1",
        messages: [
            TrainingMessage(role: .system, content: "You are a helpful assistant."),
            TrainingMessage(role: .user, content: "What is 2+2?"),
            TrainingMessage(role: .assistant, content: "2+2 equals 4.")
        ]
    )
]

// Prepare training data
let dataset = try tuner.prepareTrainingData(
    conversations: conversations,
    validationSplit: 0.2
)
```

#### Training Session Management

```swift
// Configure training
let config = TrainingConfig(
    loraRank: 8,
    learningRate: 2e-5,
    epochs: 3,
    batchSize: 1
)

// Start training session
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
// Evaluate model performance
let metrics = try tuner.evaluateModel(
    validationExamples: dataset.validation
)

print("Perplexity: \(metrics.perplexity)")
print("Average Loss: \(metrics.averageLoss)")
print("Total Tokens: \(metrics.totalTokens)")
```

#### QLoRA Configuration

```swift
// Configure QLoRA
let qLoRAConfig = QLoRAConfig(
    quantType: "nf4",
    useDoubleQuant: true,
    computeDtype: "float16"
)

let config = TrainingConfig(
    loraRank: 8,
    learningRate: 2e-5,
    epochs: 3,
    useQLoRA: true,
    qLoRAConfig: qLoRAConfig
)
```

## 🔧 API Reference

### Generation Types

#### `SwiftyLlama`
Main class for text generation operations.

**Key Methods:**
- `init(modelPath:maxCtx:)` - Initialize with model
- `start(prompt:params:conversationId:continueConversation:)` - Start generation
- `cancelGeneration(id:)` - Cancel specific generation
- `cancelAllGenerations()` - Cancel all active generations
- `getConversationState()` - Get all conversations
- `saveConversationsToJSON(path:)` - Save conversations to file
- `loadConversationsFromJSON(path:)` - Load conversations from file

#### `GenerationParams`
Configuration for text generation.

**Properties:**
- `seed: UInt32` - Random seed
- `temperature: Float` - Sampling temperature
- `topK: Int32` - Top-k sampling
- `topP: Float` - Top-p sampling
- `maxTokens: Int32` - Maximum tokens to generate
- `threads: Int32` - Number of threads
- `batchThreads: Int32` - Batch processing threads

#### `GenerationError`
Error types for generation operations.

**Cases:**
- `abortedByUser` - User cancelled generation
- `modelNotLoaded` - Model not loaded
- `contextNotInitialized` - Context not initialized
- `conversationNotFound` - Conversation not found
- `contextPreparationFailed` - Context preparation failed
- `tokenizationFailed` - Tokenization failed
- `generationFailed` - Generation failed
- `invalidState` - Invalid state

### Tuning Types

#### `SwiftyLlamaTuning`
Protocol for fine-tuning operations.

**Key Methods:**
- `loadModel(path:)` - Load base model
- `applyLoRA(path:scale:metadata:)` - Apply LoRA adapter
- `removeLoRA()` - Remove current LoRA
- `prepareTrainingData(conversations:validationSplit:)` - Prepare training data
- `startTrainingSession(dataset:config:)` - Start training
- `evaluateModel(validationExamples:)` - Evaluate model
- `validateLoRACompatibility(path:)` - Check LoRA compatibility

#### `TrainingConfig`
Configuration for training sessions.

**Properties:**
- `loraRank: Int` - LoRA rank
- `learningRate: Float` - Learning rate
- `epochs: Int` - Number of epochs
- `batchSize: Int` - Batch size
- `useQLoRA: Bool` - Use QLoRA
- `qLoRAConfig: QLoRAConfig?` - QLoRA configuration

#### `TuningError`
Error types for fine-tuning operations.

**Cases:**
- `contextNotInitialized` - Context not initialized
- `modelNotLoaded` - Model not loaded
- `tokenizerNotInitialized` - Tokenizer not initialized
- `adapterFileNotFound(path:)` - Adapter file not found
- `invalidLoRARank(rank:)` - Invalid LoRA rank
- `invalidLearningRate(rate:)` - Invalid learning rate
- `invalidEpochs(epochs:)` - Invalid number of epochs
- `trainingSessionNotFound` - No active training session
- `incompatibleAdapter` - Incompatible adapter

## 🧪 Testing

The implementation includes detailed test coverage with **45 total tests**:

### Generation Tests (22 tests)
- Initialization and model loading
- Conversation management and persistence
- Cancellation scenarios (6 different cases)
- Error handling and edge cases
- Context continuity and warm-up

### Fine-tuning Tests (25 tests)
- Initialization and model loading
- LoRA adapter management
- Training data preparation and session management
- Evaluation and metrics calculation
- Error handling and validation
- QLoRA configuration and edge cases

### Protocol Tests (3 tests)
- Protocol conformance and type safety
- Default parameter extensions

## 🔒 Error Handling

### Generation Errors
All generation operations throw `GenerationError` with descriptive messages:

```swift
do {
    let stream = llama.start(prompt: "Hello")
    for try await token in stream.stream {
        print(token)
    }
} catch GenerationError.abortedByUser {
    print("Generation was cancelled by user")
} catch GenerationError.contextNotInitialized {
    print("Context not initialized")
} catch {
    print("Other error: \(error)")
}
```

### Tuning Errors
Fine-tuning operations throw `TuningError` with specific error cases:

```swift
do {
    try tuner.applyLoRA(path: "nonexistent.gguf")
} catch TuningError.adapterFileNotFound(let path) {
    print("Adapter file not found: \(path)")
} catch TuningError.invalidLoRARank(let rank) {
    print("Invalid LoRA rank: \(rank)")
} catch {
    print("Other tuning error: \(error)")
}
```

## 🚀 Performance

### Optimizations
- **Single Context per Instance**: Efficient memory usage
- **Metal GPU Acceleration**: Native Apple Silicon support
- **Streaming Architecture**: Real-time token generation
- **Actor Isolation**: Thread-safe operations
- **Memory Management**: Proper cleanup and resource management

### Memory Usage
- Context size: Configurable (default 2048 tokens)
- Batch processing: Efficient token batching
- GPU memory: Automatic Metal memory management
- Conversation persistence: JSON-based serialization

## 🔧 Configuration

### Generation Parameters
```swift
let params = GenerationParams(
    seed: 42,                    // Random seed
    topK: 40,                    // Top-k sampling
    topP: 0.9,                   // Top-p sampling
    temperature: 0.7,             // Sampling temperature
    repeatPenalty: 1.1,          // Repetition penalty
    repetitionLookback: 64,      // Repetition lookback
    maxTokens: 100,              // Maximum tokens
    threads: 4,                  // CPU threads
    batchThreads: 4,             // Batch threads
    enableEmbeddings: false,     // Enable embeddings
    enableCausalAttention: true  // Enable causal attention
)
```

### Training Configuration
```swift
let config = TrainingConfig(
    loraRank: 8,                 // LoRA rank
    learningRate: 2e-5,          // Learning rate
    epochs: 3,                   // Number of epochs
    batchSize: 1,                // Batch size
    useQLoRA: false,             // Use QLoRA
    qLoRAConfig: nil             // QLoRA configuration
)
```

## 📦 Dependencies

- **SLlama**: Core llama.cpp Swift bindings
- **Foundation**: Basic Swift functionality
- **Testing**: Swift testing framework

## 🤝 Contributing

### Code Style
- Follow Swift API Design Guidelines
- Use detailed error handling
- Include unit tests for new features
- Document public APIs

### Testing
- Run tests: `swift test`
- Test specific target: `swift test --target SwiftyLlamaTests`
- All tests must pass before merging

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the test examples for usage patterns
2. Review error handling documentation
3. Examine the detailed test suite
4. Check actor isolation requirements

---

**SwiftyLlama** provides a production-ready, type-safe interface for LLM operations in Swift, with detailed support for both generation and fine-tuning workflows. 