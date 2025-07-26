# SwiftyLlama

A complete Swift library for Large Language Model (LLM) operations, providing both high-level generation APIs and low-level llama.cpp bindings for Apple platforms.

## 🏗️ Architecture

This package contains two main components:

- **[SwiftyLlama](./Sources/SwiftyLlama/README.md)** - High-level Swift API for text generation and fine-tuning
- **[SLlama](./Sources/SLlama/README.md)** - Low-level Swift wrapper for llama.cpp

## 🚀 SwiftyLlama Capabilities

SwiftyLlama provides a complete, production-ready solution for LLM operations with the following key capabilities:

### 📝 Text Generation
- **Streaming Generation**: Real-time token streaming with `AsyncThrowingStream`
- **Actor-based Design**: Thread-safe operations with `@SwiftyLlamaActor`
- **Rich Parameters**: Temperature, top-k, top-p, repetition penalty, and more
- **Cancellation Support**: Full cancellation with proper cleanup
- **Memory Management**: Efficient Metal GPU support with automatic resource management

### 💬 Conversation Management
- **Persistent Conversations**: Save/restore conversation state across app launches
- **Context Continuity**: Continue conversations with full context preservation
- **JSON Serialization**: Export/import conversations for backup and sharing
- **Multiple Conversations**: Manage multiple concurrent conversation threads
- **Auto-save**: Periodic conversation persistence with error handling

### 🎯 Fine-tuning & LoRA
- **LoRA Adapter Management**: Apply, remove, and manage LoRA adapters
- **Training Pipeline**: End-to-end fine-tuning workflow with validation
- **QLoRA Support**: Quantized LoRA configuration for memory efficiency
- **Training Data Preparation**: Chat formatting with validation splits
- **Evaluation Metrics**: Perplexity, loss, and token count monitoring
- **Session Management**: Start, stop, and monitor training sessions

### 🔧 Advanced Features
- **Protocol Design**: Clean interfaces for dependency injection and testing
- **Error Handling**: Complete error types with descriptive messages
- **Performance Optimization**: Conditional inlining for critical paths
- **Multi-platform Support**: iOS, macOS, tvOS, and visionOS
- **Metal Integration**: Native Apple Silicon GPU acceleration

## 🚀 Quick Start with SwiftyLlama

### Installation

Add SwiftyLlama to your project in Xcode:

1. File → Add Package Dependencies
2. Enter the repository URL
3. Select the package and add it to your target

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/your-username/swifty-llama-cpp.git", from: "1.0.0")
]
```

### Basic Text Generation

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

### Conversation Management

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
```

### Fine-tuning with LoRA

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

// Prepare training data
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

let dataset = try tuner.prepareTrainingData(
    conversations: conversations,
    validationSplit: 0.2
)

// Start training
let config = TrainingConfig(
    loraRank: 8,
    learningRate: 2e-5,
    epochs: 3,
    batchSize: 1
)

let session = try tuner.startTrainingSession(
    dataset: dataset,
    config: config
)
```

## 📚 Documentation

### SwiftyLlama (High-level API)
- **[Complete Documentation](./Sources/SwiftyLlama/README.md)** - Full API reference and examples
- **[Quick Reference](./Sources/SwiftyLlama/QUICK_REFERENCE.md)** - Fast lookup guide
- **[API Documentation](./Sources/SwiftyLlama/API_DOCUMENTATION.md)** - Detailed API docs

### SLlama (Low-level API)
- **[Complete Documentation](./Sources/SLlama/README.md)** - Full low-level API reference
- Direct llama.cpp bindings for advanced use cases

## 🎯 Key Features

### SwiftyLlama (High-level)
- ✅ **Actor-based Design** - Thread-safe operations with `@SwiftyLlamaActor`
- ✅ **Conversation Persistence** - Save/restore conversation state
- ✅ **Streaming Generation** - Real-time token streaming with cancellation
- ✅ **Fine-tuning Support** - LoRA adapter management and training
- ✅ **Error Handling** - Complete error types and handling
- ✅ **Memory Management** - Efficient Metal GPU support

### SLlama (Low-level)
- ✅ **Direct llama.cpp Access** - Native Swift interface to C library
- ✅ **Advanced Sampling** - Mirostat, XTC, Min-P, Typical, and more strategies
- ✅ **Performance Monitoring** - Benchmarking, metrics, and profiling tools
- ✅ **Protocol Design** - Dependency injection and testing support
- ✅ **Multi-platform** - iOS, macOS, tvOS, and visionOS support
- ✅ **Memory Management** - Efficient KV cache and sequence management
- ✅ **LoRA Adapters** - Low-level adapter application and management
- ✅ **Conditional Inlining** - Performance optimization with `@inlinable` methods
- ✅ **Tokenization** - Advanced text processing and vocabulary management
- ✅ **System Information** - Hardware capability detection and optimization

## 🔧 Requirements

- iOS 13.0+
- macOS 11.0+
- tvOS 13.0+
- visionOS 1.0+
- Swift 5.9+

## 🏗️ Package Structure

```
Sources/
├── SwiftyLlama/          # High-level generation and fine-tuning API
│   ├── Generation/       # Text generation functionality
│   ├── Tuning/          # Fine-tuning functionality
│   └── Common/          # Shared types
├── SLlama/              # Low-level llama.cpp Swift wrapper
│   ├── Core classes     # Model, Context, Core operations
│   ├── Tokenization     # Text processing and vocabulary
│   ├── Sampling         # Advanced sampling strategies
│   └── Performance      # Monitoring and benchmarking
└── llama.cpp/           # Binary framework for Apple platforms
```

## 🚀 Performance

- **Metal GPU Acceleration** - Native Apple Silicon support
- **Efficient Memory Management** - Proper cleanup and resource management
- **Streaming Architecture** - Real-time token generation
- **Actor Isolation** - Thread-safe operations
- **Optimized Batching** - Efficient token processing

## 🧪 Testing

Complete test coverage with **70+ total tests**:

- **SwiftyLlama Tests** - Generation, fine-tuning, and protocol tests
- **SLlama Tests** - Low-level API, performance, and integration tests

## 🔒 Error Handling

Both packages provide detailed error handling:

```swift
// SwiftyLlama errors
do {
    let stream = llama.start(prompt: "Hello")
    for try await token in stream.stream {
        print(token)
    }
} catch GenerationError.abortedByUser {
    print("Generation was cancelled")
} catch {
    print("Other error: \(error)")
}

// SLlama errors
do {
    let model = try SLlamaModel(modelPath: path)
} catch SLlamaError.fileNotFound(let path) {
    print("Model not found: \(path)")
} catch {
    print("Other error: \(error)")
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For development setup, see [PRE_COMMIT_SETUP.md](PRE_COMMIT_SETUP.md) for information about our code quality tools and pre-commit hooks.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the detailed documentation for each package
2. Review the test examples for usage patterns
3. Examine the complete test suites
4. Check actor isolation requirements for SwiftyLlama

---

**SwiftyLlama** provides a production-ready, type-safe interface for LLM operations in Swift, with both high-level APIs for common use cases and low-level access for advanced scenarios. 
