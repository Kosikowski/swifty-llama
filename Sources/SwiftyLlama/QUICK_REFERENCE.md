# SwiftyLlama Quick Reference

## 🚀 Quick Start

### Basic Generation
```swift
import SwiftyLlama

// Initialize
let llama = try SwiftyLlama(modelPath: "model.gguf")

// Generate text
let stream = llama.start(prompt: "Hello, how are you?")
for try await token in stream.stream {
    print(token, terminator: "")
}
```

### Basic Fine-tuning
```swift
import SwiftyLlama

// Initialize tuner
let tuner: SwiftyLlamaTuning = SwiftyLlamaTuner()

// Load model and apply LoRA
try tuner.loadModel(path: "base-model.gguf")
try tuner.applyLoRA(path: "adapter.gguf")

// Prepare and train
let dataset = try tuner.prepareTrainingData(conversations: conversations)
let session = try tuner.startTrainingSession(dataset: dataset, config: config)
```

## 📝 Common Patterns

### Generation Patterns

#### 1. Simple Generation
```swift
let stream = llama.start(prompt: "Write a poem")
for try await token in stream.stream {
    print(token, terminator: "")
}
```

#### 2. Generation with Parameters
```swift
let params = GenerationParams(
    temperature: 0.8,
    maxTokens: 200,
    topK: 50,
    topP: 0.95
)
let stream = llama.start(prompt: "Tell a story", params: params)
```

#### 3. Conversation Management
```swift
let conversationId = ConversationID()

// Start conversation
let stream1 = llama.start(
    prompt: "Tell me about AI",
    conversationId: conversationId,
    continueConversation: false
)

// Continue conversation
let stream2 = llama.start(
    prompt: "What are the risks?",
    conversationId: conversationId,
    continueConversation: true
)
```

#### 4. Cancellation
```swift
let stream = llama.start(prompt: "Generate...")

// Cancel after 5 seconds
Task {
    try await Task.sleep(nanoseconds: 5_000_000_000)
    stream.cancel()
}

// Handle cancellation
for try await token in stream.stream {
    print(token, terminator: "")
}
```

#### 5. Error Handling
```swift
do {
    let stream = llama.start(prompt: "Hello")
    for try await token in stream.stream {
        print(token)
    }
} catch GenerationError.abortedByUser {
    print("Cancelled by user")
} catch GenerationError.contextNotInitialized {
    print("Context not ready")
} catch {
    print("Error: \(error)")
}
```

#### 6. Conversation Persistence
```swift
// Save conversations
try llama.saveConversationsToJSON(path: "conversations.json")

// Load conversations
let conversations = try llama.loadConversationsFromJSON(path: "conversations.json")
try llama.restoreConversations(conversations)
```

### Fine-tuning Patterns

#### 1. LoRA Management
```swift
// Apply LoRA
try tuner.applyLoRA(path: "adapter.gguf", scale: 1.0)

// Remove LoRA
try tuner.removeLoRA()

// Check current LoRA
let currentLoRA = tuner.getCurrentLoRA()
```

#### 2. Training Data Preparation
```swift
let conversations = [
    TrainingConversation(
        id: "conv1",
        messages: [
            TrainingMessage(role: .system, content: "You are helpful."),
            TrainingMessage(role: .user, content: "What is 2+2?"),
            TrainingMessage(role: .assistant, content: "2+2 equals 4.")
        ]
    )
]

let dataset = try tuner.prepareTrainingData(
    conversations: conversations,
    validationSplit: 0.2
)
```

#### 3. Training Session
```swift
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

// Monitor
let currentSession = tuner.getCurrentTrainingSession()
let metrics = tuner.getTrainingMetrics()

// Stop
tuner.stopTrainingSession()
```

#### 4. Evaluation
```swift
let metrics = try tuner.evaluateModel(
    validationExamples: dataset.validation
)

print("Perplexity: \(metrics.perplexity)")
print("Loss: \(metrics.averageLoss)")
print("Tokens: \(metrics.totalTokens)")
```

#### 5. QLoRA Configuration
```swift
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

#### 6. Safety Features
```swift
// Check compatibility
let compatibility = try tuner.validateLoRACompatibility(
    path: "adapter.gguf"
)

// Enable fallback
tuner.setLoRAFallbackMode(true)
```

## ⚙️ Configuration

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

## 🎯 Best Practices

### Generation
1. **Always handle cancellation**: Use `stream.cancel()` for cleanup
2. **Use conversation IDs**: For multi-turn conversations
3. **Save state regularly**: Use `saveConversationsToJSON()`
4. **Handle errors gracefully**: Catch specific `GenerationError` cases
5. **Monitor memory**: Single context per instance

### Fine-tuning
1. **Validate LoRA compatibility**: Use `validateLoRACompatibility()`
2. **Use appropriate validation splits**: 0.1-0.2 for validation
3. **Monitor training metrics**: Check `getTrainingMetrics()`
4. **Enable fallback mode**: For safety with `setLoRAFallbackMode(true)`
5. **Evaluate regularly**: Use `evaluateModel()` during training

### Performance
1. **Use Metal GPU**: Automatic on Apple Silicon
2. **Configure threads appropriately**: Based on CPU cores
3. **Batch operations**: When possible
4. **Memory management**: Proper cleanup with cancellation

## 🔧 Troubleshooting

### Common Issues

#### Generation Issues
- **Context not initialized**: Ensure model is loaded
- **Conversation not found**: Check conversation ID
- **Cancellation not working**: Use `stream.cancel()` not `Task.cancel()`

#### Fine-tuning Issues
- **Adapter not found**: Check file path and permissions
- **Invalid LoRA rank**: Use values between 1-128
- **Training session not found**: Ensure session is started

### Error Messages
```swift
// Common error patterns
GenerationError.contextNotInitialized  // Model not loaded
GenerationError.conversationNotFound   // Invalid conversation ID
TuningError.adapterFileNotFound        // LoRA file missing
TuningError.invalidLoRARank           // Invalid rank value
```

## 📊 Monitoring

### Generation Metrics
```swift
// Track generation progress
var tokenCount = 0
for try await token in stream.stream {
    tokenCount += 1
    print("Token \(tokenCount): \(token)")
}
```

### Training Metrics
```swift
// Monitor training
let metrics = tuner.getTrainingMetrics()
if let metrics = metrics {
    print("Epoch: \(metrics.epoch)")
    print("Loss: \(metrics.loss)")
    print("Learning Rate: \(metrics.learningRate)")
}
```

### Evaluation Metrics
```swift
// Evaluate model performance
let evalMetrics = try tuner.evaluateModel(validationExamples: examples)
print("Perplexity: \(evalMetrics.perplexity)")
print("Average Loss: \(evalMetrics.averageLoss)")
print("Total Examples: \(evalMetrics.totalExamples)")
print("Total Tokens: \(evalMetrics.totalTokens)")
```

---

This quick reference provides the most common patterns and configurations for SwiftyLlama. For detailed API documentation, see `API_DOCUMENTATION.md`. 