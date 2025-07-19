# 🔮 Omen

> 🔮 "The future leaves traces."

Omen isn't just a logger — it's a prophetic trace system for Swift. 🔮
Every action, every event, every warning is a sign.
A whisper before the crash. A shadow of what's to come.

Designed to feel like magic, but built on solid observability,
Omen captures the soul of your app — so you can read its fate
before the exception even knows it's coming. 🔮

Whether you're debugging live issues or watching quiet patterns,
Omen gives you visibility with style.
Think structured logs, symbol-rich traces, and just enough drama.

## ✨ Features

### 🔮 **Mystical Categories**
- ⚡️ **Performance** — Speed omens and timing visions
- 💾 **Memory** — Memory omens from allocation patterns
- 🌐 **Network** — Network omens from connectivity visions  
- 🔒 **Security** — Security omens from protection spells
- 🗄️ **Database** — Database omens from data visions
- ✨ **General** — Universal omens for everything else

*🔮 Extensible by users — create your own mystical categories!*

### 🌟 **Prophetic Levels**
- 🔍 **Debug** — Whispers from the depths
- ℹ️ **Info** — Signs in plain sight  
- 📋 **Notice** — Omens of note
- ⚠️ **Warning** — Shadows of trouble ahead
- ❌ **Error** — Dark portents manifested
- 💥 **Fault** — Apocalyptic visions

## 🚀 Usage

### 🔮 Static API (Convenience)
```swift
import Omen

// Core category omens
Omen.performance("Model loaded successfully")
Omen.memory("Allocated 256MB for model weights")
Omen.security("Authentication token validated")

// Custom categories with levels
Omen.info(MyCategories.ai, "Neural network initialized")
Omen.warning(MyCategories.database, "Connection pool running low")

// Time measurement magic 🔮
let result = Omen.measureTime(operation: "Model inference") {
    return model.generate(prompt: "Tell me the future...")
}
```

### 🔮 Protocol-Based (Dependency Injection)
```swift
import Omen

class MyService {
    private let logger: OmenLogger
    
    init(logger: OmenLogger = Omen.shared) {
        self.logger = logger
    }
    
    func processData() {
        logger.performance("Data processing started")
        // ... your mystical code ... 🔮
        logger.performance("Data processing completed")
    }
}

// For testing - Mock the oracle 🔮
class MockOmen: OmenLogger {
    func log<T: OmenCategory>(_ level: Omen.Level, category: T, _ message: () -> String) {
        // Capture omens for testing
    }
    
    func setEnabled(_ enabled: Bool) { }
    func setMinimumLogLevel(_ level: Omen.Level) { }
}
```

### 🔮 Extending Categories
```swift
// Create your own mystical categories
public enum MyCategories: String, OmenCategory, CaseIterable {
    case ai = "AI"
    case blockchain = "Blockchain"
    
    public var description: String {
        switch self {
        case .ai: return "🤖 Artificial intelligence omens"
        case .blockchain: return "⛓️ Blockchain transaction omens"
        }
    }
    
    public var symbol: String {
        switch self {
        case .ai: return "🤖"
        case .blockchain: return "⛓️"
        }
    }
}

// Register your categories 🔮
OmenCategories.register(MyCategories.ai)
OmenCategories.register(MyCategories.blockchain)

// Use them immediately
Omen.info(MyCategories.ai, "Neural network awakening...")
```

### 🔮 Configuration
```swift
// Control the oracle's sight
Omen.setEnabled(false)  // Silence all omens
Omen.setMinimumLogLevel(.warning)  // Only shadows and portents

// Per-instance configuration  
let customOmen = Omen()
customOmen.setMinimumLogLevel(.debug)
```

## 🔮 Architecture

### **🔮 Protocol-Based Design**
- `OmenLogger` protocol for dependency injection
- `OmenCategory` protocol for extensible categories
- `@inlinable` methods for zero-overhead performance
- Thread-safe with `@unchecked Sendable` compliance
- Built on Apple's `os_log` for system integration

### **🔮 Performance Magic**
- **Inlinable static methods** — Zero overhead when optimized
- **Lazy evaluation** — Messages only evaluated if logged
- **Queue-based synchronization** — Thread-safe configuration
- **os_log integration** — Native performance and privacy

### **🔮 Observability**
- **Console.app integration** — View logs in system tools
- **Instruments compatibility** — Profile with Apple's tools  
- **Privacy controls** — Automatic log rotation and filtering
- **Structured categories** — Easy filtering and analysis

## 🎨 Design Philosophy

🔮 Omen believes that logging should be both **functional** and **beautiful**.
Every log message is a story, every category a character,
every level a plot twist in your app's narrative.

The mystical theming isn't just aesthetic — it encourages developers
to think about logs as **signals** and **patterns** rather than
just debugging noise. 🔮

When you see `⚠️ Warning shadows — for trouble brewing ahead`,
you're more likely to pay attention than to a plain "WARNING".

## 🔧 Integration

### 🔮 Swift Package Manager
```swift
dependencies: [
    .package(path: "path/to/Omen")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: ["Omen"]
    )
]
```

### 🔮 Pre-commit Hooks
Omen integrates with development workflows to ensure clean logging:
- Detects stray `print()` statements
- Enforces proper category usage  
- Validates log message formatting

---

*🔮 Remember: The future leaves traces. Make sure yours tell the right story.* 🔮✨ 