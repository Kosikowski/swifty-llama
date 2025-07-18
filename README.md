# SwiftyLlamaCpp

A Swift wrapper for the [llama.cpp](https://github.com/ggerganov/llama.cpp) library, providing a native Swift interface to run Large Language Models (LLMs) on Apple platforms.

## Features

- 🚀 Native Swift interface to llama.cpp
- 📱 Support for iOS, macOS, tvOS, and visionOS
- 🔧 Metal acceleration support
- 🎯 Easy-to-use Swift API
- 📦 Swift Package Manager integration

## Requirements

- iOS 13.0+
- macOS 11.0+
- tvOS 13.0+
- visionOS 1.0+
- Swift 5.9+

## Installation

### Swift Package Manager

Add SwiftyLlamaCpp to your project in Xcode:

1. File → Add Package Dependencies
2. Enter the repository URL
3. Select the package and add it to your target

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/your-username/swifty-llama-cpp.git", from: "1.0.0")
]
```

## Usage

### Basic Setup

```swift
import SwiftyLlamaCpp

// Initialize the library
SwiftyLlamaCpp.initialize()

// Get library information
let version = SwiftyLlamaCpp.getVersion()
let buildInfo = SwiftyLlamaCpp.getBuildInfo()

// Check Metal support
let supportsMetal = SwiftyLlamaCpp.supportsMetal()
let deviceCount = SwiftyLlamaCpp.getMetalDeviceCount()
```

### Working with Models

```swift
// Load a model
guard let model = LlamaModel(modelPath: "/path/to/your/model.gguf") else {
    print("Failed to load model")
    return
}

// Create a context
guard let context = LlamaContext(modelPath: "/path/to/your/model.gguf") else {
    print("Failed to create context")
    return
}
```

## Architecture

This package includes:

- **Binary Target**: The `llama.xcframework` containing the compiled llama.cpp library for multiple Apple platforms
- **Swift Wrapper**: High-level Swift classes that provide a native interface to the C library
- **Platform Support**: Support for iOS, macOS, tvOS, and visionOS with appropriate optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 