import Foundation
@testable import SLlama

/// Test utilities for SLlama tests
public enum SLlamaTestUtilities {
    // MARK: Static Properties

    /// Default model path for tests - resolved to absolute path
    public static let testModelPath: String = {
        // Try different possible locations for the model file
        let possiblePaths = [
            "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "./Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "../../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
        ]

        // Also try to find it relative to the source file location
        let currentFile = #file
        let currentDir = URL(fileURLWithPath: currentFile).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let bundlePath = currentDir.appendingPathComponent("Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf").path

        let allPaths = possiblePaths + [bundlePath]

        for path in allPaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        // Fallback to original relative path
        return "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
    }()

    // MARK: Static Computed Properties

    /// Check if test model exists
    public static var testModelExists: Bool {
        FileManager.default.fileExists(atPath: testModelPath)
    }

    // MARK: Static Functions

    /// Initialize SLlama backend for testing
    public static func setupSLlama() {
        SLlama.initialize()
    }

    /// Cleanup SLlama backend after testing
    public static func teardownSLlama() {
        SLlama.cleanup()
    }

    /// Run a test with proper SLlama setup and teardown
    public static func withSLlamaSetup<T>(_ test: () throws -> T) throws -> T {
        setupSLlama()
        defer { teardownSLlama() }
        return try test()
    }

    /// Skip test if model doesn't exist
    public static func skipIfModelMissing() {
        guard !testModelExists else { return }
        print("Test skipped: Model file not found at \(testModelPath)")
    }
}
