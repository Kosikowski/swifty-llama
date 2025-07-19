import Foundation
@testable import SLlama

/// Test utilities for SLlama tests
public enum SLlamaTestUtilities {
    // MARK: Static Properties

    /// Default model path for tests
    public static let testModelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"

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
