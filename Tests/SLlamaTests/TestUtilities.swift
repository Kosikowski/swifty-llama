import Foundation
@testable import SLlama

/// Test utilities for SLlama tests
public enum SLlamaTestUtilities {
    // MARK: Static Properties

    /// Default model path for tests - resolved from bundle resources
    public static let testModelPath: String = {
        // Try to get the model from bundle resources first
        if let bundleURL = Bundle.module.url(forResource: "tinystories-gpt-0.1-3m.fp16", withExtension: "gguf") {
            return bundleURL.path
        }

        // Fallback to bundle path method
        if let bundlePath = Bundle.module.path(forResource: "tinystories-gpt-0.1-3m.fp16", ofType: "gguf") {
            return bundlePath
        }

        // Final fallback: try relative paths (for development/debugging)
        let fallbackPaths = [
            "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "./Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "../../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
        ]

        for path in fallbackPaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        // If nothing works, return the first fallback path and let the test handle the missing file
        return fallbackPaths[0]
    }()

    // MARK: Utility Methods

    /// Check if the test model is available
    /// - Returns: True if the test model file exists and is readable
    public static func isTestModelAvailable() -> Bool {
        FileManager.default.fileExists(atPath: testModelPath) &&
            FileManager.default.isReadableFile(atPath: testModelPath)
    }

    /// Get the test model file size
    /// - Returns: Size in bytes, or nil if file doesn't exist
    public static func getTestModelSize() -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: testModelPath),
              let size = attributes[.size] as? Int64
        else {
            return nil
        }
        return size
    }

    /// Skip test with message if model is not available
    /// - Parameter testName: Name of the test being skipped
    public static func skipIfModelUnavailable(testName: String = #function) {
        guard isTestModelAvailable() else {
            print("⚠️ Test '\(testName)' skipped: Model file not found at \(testModelPath)")
            return
        }
    }
}
