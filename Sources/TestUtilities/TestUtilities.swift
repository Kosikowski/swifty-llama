import Foundation

/// Shared test utilities for both SLlama and SwiftyLlama tests
public enum TestUtilities {
    // MARK: Static Properties

    /// Default model path for tests - using direct file paths
    public static let testModelPath: String = {
        // Try to find the model in common locations
        let sourcePaths = [
            "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "./Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "../../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
        ]

        for sourcePath in sourcePaths {
            if FileManager.default.fileExists(atPath: sourcePath) {
                return sourcePath
            }
        }

        // If nothing works, return the first fallback path and let the test handle the missing file
        return sourcePaths[0]
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
    public static func skipIfModelUnavailable(testName _: String = #function) {
        guard isTestModelAvailable() else {
            // Test skipped due to missing model file
            return
        }
    }

    /// Check if we're running in iOS Simulator and skip model tests if needed
    /// - Parameter testName: Name of the test being skipped
    public static func skipIfIOSSimulator(testName _: String = #function) {
        #if targetEnvironment(simulator)
            // Test skipped due to iOS Simulator environment
            return
        #endif
    }

    /// Try to load the model and return success status (SLlama specific)
    /// - Returns: True if model loads successfully, false otherwise
    public static func canLoadModel() -> Bool {
        guard isTestModelAvailable() else { return false }

        // This function requires SLlama to be available
        // It will be implemented in the specific test utilities that import SLlama
        return false
    }

    /// Skip test if model cannot be loaded (useful for iOS Simulator)
    /// - Parameter testName: Name of the test being skipped
    public static func skipIfModelCannotLoad(testName _: String = #function) {
        guard canLoadModel() else {
            // Test skipped due to model loading failure
            return
        }
    }
}
