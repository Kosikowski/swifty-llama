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

        // Try to copy the model file to the bundle if it doesn't exist
        let bundlePath = Bundle.module.bundlePath
        let targetPath = "\(bundlePath)/tinystories-gpt-0.1-3m.fp16.gguf"

        // Check if the model already exists in the bundle
        if FileManager.default.fileExists(atPath: targetPath) {
            return targetPath
        }

        // Try to copy from the source location
        let sourcePaths = [
            "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "./Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
            "../../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
        ]

        for sourcePath in sourcePaths {
            if FileManager.default.fileExists(atPath: sourcePath) {
                do {
                    try FileManager.default.copyItem(atPath: sourcePath, toPath: targetPath)
                    return targetPath
                } catch {
                    // Continue to next path if copy fails
                    continue
                }
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
    public static func skipIfModelUnavailable(testName: String = #function) {
        guard isTestModelAvailable() else {
            print("⚠️ Test '\(testName)' skipped: Model file not found at \(testModelPath)")
            return
        }
    }

    /// Check if we're running in iOS Simulator and skip model tests if needed
    /// - Parameter testName: Name of the test being skipped
    public static func skipIfIOSSimulator(testName: String = #function) {
        #if targetEnvironment(simulator)
            print("⚠️ Test '\(testName)' skipped: Model loading not supported in iOS Simulator")
            return
        #endif
    }

    /// Try to load the model and return success status
    /// - Returns: True if model loads successfully, false otherwise
    public static func canLoadModel() -> Bool {
        guard isTestModelAvailable() else { return false }

        do {
            SLlama.initialize()
            defer { SLlama.cleanup() }

            let _ = try SLlamaModel(modelPath: testModelPath)
            return true
        } catch {
            print("❌ Model loading failed: \(error)")
            return false
        }
    }

    /// Skip test if model cannot be loaded (useful for iOS Simulator)
    /// - Parameter testName: Name of the test being skipped
    public static func skipIfModelCannotLoad(testName: String = #function) {
        guard canLoadModel() else {
            print("⚠️ Test '\(testName)' skipped: Model cannot be loaded at \(testModelPath)")
            return
        }
    }
}
