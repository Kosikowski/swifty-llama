import Foundation

/// Shared test utilities for both SLlama and SwiftyLlama tests
public enum TestUtilities {
    // MARK: Static Properties

    /// Default model path for tests - using direct file paths
    public static let testModelPath: String = {
        // First check if there's an environment variable set
        if let envPath = ProcessInfo.processInfo.environment["SLLAMA_TEST_MODEL_PATH"] {
            if FileManager.default.fileExists(atPath: envPath) {
                #if DEBUG
                    print("✅ TestUtilities: Found model via environment variable: \(envPath)")
                #endif
                return envPath
            }
        }

        // Get the current working directory
        let currentDirectory = FileManager.default.currentDirectoryPath

        let url = Bundle.module.url(forResource: "tinystories-gpt-0.1-3m.fp16", withExtension: "gguf")!
        print(url)
        print(url.path())
        // Try to find the model in common locations, including Xcode's working directory
        let sourcePaths = [
            url.path(),
//            // Relative to current working directory
//            "Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "./Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "../Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "../../Models/tinystories-gpt-0.1-3m.fp16.gguf",
//
//            // Try to find the project root by looking for Package.swift
//            findProjectRoot() + "/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//
//            // Common Xcode test working directories
//            "/Users/\(NSUserName())/Projects/swifty-llama-cpp/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "/Users/\(NSUserName())/Development/swifty-llama-cpp/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//
//            // Relative to project root (common Xcode working directory)
//            "swifty-llama/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "./swifty-llama/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "../swifty-llama/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "../../swifty-llama/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//
//            // Relative to DerivedData (Xcode test working directory)
//            "SourcePackages/checkouts/swifty-llama-cpp/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "./SourcePackages/checkouts/swifty-llama-cpp/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//
//            // Absolute paths based on common Xcode working directories
//            "\(currentDirectory)/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "\(currentDirectory)/swifty-llama-cpp/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//
//            // Test-specific paths
//            "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "./Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
//            "../../Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf",
        ]

        // Debug: Print current directory and search paths
        #if DEBUG
            print("🔍 TestUtilities: Current working directory: \(currentDirectory)")
            print("🔍 TestUtilities: Project root: \(findProjectRoot())")
            print("🔍 TestUtilities: Searching for model in paths:")
            for (index, path) in sourcePaths.enumerated() {
                let exists = FileManager.default.fileExists(atPath: path)
                if exists {
                    print("   \(index): \(path) - \(exists ? "✅ Found" : "❌ Not found")")
                }
            }
        #endif

        for sourcePath in sourcePaths {
            if FileManager.default.fileExists(atPath: sourcePath) {
                #if DEBUG
                    print("✅ TestUtilities: Found model at: \(sourcePath)")
                #endif
                return sourcePath
            }
        }

        // If nothing works, return the first fallback path and let the test handle the missing file
        #if DEBUG
            print("❌ TestUtilities: Model not found in any path, using fallback: \(sourcePaths[0])")
            print("💡 Tip: Set SLLAMA_TEST_MODEL_PATH environment variable to specify model location")
        #endif
        return sourcePaths[0]
    }()

    // MARK: - Helper Methods

    /// Find the project root by looking for Package.swift
    private static func findProjectRoot() -> String {
        var currentPath = FileManager.default.currentDirectoryPath

        // Walk up the directory tree looking for Package.swift
        while !currentPath.isEmpty, currentPath != "/" {
            let packagePath = currentPath + "/Package.swift"
            if FileManager.default.fileExists(atPath: packagePath) {
                return currentPath
            }
            currentPath = (currentPath as NSString).deletingLastPathComponent
        }

        // If we can't find Package.swift, try to guess based on common patterns
        let username = NSUserName()
        let commonPaths = [
            "/Users/\(username)/Projects/swifty-llama-cpp",
            "/Users/\(username)/Development/swifty-llama-cpp",
            "/Users/\(username)/Code/swifty-llama-cpp",
        ]

        for path in commonPaths {
            if FileManager.default.fileExists(atPath: path + "/Package.swift") {
                return path
            }
        }

        return FileManager.default.currentDirectoryPath
    }

    // MARK: Utility Methods

    /// Check if the test model is available
    /// - Returns: True if the test model file exists and is readable
    public static func isTestModelAvailable() -> Bool {
        let available = FileManager.default.fileExists(atPath: testModelPath) &&
            FileManager.default.isReadableFile(atPath: testModelPath)

        #if DEBUG
            if !available {
                print("❌ TestUtilities: Model not available at path: \(testModelPath)")
                print("   Current working directory: \(FileManager.default.currentDirectoryPath)")
                print("   Project root: \(findProjectRoot())")
                print("💡 To fix this:")
                print("   1. Set SLLAMA_TEST_MODEL_PATH environment variable")
                print("   2. Copy model to: \(findProjectRoot())/Models/")
                print("   3. Set Xcode scheme working directory to project root")
            }
        #endif

        return available
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
