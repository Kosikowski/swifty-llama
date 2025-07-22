import Foundation
import llama

// MARK: - Model Splitting Configuration

/// Configuration for SLlamaModelSplitting buffer sizes
public struct SLlamaModelSplittingConfig: Sendable {
    /// Maximum buffer size for split path generation
    public let maxSplitPathLength: Int32
    /// Maximum buffer size for path prefix extraction
    public let maxPathPrefixLength: Int32
    
    public init(
        maxSplitPathLength: Int32 = 1024,
        maxPathPrefixLength: Int32 = 1024
    ) {
        self.maxSplitPathLength = maxSplitPathLength
        self.maxPathPrefixLength = maxPathPrefixLength
    }
}

// MARK: - SLlamaModelSplitting

/// A wrapper for llama model splitting operations
public class SLlamaModelSplitting: @unchecked Sendable {
    /// Build a split model file path
    /// - Parameters:
    ///   - pathPrefix: Base path for the model files
    ///   - splitNumber: Current split number (0-based)
    ///   - totalSplits: Total number of splits
    ///   - config: Model splitting configuration (optional, uses defaults if nil)
    /// - Returns: Generated split path, or nil if failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func buildSplitPath(
        pathPrefix: String,
        splitNumber: Int,
        totalSplits: Int,
        config: SLlamaModelSplittingConfig? = nil
    )
        -> String?
    {
        let splittingConfig = config ?? SLlamaModelSplittingConfig()
        let maxLength = splittingConfig.maxSplitPathLength
        var splitPath = [CChar](repeating: 0, count: Int(maxLength))

        let result = llama_split_path(
            &splitPath,
            maxLength,
            pathPrefix,
            Int32(splitNumber),
            Int32(totalSplits)
        )

        if result > 0 {
            return String(cString: splitPath, encoding: .utf8) ?? ""
        }
        return nil
    }

    /// Extract the path prefix from a split path
    /// - Parameters:
    ///   - splitPath: Full path of a split file
    ///   - splitNumber: Expected split number
    ///   - totalSplits: Expected total number of splits
    ///   - config: Model splitting configuration (optional, uses defaults if nil)
    /// - Returns: Extracted path prefix, or nil if failed
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func extractPathPrefix(
        from splitPath: String,
        splitNumber: Int,
        totalSplits: Int,
        config: SLlamaModelSplittingConfig? = nil
    )
        -> String?
    {
        let splittingConfig = config ?? SLlamaModelSplittingConfig()
        let maxLength = splittingConfig.maxPathPrefixLength
        var pathPrefix = [CChar](repeating: 0, count: Int(maxLength))

        let result = llama_split_prefix(
            &pathPrefix,
            maxLength,
            splitPath,
            Int32(splitNumber),
            Int32(totalSplits)
        )

        if result > 0 {
            return String(cString: pathPrefix, encoding: .utf8) ?? ""
        }
        return nil
    }

    /// Generate all split paths for a model
    /// - Parameters:
    ///   - pathPrefix: Base path for the model files
    ///   - totalSplits: Total number of splits
    /// - Returns: Array of all split paths
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func generateAllSplitPaths(
        pathPrefix: String,
        totalSplits: Int
    )
        -> [String]
    {
        var paths: [String] = []

        for i in 0 ..< totalSplits {
            if
                let path = buildSplitPath(
                    pathPrefix: pathPrefix,
                    splitNumber: i,
                    totalSplits: totalSplits
                )
            {
                paths.append(path)
            }
        }

        return paths
    }

    /// Validate if a split path matches the expected pattern
    /// - Parameters:
    ///   - splitPath: Path to validate
    ///   - splitNumber: Expected split number
    ///   - totalSplits: Expected total number of splits
    /// - Returns: True if the path is valid for the given parameters
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func validateSplitPath(
        _ splitPath: String,
        splitNumber: Int,
        totalSplits: Int
    )
        -> Bool
    {
        extractPathPrefix(
            from: splitPath,
            splitNumber: splitNumber,
            totalSplits: totalSplits
        ) != nil
    }
}

// MARK: - SLlamaSplitModelInfo

/// Information about a split model
public struct SLlamaSplitModelInfo {
    // MARK: Properties

    public let pathPrefix: String
    public let splitNumber: Int
    public let totalSplits: Int
    public let splitPath: String

    // MARK: Lifecycle

    public init(pathPrefix: String, splitNumber: Int, totalSplits: Int) throws {
        self.pathPrefix = pathPrefix
        self.splitNumber = splitNumber
        self.totalSplits = totalSplits

        guard
            let path = SLlamaModelSplitting.buildSplitPath(
                pathPrefix: pathPrefix,
                splitNumber: splitNumber,
                totalSplits: totalSplits
            )
        else {
            throw NSError(
                domain: "SLlamaModelSplitting",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to generate split path"]
            )
        }

        splitPath = path
    }

    // MARK: Functions

    /// Get all split paths for this model
    /// - Returns: Array of all split paths
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getAllSplitPaths() -> [String] {
        SLlamaModelSplitting.generateAllSplitPaths(
            pathPrefix: pathPrefix,
            totalSplits: totalSplits
        )
    }

    /// Check if all split files exist
    /// - Returns: True if all split files exist
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func allSplitFilesExist() -> Bool {
        let paths = getAllSplitPaths()
        return paths.allSatisfy { FileManager.default.fileExists(atPath: $0) }
    }

    /// Get the size of all split files combined
    /// - Returns: Total size in bytes, or nil if any file is missing
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getTotalSize() -> Int64? {
        let paths = getAllSplitPaths()
        var totalSize: Int64 = 0

        for path in paths {
            guard
                let attributes = try? FileManager.default.attributesOfItem(atPath: path),
                let fileSize = attributes[.size] as? Int64
            else {
                return nil
            }
            totalSize += fileSize
        }

        return totalSize
    }
}
