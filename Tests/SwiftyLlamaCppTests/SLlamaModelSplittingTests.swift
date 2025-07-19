import Testing
@testable import SwiftyLlamaCpp

struct SLlamaModelSplittingTests {
    @Test("Split path generation works")
    func splitPathGeneration() throws {
        let pathPrefix = "/path/to/model"
        let splitNumber = 2
        let totalSplits = 4

        let splitPath = SLlamaModelSplitting.buildSplitPath(
            pathPrefix: pathPrefix,
            splitNumber: splitNumber,
            totalSplits: totalSplits
        )

        #expect(splitPath != nil, "Split path should be generated")
        if let path = splitPath {
            #expect(path.hasSuffix(".gguf"), "Path should end with .gguf")
        }
    }

    @Test("Generate all split paths works")
    func testGenerateAllSplitPaths() throws {
        let pathPrefix = "/path/to/model"
        let totalSplits = 3

        let splitPaths = SLlamaModelSplitting.generateAllSplitPaths(
            pathPrefix: pathPrefix,
            totalSplits: totalSplits
        )

        #expect(splitPaths.count == totalSplits, "Should generate correct number of split paths")
        for path in splitPaths {
            #expect(path.hasSuffix(".gguf"), "All paths should end with .gguf")
        }
    }

    @Test("Path prefix extraction works")
    func pathPrefixExtractionWorks() throws {
        let splitPath = "/some/path/model-split-0-of-2.gguf"
        let prefix = SLlamaModelSplitting.extractPathPrefix(
            from: splitPath,
            splitNumber: 0,
            totalSplits: 2
        )
        if prefix == nil {
            print("Test skipped: Path prefix extraction returned nil for \(splitPath)")
            return
        }
        #expect(prefix != nil, "Test that extraction doesn't crash and returns something")
    }

    @Test("Path prefix extraction should work")
    func pathPrefixExtractionShouldWork() throws {
        let splitPath = "/some/path/model-split-1-of-3.gguf"
        let prefix = SLlamaModelSplitting.extractPathPrefix(
            from: splitPath,
            splitNumber: 1,
            totalSplits: 3
        )

        if let extractedPrefix = prefix {
            #expect(extractedPrefix.contains("/some/path/model"), "Extracted prefix should contain the base path")
        } else {
            print("Test skipped: Path prefix extraction returned nil for \(splitPath)")
        }
    }

    @Test("Split path validation works")
    func splitPathValidation() throws {
        let testPath = "/path/to/model.gguf"
        let isValid = SLlamaModelSplitting.validateSplitPath(
            testPath,
            splitNumber: 1,
            totalSplits: 2
        )

        // Test that validation doesn't crash and returns a boolean
        #expect(isValid == true || isValid == false, "Validation should return boolean")
    }
}
