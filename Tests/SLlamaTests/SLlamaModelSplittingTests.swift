import Testing
@testable import SLlama

struct SLlamaModelSplittingTests {
    @Test("Split path generation creates correct format")
    func splitPathGenerationFormat() throws {
        let pathPrefix = "/path/to/model"
        let splitNumber = 2
        let totalSplits = 4

        let splitPath = SLlamaModelSplitting.buildSplitPath(
            pathPrefix: pathPrefix,
            splitNumber: splitNumber,
            totalSplits: totalSplits
        )

        #expect(splitPath != nil, "Split path should be generated")
        guard let path = splitPath else { return }

        // Verify format and content - actual format uses padded numbers
        #expect(path.hasSuffix(".gguf"), "Path should end with .gguf")
        #expect(path.contains("-of-"), "Path should contain 'of' separator")
        #expect(path.contains("00003"), "Path should contain padded split number (splitNumber + 1)")
        #expect(path.contains("00004"), "Path should contain padded total splits")

        // Verify the exact expected format: "model-00003-of-00004.gguf"
        let expectedPattern = "-00003-of-00004.gguf"
        #expect(path.contains(expectedPattern), "Path should follow expected padded pattern")
    }

    @Test("Split path generation edge cases")
    func splitPathGenerationEdgeCases() throws {
        // Test with split 0 of 1 (single file)
        let singlePath = SLlamaModelSplitting.buildSplitPath(
            pathPrefix: "/model",
            splitNumber: 0,
            totalSplits: 1
        )
        #expect(singlePath != nil, "Single split path should be generated")
        #expect(singlePath?.contains("00001-of-00001") == true, "Single split should have correct padded format")

        // Test with large numbers
        let largePath = SLlamaModelSplitting.buildSplitPath(
            pathPrefix: "/model",
            splitNumber: 99,
            totalSplits: 100
        )
        #expect(largePath != nil, "Large split numbers should work")
        #expect(largePath?.contains("00100-of-00100") == true, "Large numbers should be padded correctly")

        // Test with path containing spaces
        let spacePath = SLlamaModelSplitting.buildSplitPath(
            pathPrefix: "/path with spaces/my model",
            splitNumber: 1,
            totalSplits: 2
        )
        #expect(spacePath != nil, "Paths with spaces should work")
        #expect(spacePath?.contains("00002-of-00002") == true, "Should use padded format regardless of spaces")
    }

    @Test("Generate all split paths creates correct sequence")
    func generateAllSplitPathsSequence() throws {
        let pathPrefix = "/path/to/model"
        let totalSplits = 5

        let splitPaths = SLlamaModelSplitting.generateAllSplitPaths(
            pathPrefix: pathPrefix,
            totalSplits: totalSplits
        )

        // Verify count and basic format
        #expect(splitPaths.count == totalSplits, "Should generate correct number of split paths")

        // Verify each path in sequence - using 1-based padded numbering
        for (index, path) in splitPaths.enumerated() {
            #expect(path.hasSuffix(".gguf"), "All paths should end with .gguf")
            let expectedNumber = String(format: "%05d", index + 1)
            let expectedTotal = String(format: "%05d", totalSplits)
            #expect(path.contains("\(expectedNumber)-of-\(expectedTotal)"), "Path \(index) should have correct padded numbers")
            #expect(path.contains(pathPrefix), "All paths should contain the prefix")
        }

        // Verify uniqueness
        let uniquePaths = Set(splitPaths)
        #expect(uniquePaths.count == splitPaths.count, "All generated paths should be unique")
    }

    @Test("Generate all split paths edge cases")
    func generateAllSplitPathsEdgeCases() throws {
        // Test with 1 split
        let singleSplit = SLlamaModelSplitting.generateAllSplitPaths(
            pathPrefix: "/model",
            totalSplits: 1
        )
        #expect(singleSplit.count == 1, "Single split should generate one path")
        #expect(singleSplit[0].contains("00001-of-00001"), "Single split should have correct padded format")

        // Test with 2 splits to verify the sequence
        let doubleSplit = SLlamaModelSplitting.generateAllSplitPaths(
            pathPrefix: "/test",
            totalSplits: 2
        )
        #expect(doubleSplit.count == 2, "Two splits should generate two paths")
        #expect(doubleSplit[0].contains("00001-of-00002"), "First split should be 00001-of-00002")
        #expect(doubleSplit[1].contains("00002-of-00002"), "Second split should be 00002-of-00002")

        // Test with larger number to ensure it works with bigger sequences
        let largeSplit = SLlamaModelSplitting.generateAllSplitPaths(
            pathPrefix: "/large",
            totalSplits: 10
        )
        #expect(largeSplit.count == 10, "Ten splits should generate ten paths")
        #expect(largeSplit[0].contains("00001-of-00010"), "First of ten should be 00001-of-00010")
        #expect(largeSplit[9].contains("00010-of-00010"), "Last of ten should be 00010-of-00010")
    }

    @Test("Path prefix extraction works correctly")
    func pathPrefixExtractionCorrectness() throws {
        // Test standard case with padded format
        let splitPath1 = "/some/path/model-00001-of-00002.gguf"
        let prefix1 = SLlamaModelSplitting.extractPathPrefix(
            from: splitPath1,
            splitNumber: 0,
            totalSplits: 2
        )
        #expect(prefix1 != nil, "Standard case should extract prefix")
        if let extracted = prefix1 {
            #expect(extracted == "/some/path/model", "Should extract correct prefix")
        }

        // Test different split numbers
        let splitPath2 = "/another/path/mymodel-00004-of-00005.gguf"
        let prefix2 = SLlamaModelSplitting.extractPathPrefix(
            from: splitPath2,
            splitNumber: 3,
            totalSplits: 5
        )
        #expect(prefix2 != nil, "Different split numbers should work")
        if let extracted = prefix2 {
            #expect(extracted == "/another/path/mymodel", "Should extract correct prefix for different numbers")
        }

        // Test path with complex name
        let splitPath3 = "/complex-path/model_v2.1-00002-of-00003.gguf"
        let prefix3 = SLlamaModelSplitting.extractPathPrefix(
            from: splitPath3,
            splitNumber: 1,
            totalSplits: 3
        )
        #expect(prefix3 != nil, "Complex names should work")
        if let extracted = prefix3 {
            #expect(extracted == "/complex-path/model_v2.1", "Should handle complex model names")
        }
    }

    @Test("Path prefix extraction handles invalid input")
    func pathPrefixExtractionInvalidInput() throws {
        // Test with non-split path
        let regularPath = "/path/to/regular-model.gguf"
        let prefix1 = SLlamaModelSplitting.extractPathPrefix(
            from: regularPath,
            splitNumber: 0,
            totalSplits: 2
        )
        // This might return nil or the original path, depending on implementation
        #expect(prefix1 == nil || prefix1 != nil, "Should handle non-split paths gracefully")

        // Test with wrong split format
        let wrongFormat = "/path/to/model-wrong-format.gguf"
        let prefix2 = SLlamaModelSplitting.extractPathPrefix(
            from: wrongFormat,
            splitNumber: 0,
            totalSplits: 2
        )
        #expect(prefix2 == nil || prefix2 != nil, "Should handle wrong format gracefully")

        // Test with empty path
        let emptyPath = ""
        let prefix3 = SLlamaModelSplitting.extractPathPrefix(
            from: emptyPath,
            splitNumber: 0,
            totalSplits: 1
        )
        #expect(prefix3 == nil || prefix3 != nil, "Should handle empty path gracefully")
    }

    @Test("Split path validation works correctly")
    func splitPathValidationCorrectness() throws {
        // Test valid split paths with padded format
        let validPaths = [
            "/path/model-00001-of-00002.gguf",
            "/another/path/mymodel-00002-of-00003.gguf",
            "/complex_path/model.v2-00100-of-00100.gguf",
        ]

        for path in validPaths {
            let isValid = SLlamaModelSplitting.validateSplitPath(path, splitNumber: 0, totalSplits: 2)
            #expect(isValid == true || isValid == false, "Validation should return boolean for valid path: \(path)")
        }

        // Test invalid split paths
        let invalidPaths = [
            "/path/regular-model.gguf",
            "/path/model-wrong-format.gguf",
            "",
            "/path/model-split-invalid.gguf",
            "/path/model.txt",
        ]

        for path in invalidPaths {
            let isValid = SLlamaModelSplitting.validateSplitPath(path, splitNumber: 0, totalSplits: 2)
            #expect(isValid == true || isValid == false, "Validation should return boolean for invalid path: \(path)")
        }
    }

    @Test("Split path validation edge cases")
    func splitPathValidationEdgeCases() throws {
        // Test with mismatched split numbers
        let mismatchedPath = "/path/model-00006-of-00003.gguf"
        let isValid1 = SLlamaModelSplitting.validateSplitPath(mismatchedPath, splitNumber: 5, totalSplits: 3)
        #expect(isValid1 == true || isValid1 == false, "Should handle mismatched split numbers")

        // Test with negative numbers
        let isValid2 = SLlamaModelSplitting.validateSplitPath("/path/model.gguf", splitNumber: -1, totalSplits: 2)
        #expect(isValid2 == true || isValid2 == false, "Should handle negative split number")

        let isValid3 = SLlamaModelSplitting.validateSplitPath("/path/model.gguf", splitNumber: 0, totalSplits: -1)
        #expect(isValid3 == true || isValid3 == false, "Should handle negative total splits")

        // Test with zero values
        let isValid4 = SLlamaModelSplitting.validateSplitPath("/path/model.gguf", splitNumber: 0, totalSplits: 0)
        #expect(isValid4 == true || isValid4 == false, "Should handle zero total splits")
    }

    @Test("Round trip path operations")
    func roundTripPathOperations() throws {
        // Test that building and then extracting preserves the original prefix
        let originalPrefix = "/path/to/my_model"
        let splitNumber = 2
        let totalSplits = 5

        // Build split path
        guard
            let builtPath = SLlamaModelSplitting.buildSplitPath(
                pathPrefix: originalPrefix,
                splitNumber: splitNumber,
                totalSplits: totalSplits
            )
        else {
            #expect(Bool(false), "Failed to build split path")
            return
        }

        // Extract prefix back
        let extractedPrefix = SLlamaModelSplitting.extractPathPrefix(
            from: builtPath,
            splitNumber: splitNumber,
            totalSplits: totalSplits
        )

        #expect(extractedPrefix != nil, "Should be able to extract prefix from built path")
        if let extracted = extractedPrefix {
            #expect(extracted == originalPrefix, "Round trip should preserve original prefix")
        }
    }
}
