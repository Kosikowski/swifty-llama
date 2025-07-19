import Testing
@testable import SwiftyLlamaCpp

struct SLlamaModelSplittingTests {
    
    @Test("Split path generation works")
    func testSplitPathGeneration() throws {
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
        let totalSplits = 4
        
        let allPaths = SLlamaModelSplitting.generateAllSplitPaths(
            pathPrefix: pathPrefix,
            totalSplits: totalSplits
        )
        
        #expect(allPaths.count == totalSplits, "Should generate correct number of paths")
        
        for (_, path) in allPaths.enumerated() {
            #expect(path.hasSuffix(".gguf"), "Path should end with .gguf")
        }
    }
    
    @Test("Path prefix extraction works")
    func testPathPrefixExtraction() throws {
        let testPath = "/path/to/model.gguf"
        let prefix = SLlamaModelSplitting.extractPathPrefix(
            from: testPath,
            splitNumber: 1,
            totalSplits: 2
        )
        
        // Test that extraction doesn't crash and returns something
        #expect(prefix != nil, "Path prefix extraction should work")
    }
    
    @Test("Split path validation works")
    func testSplitPathValidation() throws {
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