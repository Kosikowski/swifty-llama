import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct SwiftyLlamaCppTests {
    
    @Test("Package compilation and basic types")
    func testPackageCompilation() throws {
        // Test that the package compiles and basic types are available
        #expect(SwiftyLlamaCpp.self != nil, "SwiftyLlamaCpp class should be available")
        #expect(LlamaModel.self != nil, "LlamaModel class should be available")
        #expect(LlamaContext.self != nil, "LlamaContext class should be available")
        #expect(LlamaVocab.self != nil, "LlamaVocab class should be available")
    }
    
    @Test("Type aliases are properly defined")
    func testTypeAliases() throws {
        // Test that type aliases are properly defined
        #expect(type(of: LlamaTokenNull) == LlamaToken.self, "LlamaTokenNull should be LlamaToken type")
        #expect(type(of: LlamaDefaultSeed) == UInt32.self, "LlamaDefaultSeed should be UInt32 type")
        #expect(type(of: LlamaFileMagicGGLA) == UInt32.self, "LlamaFileMagicGGLA should be UInt32 type")
    }
    
    @Test("Enum type aliases work correctly")
    func testEnumTypeAliases() throws {
        // Test enum type aliases exist and have correct types
        #expect(LlamaVocabType.none.rawValue == 0, "LlamaVocabType.none should have raw value 0")
        #expect(LlamaRopeType.norm.rawValue == 0, "LlamaRopeType.norm should have raw value 0")
        #expect(LlamaTokenType.normal.rawValue == 1, "LlamaTokenType.normal should have raw value 1")
        #expect(LlamaFileType.mostlyQ4_0.rawValue == 2, "LlamaFileType.mostlyQ4_0 should have raw value 2")
    }
    
    @Test("Model creation with invalid path")
    func testModelCreationWithInvalidPath() throws {
        // Test that model creation fails gracefully with invalid path
        let model = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        #expect(model == nil, "Model creation should fail with invalid path")
    }
    
    @Test("Context creation with invalid model")
    func testContextCreationWithInvalidModel() throws {
        // Test that context creation fails gracefully with invalid model
        let invalidModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = invalidModel {
            let context = LlamaContext(model: model)
            #expect(context == nil, "Context creation should fail with invalid model")
        } else {
            // If model creation itself failed, that's also acceptable
            #expect(true, "Model creation failed as expected")
        }
    }
    
    @Test("Vocabulary creation with nil pointer")
    func testVocabularyCreationWithNilPointer() throws {
        // Test that vocabulary creation fails gracefully with nil pointer
        let vocab = LlamaVocab(vocab: nil)
        #expect(vocab == nil, "Vocabulary creation should fail with nil pointer")
    }
    
    @Test("Basic type definitions")
    func testBasicTypeDefinitions() throws {
        // Test that basic C types are properly bridged
        #expect(type(of: LlamaTokenNull) == LlamaToken.self, "LlamaTokenNull should be LlamaToken")
        #expect(type(of: LlamaVocabType.none) == LlamaVocabType.self, "LlamaVocabType.none should be LlamaVocabType")
        #expect(type(of: LlamaTokenAttribute.undefined) == LlamaTokenAttribute.self, "LlamaTokenAttribute.undefined should be LlamaTokenAttribute")
    }
    
    @Test("Pointer type aliases")
    func testPointerTypeAliases() throws {
        // Test that pointer type aliases are properly defined
        #expect(LlamaModelPointer.self == OpaquePointer.self, "LlamaModelPointer should be OpaquePointer")
        #expect(LlamaContextPointer.self == OpaquePointer.self, "LlamaContextPointer should be OpaquePointer")
        #expect(LlamaVocabPointer.self == OpaquePointer.self, "LlamaVocabPointer should be OpaquePointer")
    }
    
    @Test("Struct type aliases")
    func testStructTypeAliases() throws {
        // Test that struct type aliases are properly defined
        #expect(LlamaTokenData.self != nil, "LlamaTokenData should be defined")
        #expect(LlamaBatch.self != nil, "LlamaBatch should be defined")
        #expect(LlamaModelParams.self != nil, "LlamaModelParams should be defined")
        #expect(LlamaContextParams.self != nil, "LlamaContextParams should be defined")
    }
} 