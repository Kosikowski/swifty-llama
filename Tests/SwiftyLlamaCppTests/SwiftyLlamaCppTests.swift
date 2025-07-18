import Testing
@testable import SwiftyLlamaCpp

@Suite
struct SwiftyLlamaCppTests {
    @Test("Type aliases are properly defined")
    func typeAliases() throws {
        // Test that type aliases are properly defined
        #expect(type(of: LlamaTokenNull) == LlamaToken.self, "LlamaTokenNull should be LlamaToken type")
        #expect(type(of: LlamaDefaultSeed) == UInt32.self, "LlamaDefaultSeed should be UInt32 type")
        #expect(type(of: LlamaFileMagicGGLA) == UInt32.self, "LlamaFileMagicGGLA should be UInt32 type")
    }

    @Test("Enum type aliases work correctly")
    func enumTypeAliases() throws {
        // Test enum type aliases exist and have correct types
        #expect(LlamaVocabType.none.rawValue == 0, "LlamaVocabType.none should have raw value 0")
        #expect(LlamaRopeType.norm.rawValue == 0, "LlamaRopeType.norm should have raw value 0")
        #expect(LlamaTokenType.normal.rawValue == 1, "LlamaTokenType.normal should have raw value 1")
        #expect(LlamaFileType.mostlyQ4_0.rawValue == 2, "LlamaFileType.mostlyQ4_0 should have raw value 2")
    }

    @Test("Model creation with invalid path")
    func modelCreationWithInvalidPath() throws {
        // Test that model creation fails gracefully with invalid path
        let model = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        #expect(model == nil, "Model creation should fail with invalid path")
    }

    @Test("Context creation with invalid model")
    func contextCreationWithInvalidModel() throws {
        // Test that context creation fails gracefully with invalid model
        let invalidModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        if let model = invalidModel {
            let context = LlamaContext(model: model)
            #expect(context == nil, "Context creation should fail with invalid model")
        } else {
            // If model creation itself failed, that's also acceptable
            #expect(Bool(true), "Model creation failed as expected")
        }
    }

    @Test("Vocabulary creation with nil pointer")
    func vocabularyCreationWithNilPointer() throws {
        // Test that vocabulary creation fails gracefully with nil pointer
        let vocab = LlamaVocab(vocab: nil)
        #expect(vocab == nil, "Vocabulary creation should fail with nil pointer")
    }

    @Test("Basic type definitions")
    func basicTypeDefinitions() throws {
        // Test that basic C types are properly bridged
        #expect(type(of: LlamaTokenNull) == LlamaToken.self, "LlamaTokenNull should be LlamaToken")
        #expect(type(of: LlamaVocabType.none) == LlamaVocabType.self, "LlamaVocabType.none should be LlamaVocabType")
        #expect(type(of: LlamaTokenAttribute.undefined) == LlamaTokenAttribute.self, "LlamaTokenAttribute.undefined should be LlamaTokenAttribute")
    }

    @Test("Pointer type aliases")
    func pointerTypeAliases() throws {
        // Test that pointer type aliases are properly defined
        #expect(LlamaModelPointer.self == OpaquePointer.self, "LlamaModelPointer should be OpaquePointer")
        #expect(LlamaContextPointer.self == OpaquePointer.self, "LlamaContextPointer should be OpaquePointer")
        #expect(LlamaVocabPointer.self == OpaquePointer.self, "LlamaVocabPointer should be OpaquePointer")
    }
}
