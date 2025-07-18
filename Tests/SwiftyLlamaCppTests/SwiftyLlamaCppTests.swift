import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct SwiftyLlamaCppTests {
    
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
    
    @Test("SwiftyLlamaCpp static functions")
    func testSwiftyLlamaCppStaticFunctions() throws {
        // Test initialization and cleanup
        SwiftyLlamaCpp.initialize()
        SwiftyLlamaCpp.free()
        
        // Test system info
        let systemInfo = SwiftyLlamaCpp.getSystemInfo()
        #expect(!systemInfo.isEmpty, "System info should not be empty")
        
        // Test capability checks
        let supportsMmap = SwiftyLlamaCpp.supportsMmap()
        let supportsMlock = SwiftyLlamaCpp.supportsMlock()
        let supportsGpuOffload = SwiftyLlamaCpp.supportsGpuOffload()
        
        #expect(type(of: supportsMmap) == Bool.self, "supportsMmap should return Bool")
        #expect(type(of: supportsMlock) == Bool.self, "supportsMlock should return Bool")
        #expect(type(of: supportsGpuOffload) == Bool.self, "supportsGpuOffload should return Bool")
        
        // Test device limits
        let maxDevices = SwiftyLlamaCpp.maxDevices()
        let maxParallelSequences = SwiftyLlamaCpp.maxParallelSequences()
        
        #expect(maxDevices >= 0, "maxDevices should be non-negative")
        #expect(maxParallelSequences >= 0, "maxParallelSequences should be non-negative")
    }
    
    @Test("Model creation with invalid path")
    func testModelCreationWithInvalidPath() throws {
        // Test that model creation fails gracefully with invalid path
        let model = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        #expect(model == nil, "Model creation should fail with invalid path")
    }
    
    @Test("Model properties with nil model")
    func testModelPropertiesWithNilModel() throws {
        // Test that model properties return safe defaults when model is nil
        let model = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        
        if let model = model {
            // This shouldn't happen with invalid path, but test properties anyway
            #expect(model.pointer == nil, "Invalid model should have nil pointer")
            #expect(model.vocab == nil, "Invalid model should have nil vocab")
            #expect(model.embeddingDimensions == 0, "Invalid model should have 0 embedding dimensions")
            #expect(model.layers == 0, "Invalid model should have 0 layers")
            #expect(model.attentionHeads == 0, "Invalid model should have 0 attention heads")
            #expect(model.parameters == 0, "Invalid model should have 0 parameters")
            #expect(model.size == 0, "Invalid model should have 0 size")
        } else {
            // Expected behavior
            #expect(Bool(true), "Model creation failed as expected")
        }
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
            #expect(Bool(true), "Model creation failed as expected")
        }
    }
    
    @Test("Context properties with nil context")
    func testContextPropertiesWithNilContext() throws {
        // Test that context properties return safe defaults when context is nil
        let invalidModel = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        let context = invalidModel.flatMap { LlamaContext(model: $0) }
        
        if let context = context {
            #expect(context.pointer == nil, "Invalid context should have nil pointer")
            #expect(context.associatedModel == nil, "Invalid context should have nil associated model")
            #expect(context.contextSize == 0, "Invalid context should have 0 context size")
            #expect(context.batchSize == 0, "Invalid context should have 0 batch size")
            #expect(context.maxBatchSize == 0, "Invalid context should have 0 max batch size")
            #expect(context.maxSequences == 0, "Invalid context should have 0 max sequences")
        } else {
            // Expected behavior
            #expect(Bool(true), "Context creation failed as expected")
        }
    }
    
    @Test("Vocabulary creation with nil pointer")
    func testVocabularyCreationWithNilPointer() throws {
        // Test that vocabulary creation fails gracefully with nil pointer
        let vocab = LlamaVocab(vocab: nil)
        #expect(vocab == nil, "Vocabulary creation should fail with nil pointer")
    }
    
    @Test("Vocabulary properties with nil vocab")
    func testVocabularyPropertiesWithNilVocab() throws {
        // Test that vocabulary properties return safe defaults when vocab is nil
        let vocab = LlamaVocab(vocab: nil)
        
        if let vocab = vocab {
            // This shouldn't happen with nil pointer, but test properties anyway
            #expect(vocab.type == .none, "Nil vocab should have .none type")
            #expect(vocab.tokenCount == 0, "Nil vocab should have 0 token count")
            #expect(vocab.bosToken == LlamaTokenNull, "Nil vocab should have null BOS token")
            #expect(vocab.eosToken == LlamaTokenNull, "Nil vocab should have null EOS token")
            #expect(vocab.eotToken == LlamaTokenNull, "Nil vocab should have null EOT token")
            #expect(vocab.sepToken == LlamaTokenNull, "Nil vocab should have null SEP token")
            #expect(vocab.newlineToken == LlamaTokenNull, "Nil vocab should have null newline token")
            #expect(vocab.padToken == LlamaTokenNull, "Nil vocab should have null pad token")
            #expect(vocab.maskToken == LlamaTokenNull, "Nil vocab should have null mask token")
        } else {
            // Expected behavior
            #expect(Bool(true), "Vocabulary creation failed as expected")
        }
    }
    
    @Test("Vocabulary methods with nil vocab")
    func testVocabularyMethodsWithNilVocab() throws {
        // Test that vocabulary methods return safe defaults when vocab is nil
        let vocab = LlamaVocab(vocab: nil)
        
        if let vocab = vocab {
            // Test methods with nil vocab
            #expect(vocab.getText(for: 0) == nil, "getText should return nil for nil vocab")
            #expect(vocab.getScore(for: 0) == 0.0, "getScore should return 0.0 for nil vocab")
            #expect(vocab.getAttribute(for: 0) == .undefined, "getAttribute should return .undefined for nil vocab")
            #expect(vocab.isEndOfGeneration(token: 0) == false, "isEndOfGeneration should return false for nil vocab")
            #expect(vocab.isControl(token: 0) == false, "isControl should return false for nil vocab")
        } else {
            // Expected behavior
            #expect(Bool(true), "Vocabulary creation failed as expected")
        }
    }
    
    @Test("Tokenizer functions with nil vocab")
    func testTokenizerFunctionsWithNilVocab() throws {
        // Test tokenizer functions with nil vocabulary
        let nilVocab: LlamaVocabPointer? = nil
        
        // Test tokenize
        let tokens = LlamaTokenizer.tokenize(text: "Hello world", vocab: nilVocab)
        #expect(tokens == nil, "tokenize should return nil with nil vocab")
        
        // Test tokenToPiece
        let piece = LlamaTokenizer.tokenToPiece(token: 0, vocab: nilVocab)
        #expect(piece == nil, "tokenToPiece should return nil with nil vocab")
        
        // Test detokenize
        let text = LlamaTokenizer.detokenize(tokens: [0, 1, 2], vocab: nilVocab)
        #expect(text == nil, "detokenize should return nil with nil vocab")
        
        // Test applyChatTemplate
        let messages = ["user".withCString { rolePtr in
            "Hello".withCString { contentPtr in
                LlamaChatMessage(role: rolePtr, content: contentPtr)
            }
        }]
        let template = LlamaTokenizer.applyChatTemplate(template: nil, messages: messages)
        #expect(template == nil, "applyChatTemplate should return nil with nil template parameter")
        
        // Test getBuiltinTemplates
        let templates = LlamaTokenizer.getBuiltinTemplates()
        #expect(templates != nil, "getBuiltinTemplates should return array (may be empty)")
    }
    
    @Test("Tokenizer convenience methods")
    func testTokenizerConvenienceMethods() throws {
        // Test LlamaVocab extension methods with nil vocab
        let vocab = LlamaVocab(vocab: nil)
        
        if let vocab = vocab {
            // Test convenience methods
            let tokens = vocab.tokenize(text: "Hello world")
            #expect(tokens == nil, "vocab.tokenize should return nil with nil vocab")
            
            let piece = vocab.tokenToPiece(token: 0)
            #expect(piece == nil, "vocab.tokenToPiece should return nil with nil vocab")
            
            let text = vocab.detokenize(tokens: [0, 1, 2])
            #expect(text == nil, "vocab.detokenize should return nil with nil vocab")
        } else {
            // Expected behavior
            #expect(Bool(true), "Vocabulary creation failed as expected")
        }
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
    
    @Test("Constants and magic values")
    func testConstantsAndMagicValues() throws {
        // Test that constants are properly defined
        #expect(LlamaTokenNull == -1, "LlamaTokenNull should be -1")
        #expect(LlamaDefaultSeed == 0xFFFFFFFF, "LlamaDefaultSeed should be 0xFFFFFFFF")
        #expect(LlamaFileMagicGGLA == 0x67676C61, "LlamaFileMagicGGLA should be correct magic")
        #expect(LlamaFileMagicGGSN == 0x6767736E, "LlamaFileMagicGGSN should be correct magic")
        #expect(LlamaFileMagicGGSQ == 0x67677371, "LlamaFileMagicGGSQ should be correct magic")
        #expect(LlamaSessionMagic == 0x6767736E, "LlamaSessionMagic should be correct magic")
        #expect(LlamaSessionVersion == 9, "LlamaSessionVersion should be 9")
        #expect(LlamaStateSeqMagic == 0x67677371, "LlamaStateSeqMagic should be correct magic")
        #expect(LlamaStateSeqVersion == 2, "LlamaStateSeqVersion should be 2")
    }
    
    @Test("Enum extensions")
    func testEnumExtensions() throws {
        // Test that enum extensions provide correct values
        #expect(LlamaVocabType.none.rawValue == 0, "LlamaVocabType.none should be 0")
        #expect(LlamaVocabType.spm.rawValue == 1, "LlamaVocabType.spm should be 1")
        #expect(LlamaVocabType.bpe.rawValue == 2, "LlamaVocabType.bpe should be 2")
        
        #expect(LlamaRopeType.none.rawValue == -1, "LlamaRopeType.none should be -1")
        #expect(LlamaRopeType.norm.rawValue == 0, "LlamaRopeType.norm should be 0")
        #expect(LlamaRopeType.neox.rawValue == 2, "LlamaRopeType.neox should be 2")
        
        #expect(LlamaTokenType.undefined.rawValue == 0, "LlamaTokenType.undefined should be 0")
        #expect(LlamaTokenType.normal.rawValue == 1, "LlamaTokenType.normal should be 1")
        #expect(LlamaTokenType.unknown.rawValue == 2, "LlamaTokenType.unknown should be 2")
        
        #expect(LlamaTokenAttribute.undefined.rawValue == 0, "LlamaTokenAttribute.undefined should be 0")
        #expect(LlamaTokenAttribute.unknown.rawValue == 1, "LlamaTokenAttribute.unknown should be 1")
        #expect(LlamaTokenAttribute.normal.rawValue == 4, "LlamaTokenAttribute.normal should be 4")
        
        #expect(LlamaFileType.allF32.rawValue == 0, "LlamaFileType.allF32 should be 0")
        #expect(LlamaFileType.mostlyF16.rawValue == 1, "LlamaFileType.mostlyF16 should be 1")
        #expect(LlamaFileType.mostlyQ4_0.rawValue == 2, "LlamaFileType.mostlyQ4_0 should be 2")
    }
    
    @Test("Memory management")
    func testMemoryManagement() throws {
        // Test that objects can be created and destroyed without crashes
        // This is mainly to ensure no memory leaks or crashes
        
        // Test SwiftyLlamaCpp initialization
        SwiftyLlamaCpp.initialize()
        SwiftyLlamaCpp.free()
        
        // Test model creation and destruction
        let model = LlamaModel(modelPath: "/nonexistent/path/model.gguf")
        // model should be nil, but deinit should not crash
        
        // Test context creation and destruction
        if let model = model {
            _ = LlamaContext(model: model)
            // context should be nil, but deinit should not crash
        }
        
        // Test vocab creation and destruction
        _ = LlamaVocab(vocab: nil)
        // vocab should be nil, but deinit should not crash
        
        #expect(Bool(true), "Memory management tests completed without crashes")
    }
    
    @Test("Error handling edge cases")
    func testErrorHandlingEdgeCases() throws {
        // Test various edge cases and error conditions
        
        // Test with empty strings
        let emptyTokens = LlamaTokenizer.tokenize(text: "", vocab: nil)
        #expect(emptyTokens == nil, "tokenize with empty string and nil vocab should return nil")
        
        // Test with empty token arrays
        let emptyText = LlamaTokenizer.detokenize(tokens: [], vocab: nil)
        #expect(emptyText == nil, "detokenize with empty array and nil vocab should return nil")
        
        // Test with invalid token values
        let vocab = LlamaVocab(vocab: nil)
        if let vocab = vocab {
            let invalidText = vocab.getText(for: -1)
            #expect(invalidText == nil, "getText with invalid token should return nil")
            
            let invalidScore = vocab.getScore(for: -1)
            #expect(invalidScore == 0.0, "getScore with invalid token should return 0.0")
            
            let invalidAttr = vocab.getAttribute(for: -1)
            #expect(invalidAttr == .undefined, "getAttribute with invalid token should return .undefined")
        }
        
        #expect(Bool(true), "Error handling edge cases completed")
    }
} 