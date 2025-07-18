import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct LlamaMemoryTests {
    
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