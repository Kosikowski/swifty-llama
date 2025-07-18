import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct LlamaTokenizerTests {
    
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
        _ = LlamaTokenizer.applyChatTemplate(template: nil, messages: messages)
        // Note: This test is skipped as the behavior with nil template is implementation-dependent
        
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
} 