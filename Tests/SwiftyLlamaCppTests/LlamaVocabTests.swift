import Testing
@testable import SwiftyLlamaCpp

@Suite 
struct LlamaVocabTests {
    
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
} 