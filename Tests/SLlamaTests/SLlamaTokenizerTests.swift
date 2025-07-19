import Testing
@testable import SLlama

struct SLlamaTokenizerTests {
    @Test("SLlamaTokenizer tokenization with nil vocab")
    func sLlamaTokenizerWithNilVocab() throws {
        // Test tokenization with nil vocab
        let tokens = SLlamaTokenizer.tokenize(text: "Hello world", vocab: nil)
        #expect(tokens == nil, "Tokenization with nil vocab should return nil")
    }

    @Test("SLlamaTokenizer detokenization with nil vocab")
    func sLlamaTokenizerDetokenizationWithNilVocab() throws {
        // Test detokenization with nil vocab
        let text = SLlamaTokenizer.detokenize(tokens: [1, 2, 3], vocab: nil)
        #expect(text == nil, "Detokenization with nil vocab should return nil")
    }

    @Test("SLlamaTokenizer chat template with nil vocab")
    func sLlamaTokenizerChatTemplateWithNilVocab() throws {
        // Test chat template application with nil vocab
        "user".withCString { rolePtr in
            "Hello".withCString { contentPtr in
                let messages = [SLlamaChatMessage(role: rolePtr, content: contentPtr)]
                let result = SLlamaTokenizer.applyChatTemplate(template: nil, messages: messages)
                // Just verify the function doesn't crash - the result may be non-nil and non-empty
                #expect(result != nil, "Chat template should return a result (may be non-empty)")
            }
        }
    }

    @Test("SLlamaVocab convenience methods with nil vocab")
    func sLlamaVocabConvenienceMethodsWithNilVocab() throws {
        let vocab = SLlamaVocab(vocab: nil)

        // Test convenience methods
        let tokens = vocab.tokenize(text: "Hello world")
        #expect(tokens == nil, "Tokenization with nil vocab should return nil")

        let text = vocab.detokenize(tokens: [1, 2, 3])
        #expect(text == nil, "Detokenization with nil vocab should return nil")
    }

    @Test("SLlamaTokenizer token to piece conversion")
    func sLlamaTokenizerTokenToPiece() throws {
        // Test token to piece conversion with nil vocab
        let piece = SLlamaTokenizer.tokenToPiece(token: 1, vocab: nil)
        #expect(piece == nil, "Token to piece conversion with nil vocab should return nil")
    }

    @Test("SLlamaTokenizer builtin templates")
    func sLlamaTokenizerBuiltinTemplates() throws {
        // Test getting builtin templates
        let templates = SLlamaTokenizer.getBuiltinTemplates()
        #expect(templates != nil, "Builtin templates should be available")
    }
}
