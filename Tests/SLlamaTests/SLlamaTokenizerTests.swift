import Testing
@testable import SLlama

struct SLlamaTokenizerTests {
    @Test("Tokenizer with nil vocab throws error")
    func sLlamaTokenizerWithNilVocab() throws {
        // Test tokenization with nil vocab should throw
        #expect(throws: SLlamaError.self) {
            try SLlamaTokenizer.tokenize(text: "Hello world", vocab: nil)
        }
    }

    @Test("Detokenizer with nil vocab throws error")
    func sLlamaTokenizerDetokenizationWithNilVocab() throws {
        // Test detokenization with nil vocab should throw
        #expect(throws: SLlamaError.self) {
            try SLlamaTokenizer.detokenize(tokens: [1, 2, 3], vocab: nil)
        }
    }

    @Test("Chat template application")
    func sLlamaTokenizerChatTemplate() throws {
        // Test chat template with valid messages
        try "user".withCString { rolePtr in
            try "Hello".withCString { contentPtr in
                let messages = [SLlamaChatMessage(role: rolePtr, content: contentPtr)]
                let result = try SLlamaTokenizer.applyChatTemplate(template: nil, messages: messages)
                // Verify the function completes successfully
                #expect(!result.isEmpty, "Chat template should return a non-empty result")
            }
        }
    }

    @Test("Vocab convenience methods with nil vocab throw errors")
    func sLlamaVocabConvenienceMethods() throws {
        let vocab = SLlamaVocab(vocab: nil)

        // Test convenience methods should throw with nil vocab
        #expect(throws: SLlamaError.self) {
            try vocab.tokenize(text: "Hello world")
        }

        #expect(throws: SLlamaError.self) {
            try vocab.detokenize(tokens: [1, 2, 3])
        }
    }

    @Test("Token to piece with nil vocab throws error")
    func sLlamaTokenizerTokenToPiece() throws {
        // Test token to piece conversion with nil vocab should throw
        #expect(throws: SLlamaError.self) {
            try SLlamaTokenizer.tokenToPiece(token: 1, vocab: nil)
        }
    }

    @Test("Builtin templates")
    func sLlamaTokenizerBuiltinTemplates() throws {
        // Test getting builtin templates
        let templates = try SLlamaTokenizer.getBuiltinTemplates()
        #expect(templates.count >= 0, "Builtin templates should return an array (may be empty)")
    }

    @Test("Empty messages throws error")
    func sLlamaTokenizerEmptyMessages() throws {
        // Test chat template with empty messages should throw
        #expect(throws: SLlamaError.self) {
            try SLlamaTokenizer.applyChatTemplate(template: nil, messages: [])
        }
    }
}
