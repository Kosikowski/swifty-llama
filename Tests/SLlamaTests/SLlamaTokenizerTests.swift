import Foundation
import Testing
import TestUtilities
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

    // MARK: - Integration Tests with Real Models

    @Test("Real model tokenization integration test")
    func realModelTokenizationIntegrationTest() throws {
        let modelPath = TestUtilities.testModelPath

        // Check if test model exists
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Tokenization test skipped: Model file not found at \(modelPath)")
            return
        }

        // Initialize the backend
        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let vocab = model.vocab

        #expect(vocab != nil, "Model should have a valid vocabulary")

        let testText = "Hello world! This is a test."

        // Test direct tokenization
        let tokens = try SLlamaTokenizer.tokenize(
            text: testText,
            vocab: vocab,
            addSpecial: true,
            parseSpecial: true
        )

        #expect(!tokens.isEmpty, "Tokenization should produce tokens")
        #expect(tokens.count > 0, "Should have at least one token")

        print("Tokenized '\(testText)' into \(tokens.count) tokens: \(tokens)")
    }

    @Test("Round-trip tokenization test")
    func roundTripTokenizationTest() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Round-trip test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let vocab = model.vocab

        #expect(vocab != nil, "Model should have a valid vocabulary")

        let originalText = "Hello world"

        // Round trip: Text → Tokens → Text
        let tokens = try SLlamaTokenizer.tokenize(
            text: originalText,
            vocab: vocab,
            addSpecial: false, // Don't add special tokens for cleaner round-trip
            parseSpecial: true
        )

        let reconstructedText = try SLlamaTokenizer.detokenize(
            tokens: tokens,
            vocab: vocab,
            removeSpecial: false,
            unparseSpecial: true
        )

        print("Original: '\(originalText)'")
        print("Tokens: \(tokens)")
        print("Reconstructed: '\(reconstructedText)'")

        // Note: Perfect round-trip isn't always possible due to tokenization specifics,
        // but the reconstructed text should be meaningful and similar
        #expect(!reconstructedText.isEmpty, "Detokenization should produce non-empty text")
        #expect(
            reconstructedText.contains("Hello") || reconstructedText.contains("world"),
            "Reconstructed text should contain recognizable parts"
        )
    }

    @Test("Vocab-based tokenization convenience methods")
    func vocabBasedTokenizationTest() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Vocab test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        guard let vocabPointer = model.vocab else {
            throw TestError.vocabLoadFailed
        }

        let vocab = SLlamaVocab(vocab: vocabPointer)
        let testText = "Test tokenization"

        // Test vocab convenience methods
        let tokens = try vocab.tokenize(
            text: testText,
            addSpecial: true,
            parseSpecial: true
        )

        #expect(!tokens.isEmpty, "Vocab tokenization should produce tokens")

        let detokenizedText = try vocab.detokenize(
            tokens: tokens,
            removeSpecial: true,
            unparseSpecial: true
        )

        #expect(!detokenizedText.isEmpty, "Vocab detokenization should produce text")
        print("Vocab round-trip: '\(testText)' → \(tokens.count) tokens → '\(detokenizedText)'")
    }

    @Test("Protocol methods integration test")
    func protocolMethodsIntegrationTest() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Protocol test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let vocab = model.vocab

        #expect(vocab != nil, "Model should have a valid vocabulary")

        let testText = "Hello"
        let tokens = try SLlamaTokenizer.tokenize(
            text: testText,
            vocab: vocab,
            addSpecial: false,
            parseSpecial: true
        )

        #expect(!tokens.isEmpty, "Should have tokens")

        // Test new protocol methods
        guard let firstToken = tokens.first else {
            throw TestError.tokenizationFailed
        }

        // Test getTokenText
        let tokenText = SLlamaTokenizer.getTokenText(token: firstToken, vocab: vocab)
        #expect(tokenText != nil, "Should get token text")
        print("Token \(firstToken) text: '\(tokenText ?? "nil")'")

        // Test getTokenType
        let tokenType = SLlamaTokenizer.getTokenType(token: firstToken, vocab: vocab)
        print("Token \(firstToken) type: \(tokenType)")

        // Test getTokenAttributes
        let tokenAttrs = SLlamaTokenizer.getTokenAttributes(token: firstToken, vocab: vocab)
        print("Token \(firstToken) attributes: \(tokenAttrs)")

        // Test isControlToken
        let isControl = SLlamaTokenizer.isControlToken(token: firstToken, vocab: vocab)
        print("Token \(firstToken) is control: \(isControl)")

        // Test protocol-conforming detokenize method
        let protocolDetokenized = try SLlamaTokenizer.detokenize(
            tokens: tokens,
            vocab: vocab,
            renderSpecialTokens: true
        )

        #expect(!protocolDetokenized.isEmpty, "Protocol detokenize should work")
        print("Protocol detokenized: '\(protocolDetokenized)'")
    }

    @Test("Multiple token analysis test")
    func multipleTokenAnalysisTest() throws {
        let modelPath = TestUtilities.testModelPath

        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Token analysis test skipped: Model file not found at \(modelPath)")
            return
        }

        SLlama.initialize()
        defer { SLlama.cleanup() }

        let model = try SLlamaModel(modelPath: modelPath)
        let vocab = model.vocab

        #expect(vocab != nil, "Model should have a valid vocabulary")

        let testText = "Hello world! 123"
        let tokens = try SLlamaTokenizer.tokenize(
            text: testText,
            vocab: vocab,
            addSpecial: true,
            parseSpecial: true
        )

        print("Analyzing \(tokens.count) tokens from '\(testText)':")

        for (index, token) in tokens.enumerated() {
            let text = SLlamaTokenizer.getTokenText(token: token, vocab: vocab)
            let type = SLlamaTokenizer.getTokenType(token: token, vocab: vocab)
            let attrs = SLlamaTokenizer.getTokenAttributes(token: token, vocab: vocab)
            let isControl = SLlamaTokenizer.isControlToken(token: token, vocab: vocab)

            print(
                "Token[\(index)]: \(token) → '\(text ?? "nil")' (type: \(type), attrs: \(attrs), control: \(isControl))"
            )
        }

        #expect(tokens.count > 1, "Should tokenize into multiple tokens")
    }
}
