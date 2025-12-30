import Foundation
@testable import SLlama

/// Mock implementation of PLlamaVocab for testing tokenization without real vocabulary
public final class MockLlamaVocab: PLlamaVocab {
    // MARK: - Configurable Properties

    public var pointer: SLlamaVocabPointer? { nil }
    public var tokenCount: Int32
    public var type: SLlamaVocabType

    // Special tokens
    public var bosToken: SLlamaToken
    public var eosToken: SLlamaToken
    public var eotToken: SLlamaToken
    public var sepToken: SLlamaToken
    public var nlToken: SLlamaToken
    public var padToken: SLlamaToken
    public var maskToken: SLlamaToken
    public var clsToken: SLlamaToken
    public var unknownToken: SLlamaToken

    // MARK: - Configurable Token Data

    /// Maps tokens to their text representation
    public var tokenTexts: [SLlamaToken: String] = [:]

    /// Maps tokens to their types
    public var tokenTypes: [SLlamaToken: SLlamaTokenType] = [:]

    /// Maps tokens to their attributes
    public var tokenAttributes: [SLlamaToken: SLlamaTokenAttribute] = [:]

    /// Maps tokens to their scores
    public var tokenScores: [SLlamaToken: Float] = [:]

    /// Set of tokens that are end-of-generation
    public var eogTokens: Set<SLlamaToken> = []

    /// Set of control tokens
    public var controlTokens: Set<SLlamaToken> = []

    // MARK: - Call Tracking

    public private(set) var isEogCalls: [SLlamaToken] = []
    public private(set) var isControlCalls: [SLlamaToken] = []
    public private(set) var getTokenTextCalls: [SLlamaToken] = []
    public private(set) var getTokenTypeCalls: [SLlamaToken] = []
    public private(set) var getTokenAttributesCalls: [SLlamaToken] = []
    public private(set) var getTokenScoreCalls: [SLlamaToken] = []

    // MARK: - Initialization

    public init(
        tokenCount: Int32 = 32000,
        type: SLlamaVocabType = .bpe,
        bosToken: SLlamaToken = 1,
        eosToken: SLlamaToken = 2,
        eotToken: SLlamaToken = 2,
        sepToken: SLlamaToken = -1,
        nlToken: SLlamaToken = 13,
        padToken: SLlamaToken = 0,
        maskToken: SLlamaToken = -1,
        clsToken: SLlamaToken = -1,
        unknownToken: SLlamaToken = 0
    ) {
        self.tokenCount = tokenCount
        self.type = type
        self.bosToken = bosToken
        self.eosToken = eosToken
        self.eotToken = eotToken
        self.sepToken = sepToken
        self.nlToken = nlToken
        self.padToken = padToken
        self.maskToken = maskToken
        self.clsToken = clsToken
        self.unknownToken = unknownToken

        // Set up default EOG tokens
        eogTokens = [eosToken, eotToken]
    }

    // MARK: - PLlamaVocab Methods

    public func isEog(_ token: SLlamaToken) -> Bool {
        isEogCalls.append(token)
        return eogTokens.contains(token)
    }

    public func isControl(_ token: SLlamaToken) -> Bool {
        isControlCalls.append(token)
        return controlTokens.contains(token)
    }

    public func getTokenText(_ token: SLlamaToken) -> String? {
        getTokenTextCalls.append(token)
        return tokenTexts[token]
    }

    public func getTokenType(_ token: SLlamaToken) -> SLlamaTokenType {
        getTokenTypeCalls.append(token)
        return tokenTypes[token] ?? .normal
    }

    public func getTokenAttributes(_ token: SLlamaToken) -> SLlamaTokenAttribute {
        getTokenAttributesCalls.append(token)
        return tokenAttributes[token] ?? SLlamaTokenAttribute(rawValue: 0)
    }

    public func getTokenScore(_ token: SLlamaToken) -> Float {
        getTokenScoreCalls.append(token)
        return tokenScores[token] ?? 0.0
    }

    // MARK: - Test Helpers

    /// Add a simple vocabulary mapping
    public func addToken(_ token: SLlamaToken, text: String, score: Float = 0.0) {
        tokenTexts[token] = text
        tokenScores[token] = score
    }

    /// Set up a simple vocabulary with common tokens
    public func setupBasicVocabulary() {
        // Common tokens
        addToken(0, text: "<pad>")
        addToken(1, text: "<s>") // BOS
        addToken(2, text: "</s>") // EOS
        addToken(13, text: "\n")

        // Some word tokens
        addToken(100, text: "Hello")
        addToken(101, text: " world")
        addToken(102, text: "!")
        addToken(103, text: " ")
        addToken(104, text: "The")
        addToken(105, text: " quick")
        addToken(106, text: " brown")
        addToken(107, text: " fox")

        controlTokens = [0, 1, 2]
    }

    public func reset() {
        isEogCalls.removeAll()
        isControlCalls.removeAll()
        getTokenTextCalls.removeAll()
        getTokenTypeCalls.removeAll()
        getTokenAttributesCalls.removeAll()
        getTokenScoreCalls.removeAll()
    }
}
