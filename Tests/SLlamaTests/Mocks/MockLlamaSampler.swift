import Foundation
@testable import SLlama

/// Mock implementation of PLlamaSampler for testing sampling operations
public final class MockLlamaSampler: PLlamaSampler {
    // MARK: - Configurable Properties

    public var cSampler: SLlamaSamplerPointer? { nil }
    public var name: String?

    // MARK: - Configurable Behavior

    /// Sequence of tokens to return from sample() calls
    public var tokensToReturn: [SLlamaToken] = []
    private var tokenIndex: Int = 0

    /// Default token to return when tokensToReturn is exhausted
    public var defaultToken: SLlamaToken = 0

    // MARK: - Call Tracking

    public private(set) var acceptCalls: [SLlamaToken] = []
    public private(set) var applyCalls: [SLlamaTokenDataArrayPointer] = []
    public private(set) var resetCallCount: Int = 0
    public private(set) var cloneCallCount: Int = 0
    public private(set) var sampleCalls: [SLlamaTokenDataArrayPointer] = []

    // MARK: - Initialization

    public init(name: String? = "MockSampler") {
        self.name = name
    }

    // MARK: - PLlamaSampler Methods

    public func accept(_ token: SLlamaToken) {
        acceptCalls.append(token)
    }

    public func apply(to tokenDataArray: SLlamaTokenDataArrayPointer) {
        applyCalls.append(tokenDataArray)
    }

    public func reset() {
        resetCallCount += 1
        tokenIndex = 0
    }

    public func clone() -> PLlamaSampler? {
        cloneCallCount += 1
        let cloned = MockLlamaSampler(name: name)
        cloned.tokensToReturn = tokensToReturn
        cloned.defaultToken = defaultToken
        return cloned
    }

    public func sample(_ tokenDataArray: SLlamaTokenDataArrayPointer) -> SLlamaToken {
        sampleCalls.append(tokenDataArray)

        if tokenIndex < tokensToReturn.count {
            let token = tokensToReturn[tokenIndex]
            tokenIndex += 1
            return token
        }
        return defaultToken
    }

    // MARK: - Test Helpers

    /// Configure a sequence of tokens to return
    public func setTokenSequence(_ tokens: [SLlamaToken]) {
        tokensToReturn = tokens
        tokenIndex = 0
    }

    /// Get the next token that would be returned
    public var nextToken: SLlamaToken {
        if tokenIndex < tokensToReturn.count {
            return tokensToReturn[tokenIndex]
        }
        return defaultToken
    }

    public func resetCallTracking() {
        acceptCalls.removeAll()
        applyCalls.removeAll()
        resetCallCount = 0
        cloneCallCount = 0
        sampleCalls.removeAll()
    }

    public func fullReset() {
        resetCallTracking()
        tokenIndex = 0
    }
}
