import Foundation
@testable import SLlama

/// Mock implementation of PLlamaBatch for testing batch operations
public final class MockLlamaBatch: PLlamaBatch {
    // MARK: - Internal Storage

    private var _tokens: [SLlamaToken] = []
    private var _positions: [SLlamaPosition] = []
    private var _sequenceIds: [[SLlamaSequenceId]] = []
    private var _logitsFlags: [Bool] = []
    private let capacity: Int32

    // MARK: - PLlamaBatch Properties

    public var cBatch: SLlamaBatchStruct {
        // Return a zeroed batch - this is a mock
        SLlamaBatchStruct()
    }

    public var tokenCount: Int32 {
        Int32(_tokens.count)
    }

    public var tokens: SLlamaTokenPointer? { nil }
    public var embeddings: SLlamaFloatPointer? { nil }
    public var positions: SLlamaPositionPointer? { nil }
    public var sequenceIdCounts: SLlamaInt32Pointer? { nil }
    public var sequenceIds: SLlamaSeqIdPointerPointer? { nil }
    public var logits: SLlamaInt8Pointer? { nil }

    // MARK: - Call Tracking

    public struct AddTokenCall {
        public let token: SLlamaToken
        public let position: SLlamaPosition
        public let sequenceIds: [SLlamaSequenceId]
        public let logits: Bool
    }

    public private(set) var addTokenCalls: [AddTokenCall] = []
    public private(set) var clearCallCount: Int = 0

    // MARK: - Test Configuration

    /// If true, addToken will throw an error when capacity is exceeded
    public var enforceCapacity: Bool = true

    // MARK: - Initialization

    public init(capacity: Int32 = 512) {
        self.capacity = capacity
    }

    // MARK: - PLlamaBatch Methods

    public func addToken(
        _ token: SLlamaToken,
        position: SLlamaPosition,
        sequenceIds: [SLlamaSequenceId],
        logits: Bool
    ) throws {
        addTokenCalls.append(AddTokenCall(
            token: token,
            position: position,
            sequenceIds: sequenceIds,
            logits: logits
        ))

        if enforceCapacity, _tokens.count >= capacity {
            throw SLlamaError.batchCapacityExceeded
        }

        _tokens.append(token)
        _positions.append(position)
        _sequenceIds.append(sequenceIds)
        _logitsFlags.append(logits)
    }

    public func clear() {
        clearCallCount += 1
        _tokens.removeAll()
        _positions.removeAll()
        _sequenceIds.removeAll()
        _logitsFlags.removeAll()
    }

    public static func getSingleTokenBatch(_ token: SLlamaToken) -> PLlamaBatch {
        let batch = MockLlamaBatch(capacity: 1)
        try? batch.addToken(token, position: 0, sequenceIds: [0], logits: true)
        return batch
    }

    // MARK: - Test Helpers

    /// Get all tokens that have been added
    public var addedTokens: [SLlamaToken] {
        _tokens
    }

    /// Get all positions that have been added
    public var addedPositions: [SLlamaPosition] {
        _positions
    }

    /// Get all sequence IDs that have been added
    public var addedSequenceIds: [[SLlamaSequenceId]] {
        _sequenceIds
    }

    /// Get all logits flags that have been added
    public var addedLogitsFlags: [Bool] {
        _logitsFlags
    }

    public func reset() {
        addTokenCalls.removeAll()
        clearCallCount = 0
        clear()
    }
}
