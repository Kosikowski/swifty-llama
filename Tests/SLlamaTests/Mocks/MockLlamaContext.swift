import Foundation
@testable import SLlama

/// Mock implementation of PLlamaContext for testing without real model context
public final class MockLlamaContext: PLlamaContext {
    // MARK: - Configurable Properties

    public var pointer: SLlamaContextPointer? { nil }
    public var associatedModel: PLlamaModel?
    public var contextSize: UInt32
    public var batchSize: UInt32
    public var maxBatchSize: UInt32
    public var maxSequences: UInt32
    public var contextModel: PLlamaModel? { associatedModel }
    public var contextMemory: SLlamaMemory? { nil }
    public var poolingType: SLlamaPoolingType

    // MARK: - Configurable Behavior

    /// Error to throw on encode, if any
    public var encodeError: Error?

    /// Error to throw on decode, if any
    public var decodeError: Error?

    /// The mock core instance to return
    public var mockCore: MockLlamaCore?

    // MARK: - Call Tracking

    public struct EncodeCall {
        public let batch: PLlamaBatch
    }

    public struct DecodeCall {
        public let batch: PLlamaBatch
    }

    public struct SetThreadsCall {
        public let nThreads: Int32
        public let nThreadsBatch: Int32
    }

    public private(set) var encodeCalls: [EncodeCall] = []
    public private(set) var decodeCalls: [DecodeCall] = []
    public private(set) var setThreadsCalls: [SetThreadsCall] = []
    public private(set) var setEmbeddingsCalls: [Bool] = []
    public private(set) var setCausalAttentionCalls: [Bool] = []
    public private(set) var setWarmupCalls: [Bool] = []
    public private(set) var synchronizeCallCount: Int = 0
    public private(set) var coreCallCount: Int = 0

    // MARK: - Initialization

    public init(
        associatedModel: PLlamaModel? = nil,
        contextSize: UInt32 = 2048,
        batchSize: UInt32 = 512,
        maxBatchSize: UInt32 = 512,
        maxSequences: UInt32 = 1,
        poolingType: SLlamaPoolingType = .none
    ) {
        self.associatedModel = associatedModel
        self.contextSize = contextSize
        self.batchSize = batchSize
        self.maxBatchSize = maxBatchSize
        self.maxSequences = maxSequences
        self.poolingType = poolingType
    }

    // MARK: - PLlamaContext Methods

    public func core() -> PLlamaCore {
        coreCallCount += 1
        if let mockCore {
            return mockCore
        }
        return MockLlamaCore(context: self)
    }

    public func encode(_ batch: PLlamaBatch) throws {
        encodeCalls.append(EncodeCall(batch: batch))
        if let error = encodeError {
            throw error
        }
    }

    public func decode(_ batch: PLlamaBatch) throws {
        decodeCalls.append(DecodeCall(batch: batch))
        if let error = decodeError {
            throw error
        }
    }

    public func setThreads(nThreads: Int32, nThreadsBatch: Int32) {
        setThreadsCalls.append(SetThreadsCall(nThreads: nThreads, nThreadsBatch: nThreadsBatch))
    }

    public func setEmbeddings(_ embeddings: Bool) {
        setEmbeddingsCalls.append(embeddings)
    }

    public func setCausalAttention(_ causalAttn: Bool) {
        setCausalAttentionCalls.append(causalAttn)
    }

    public func setWarmup(_ warmup: Bool) {
        setWarmupCalls.append(warmup)
    }

    public func synchronize() {
        synchronizeCallCount += 1
    }

    // MARK: - Async Methods

    public func encodeAsync(_ batch: PLlamaBatch) async throws {
        try encode(batch)
    }

    public func decodeAsync(_ batch: PLlamaBatch) async throws {
        try decode(batch)
    }

    // MARK: - Test Helpers

    public func reset() {
        encodeCalls.removeAll()
        decodeCalls.removeAll()
        setThreadsCalls.removeAll()
        setEmbeddingsCalls.removeAll()
        setCausalAttentionCalls.removeAll()
        setWarmupCalls.removeAll()
        synchronizeCallCount = 0
        coreCallCount = 0
        encodeError = nil
        decodeError = nil
    }
}
