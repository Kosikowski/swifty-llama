import Foundation
@testable import SLlama

/// Mock implementation of PLlamaCore for testing core operations
public final class MockLlamaCore: PLlamaCore {
    // MARK: - Configurable Properties

    public var coreModel: PLlamaModel?

    // Configurable return values
    public var contextSizeValue: UInt32 = 2048
    public var batchSizeValue: UInt32 = 512
    public var unifiedBatchSizeValue: UInt32 = 512
    public var maxSequencesValue: UInt32 = 1
    public var memoryValue: SLlamaMemory?
    public var poolingTypeValue: SLlamaPoolingType = .none

    // MARK: - Configurable Behavior

    /// Error to throw on encode, if any
    public var encodeError: Error?

    /// Error to throw on decode, if any
    public var decodeError: Error?

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
    public private(set) var getContextSizeCallCount: Int = 0
    public private(set) var getBatchSizeCallCount: Int = 0
    public private(set) var getUnifiedBatchSizeCallCount: Int = 0
    public private(set) var getMaxSequencesCallCount: Int = 0
    public private(set) var getMemoryCallCount: Int = 0
    public private(set) var getPoolingTypeCallCount: Int = 0

    // MARK: - Weak Reference to Context

    private weak var context: MockLlamaContext?

    // MARK: - Initialization

    public init(context: MockLlamaContext? = nil, model: PLlamaModel? = nil) {
        self.context = context
        coreModel = model ?? context?.associatedModel
    }

    // MARK: - PLlamaCore Methods

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

    public func getContextSize() -> UInt32 {
        getContextSizeCallCount += 1
        return contextSizeValue
    }

    public func getBatchSize() -> UInt32 {
        getBatchSizeCallCount += 1
        return batchSizeValue
    }

    public func getUnifiedBatchSize() -> UInt32 {
        getUnifiedBatchSizeCallCount += 1
        return unifiedBatchSizeValue
    }

    public func getMaxSequences() -> UInt32 {
        getMaxSequencesCallCount += 1
        return maxSequencesValue
    }

    public func getMemory() -> SLlamaMemory? {
        getMemoryCallCount += 1
        return memoryValue
    }

    public func getPoolingType() -> SLlamaPoolingType {
        getPoolingTypeCallCount += 1
        return poolingTypeValue
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
        getContextSizeCallCount = 0
        getBatchSizeCallCount = 0
        getUnifiedBatchSizeCallCount = 0
        getMaxSequencesCallCount = 0
        getMemoryCallCount = 0
        getPoolingTypeCallCount = 0
        encodeError = nil
        decodeError = nil
    }
}
