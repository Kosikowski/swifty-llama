import Foundation
@testable import SLlama

/// Mock implementation of PLlamaModel for testing without loading actual models
public final class MockLlamaModel: PLlamaModel {
    // MARK: - Configurable Properties

    public var pointer: SLlamaModelPointer? { nil }
    public var vocab: SLlamaVocabPointer? { nil }
    public var embeddingDimensions: Int32
    public var layers: Int32
    public var attentionHeads: Int32
    public var kvAttentionHeads: Int32
    public var parameters: UInt64
    public var size: UInt64
    public var trainingContextLength: Int32
    public var ropeType: SLlamaRopeType
    public var ropeFreqScaleTrain: Float
    public var hasEncoder: Bool
    public var hasDecoder: Bool

    // MARK: - Configurable Method Returns

    public var metadataValues: [String: String] = [:]
    public var hasEmbeddingsResult: Bool = false
    public var descriptionResult: String?

    // MARK: - Call Tracking

    public private(set) var getMetadataCalls: [(key: String, bufferSize: Int)] = []
    public private(set) var hasEmbeddingsCallCount: Int = 0
    public private(set) var getDescriptionCalls: [Int] = []

    // MARK: - Initialization

    public init(
        embeddingDimensions: Int32 = 768,
        layers: Int32 = 12,
        attentionHeads: Int32 = 12,
        kvAttentionHeads: Int32 = 12,
        parameters: UInt64 = 125_000_000,
        size: UInt64 = 250_000_000,
        trainingContextLength: Int32 = 2048,
        ropeType: SLlamaRopeType = .none,
        ropeFreqScaleTrain: Float = 1.0,
        hasEncoder: Bool = false,
        hasDecoder: Bool = true
    ) {
        self.embeddingDimensions = embeddingDimensions
        self.layers = layers
        self.attentionHeads = attentionHeads
        self.kvAttentionHeads = kvAttentionHeads
        self.parameters = parameters
        self.size = size
        self.trainingContextLength = trainingContextLength
        self.ropeType = ropeType
        self.ropeFreqScaleTrain = ropeFreqScaleTrain
        self.hasEncoder = hasEncoder
        self.hasDecoder = hasDecoder
    }

    // MARK: - PLlamaModel Methods

    public func getMetadata(key: String, bufferSize: Int) throws -> String {
        getMetadataCalls.append((key: key, bufferSize: bufferSize))

        if let value = metadataValues[key] {
            return value
        }
        throw SLlamaError.keyNotFound(key)
    }

    public func hasEmbeddings() -> Bool {
        hasEmbeddingsCallCount += 1
        return hasEmbeddingsResult
    }

    public func getDescription(bufferSize: Int) -> String? {
        getDescriptionCalls.append(bufferSize)
        return descriptionResult
    }

    // MARK: - Test Helpers

    public func reset() {
        getMetadataCalls.removeAll()
        hasEmbeddingsCallCount = 0
        getDescriptionCalls.removeAll()
    }
}
