import Foundation

/// Information about a generation
public struct GenerationInfo {
    public let id: SwiftyLlamaID
    public let conversationId: SwiftyLlamaID?
    public let params: GenerationParams
    public let startTime: Date
    public let isActive: Bool
}
