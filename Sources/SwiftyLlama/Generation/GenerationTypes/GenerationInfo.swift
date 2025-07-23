import Foundation

/// Information about a generation
public struct GenerationInfo {
    public let id: GenerationID
    public let conversationId: ConversationID?
    public let params: GenerationParams
    public let startTime: Date
    public let isActive: Bool
}
