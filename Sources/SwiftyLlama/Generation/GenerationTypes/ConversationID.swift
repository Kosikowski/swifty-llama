import Foundation

/// A unique identifier for a generation
public struct GenerationID: Hashable, Sendable {
    private let id = UUID()
}

/// A unique identifier for a conversation
public struct ConversationID: Hashable, Sendable, Codable {
    private let id: UUID

    public init(id: UUID = UUID()) {
        self.id = id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    public static func == (lhs: ConversationID, rhs: ConversationID) -> Bool {
        lhs.id == rhs.id
    }
}
