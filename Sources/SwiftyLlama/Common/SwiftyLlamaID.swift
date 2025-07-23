import Foundation

/// A unique identifier for a conversation
public struct SwiftyLlamaID: Hashable, Sendable, Codable {
    private let id: UUID

    public init(id: UUID = UUID()) {
        self.id = id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    public static func == (lhs: SwiftyLlamaID, rhs: SwiftyLlamaID) -> Bool {
        lhs.id == rhs.id
    }
}
