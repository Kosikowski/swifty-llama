
import Foundation

// Conversation management
public struct Conversation: Codable {
    public let id: ConversationID
    public var messages: [ConversationMessage]
    public var totalTokens: Int32
    public let createdAt: Date
    public var title: String?

    public init(
        id: ConversationID,
        messages: [ConversationMessage],
        totalTokens: Int32,
        createdAt: Date,
        title: String? = nil
    ) {
        self.id = id
        self.messages = messages
        self.totalTokens = totalTokens
        self.createdAt = createdAt
        self.title = title
    }
}

// MARK: - String Extensions for Title Sanitization

public extension String {
    /// Sanitizes a string for use as a conversation title
    /// - Parameters:
    ///   - maxLength: Maximum allowed length (default: 200)
    ///   - removeQuotes: Whether to remove surrounding quotes (default: true)
    ///   - removePunctuation: Whether to remove trailing punctuation (default: true)
    ///   - minLength: Minimum length after cleaning (default: 2)
    /// - Returns: Sanitized string, or empty string if too short after cleaning
    func sanitizeForTitle(
        maxLength: Int = 200,
        removeQuotes: Bool = true,
        removePunctuation: Bool = true,
        minLength: Int = 2
    )
        -> String
    {
        var cleaned = trimmingCharacters(in: .whitespacesAndNewlines)

        // Remove quotes if present and enabled
        if removeQuotes && cleaned.hasPrefix("\"") && cleaned.hasSuffix("\"") {
            cleaned = String(cleaned.dropFirst().dropLast())
        }

        // Remove trailing punctuation if enabled
        if removePunctuation {
            cleaned = cleaned.trimmingCharacters(in: .punctuationCharacters)
        }

        // If the result is too short or just punctuation/whitespace, return empty
        if cleaned.count < minLength || cleaned.allSatisfy({ $0.isPunctuation || $0.isWhitespace }) {
            return ""
        }

        // Ensure it doesn't exceed max length
        if cleaned.count > maxLength {
            cleaned = String(cleaned.prefix(maxLength))
        }

        return cleaned
    }
}
