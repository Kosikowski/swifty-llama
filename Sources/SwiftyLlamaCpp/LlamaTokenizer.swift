import Foundation
import llama

// MARK: - LlamaTokenizer

/// A wrapper for llama tokenization functions
public class LlamaTokenizer {
    /// Tokenize text into tokens
    /// - Parameters:
    ///   - text: The text to tokenize
    ///   - addSpecial: Whether to add special tokens (BOS/EOS) if model is configured to do so
    ///   - parseSpecial: Whether to parse special and control tokens
    /// - Returns: Array of tokens, or nil if tokenization failed
    public static func tokenize(
        text: String,
        vocab: LlamaVocabPointer?,
        addSpecial: Bool = true,
        parseSpecial: Bool = true
    )
        -> [LlamaToken]?
    {
        guard let vocab else { return nil }

        let textData = text.data(using: .utf8)!
        let textLength = Int32(textData.count)

        // Allocate buffer for tokens (worst case: each character could be a token)
        let maxTokens = textLength + 10 // Add some buffer for special tokens
        var tokens = [LlamaToken](repeating: 0, count: Int(maxTokens))

        let result = llama_tokenize(
            vocab,
            text,
            textLength,
            &tokens,
            maxTokens,
            addSpecial,
            parseSpecial
        )

        if result < 0 {
            return nil
        }

        return Array(tokens.prefix(Int(result)))
    }

    /// Convert a single token to its text representation
    /// - Parameters:
    ///   - token: The token to convert
    ///   - vocab: The vocabulary to use
    ///   - lstrip: Number of leading spaces to skip
    ///   - special: Whether to render special tokens
    /// - Returns: The text representation of the token, or nil if conversion failed
    public static func tokenToPiece(
        token: LlamaToken,
        vocab: LlamaVocabPointer?,
        lstrip: Int32 = 0,
        special: Bool = false
    )
        -> String?
    {
        guard let vocab else { return nil }

        // Allocate buffer for the piece (worst case: token could be multiple characters)
        let maxLength = 256
        var buffer = [CChar](repeating: 0, count: maxLength)

        let result = llama_token_to_piece(
            vocab,
            token,
            &buffer,
            Int32(maxLength),
            lstrip,
            special
        )

        if result < 0 {
            return nil
        }

        // Convert C string to Swift string, handling null termination
        let bytes = Array(buffer.prefix(Int(result))).map { UInt8(bitPattern: $0) }
        return String(bytes: bytes, encoding: .utf8)
    }

    /// Convert tokens back to text
    /// - Parameters:
    ///   - tokens: Array of tokens to convert
    ///   - vocab: The vocabulary to use
    ///   - removeSpecial: Whether to remove special tokens (BOS/EOS) if model is configured to do so
    ///   - unparseSpecial: Whether to render special tokens in output
    /// - Returns: The text representation, or nil if conversion failed
    public static func detokenize(
        tokens: [LlamaToken],
        vocab: LlamaVocabPointer?,
        removeSpecial: Bool = true,
        unparseSpecial: Bool = true
    )
        -> String?
    {
        guard let vocab else { return nil }

        // Allocate buffer for the text (worst case: each token could be multiple characters)
        let maxLength = tokens.count * 10 // Estimate: each token averages 10 characters
        var buffer = [CChar](repeating: 0, count: maxLength)

        let result = llama_detokenize(
            vocab,
            tokens,
            Int32(tokens.count),
            &buffer,
            Int32(maxLength),
            removeSpecial,
            unparseSpecial
        )

        if result < 0 {
            return nil
        }

        // Convert C string to Swift string, handling null termination
        let bytes = Array(buffer.prefix(Int(result))).map { UInt8(bitPattern: $0) }
        return String(bytes: bytes, encoding: .utf8)
    }

    /// Apply chat template to messages
    /// - Parameters:
    ///   - template: The template to use (nil for model default)
    ///   - messages: Array of chat messages
    ///   - addAssistant: Whether to end with assistant message tokens
    /// - Returns: The formatted prompt, or nil if template application failed
    public static func applyChatTemplate(
        template: String?,
        messages: [LlamaChatMessage],
        addAssistant: Bool = true
    )
        -> String?
    {
        // Use a fixed buffer size for now, as we can't access message content easily
        let maxLength = 4096
        var buffer = [CChar](repeating: 0, count: maxLength)

        let result = llama_chat_apply_template(
            template,
            messages,
            messages.count,
            addAssistant,
            &buffer,
            Int32(maxLength)
        )

        if result < 0 {
            return nil
        }

        // Convert C string to Swift string, handling null termination
        let bytes = Array(buffer.prefix(Int(result))).map { UInt8(bitPattern: $0) }
        return String(bytes: bytes, encoding: .utf8)
    }

    /// Get list of built-in chat templates
    /// - Returns: Array of template names, or nil if failed
    public static func getBuiltinTemplates() -> [String]? {
        let maxTemplates = 100
        var templates = [UnsafePointer<CChar>?](repeating: nil, count: maxTemplates)

        let result = llama_chat_builtin_templates(&templates, maxTemplates)

        if result < 0 {
            return nil
        }

        var templateNames: [String] = []
        for i in 0 ..< Int(result) {
            if let template = templates[i] {
                templateNames.append(String(cString: template))
            }
        }

        return templateNames
    }
}

/// Extension to LlamaVocab for tokenization convenience
public extension LlamaVocab {
    /// Tokenize text using this vocabulary
    /// - Parameters:
    ///   - text: The text to tokenize
    ///   - addSpecial: Whether to add special tokens
    ///   - parseSpecial: Whether to parse special tokens
    /// - Returns: Array of tokens, or nil if tokenization failed
    func tokenize(
        text: String,
        addSpecial: Bool = true,
        parseSpecial: Bool = true
    )
        -> [LlamaToken]?
    {
        LlamaTokenizer.tokenize(
            text: text,
            vocab: vocab,
            addSpecial: addSpecial,
            parseSpecial: parseSpecial
        )
    }

    /// Convert a token to its text representation
    /// - Parameters:
    ///   - token: The token to convert
    ///   - lstrip: Number of leading spaces to skip
    ///   - special: Whether to render special tokens
    /// - Returns: The text representation, or nil if conversion failed
    func tokenToPiece(
        token: LlamaToken,
        lstrip: Int32 = 0,
        special: Bool = false
    )
        -> String?
    {
        LlamaTokenizer.tokenToPiece(
            token: token,
            vocab: vocab,
            lstrip: lstrip,
            special: special
        )
    }

    /// Convert tokens back to text
    /// - Parameters:
    ///   - tokens: Array of tokens to convert
    ///   - removeSpecial: Whether to remove special tokens
    ///   - unparseSpecial: Whether to render special tokens
    /// - Returns: The text representation, or nil if conversion failed
    func detokenize(
        tokens: [LlamaToken],
        removeSpecial: Bool = true,
        unparseSpecial: Bool = true
    )
        -> String?
    {
        LlamaTokenizer.detokenize(
            tokens: tokens,
            vocab: vocab,
            removeSpecial: removeSpecial,
            unparseSpecial: unparseSpecial
        )
    }
}
