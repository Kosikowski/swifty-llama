import Foundation
import llama

// MARK: - Tokenizer Configuration

/// Configuration for SLlamaTokenizer buffer sizes and limits
public struct SLlamaTokenizerConfig: Sendable {
    /// Maximum buffer size for token-to-piece conversion
    public let maxTokenPieceLength: Int32
    /// Maximum buffer size for detokenization (multiplier for token count)
    public let detokenizeBufferMultiplier: Int32
    /// Maximum buffer size for chat template application
    public let chatTemplateBufferSize: Int32
    /// Maximum number of built-in templates to retrieve
    public let maxBuiltinTemplates: Int32
    
    public init(
        maxTokenPieceLength: Int32 = 256,
        detokenizeBufferMultiplier: Int32 = 10,
        chatTemplateBufferSize: Int32 = 4096,
        maxBuiltinTemplates: Int32 = 100
    ) {
        self.maxTokenPieceLength = maxTokenPieceLength
        self.detokenizeBufferMultiplier = detokenizeBufferMultiplier
        self.chatTemplateBufferSize = chatTemplateBufferSize
        self.maxBuiltinTemplates = maxBuiltinTemplates
    }
}

// MARK: - SLlamaTokenizer

/// A wrapper for llama tokenization functions
public class SLlamaTokenizer: @unchecked Sendable, PLlamaTokenizer {
    /// Tokenize text into tokens
    /// - Parameters:
    ///   - text: The text to tokenize
    ///   - vocab: The vocabulary to use
    ///   - addSpecial: Whether to add special tokens (BOS/EOS) if model is configured to do so
    ///   - parseSpecial: Whether to parse special and control tokens
    /// - Returns: Array of tokens
    /// - Throws: SLlamaError if tokenization fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func tokenize(
        text: String,
        vocab: SLlamaVocabPointer?,
        addSpecial: Bool = true,
        parseSpecial: Bool = true
    ) throws
        -> [SLlamaToken]
    {
        guard let vocab else {
            throw SLlamaError.invalidVocabulary
        }

        guard !text.isEmpty else {
            return [] // Empty text produces empty token array
        }

        guard let textData = text.data(using: .utf8) else {
            throw SLlamaError.encodingFailure
        }

        let textLength = Int32(textData.count)

        // Check for reasonable text length limits
        if textLength > 1_000_000 { // 1MB limit
            throw SLlamaError.textTooLong
        }

        // Allocate buffer for tokens (worst case: each character could be a token)
        let maxTokens = textLength + 10 // Add some buffer for special tokens
        var tokens = [SLlamaToken](repeating: 0, count: Int(maxTokens))

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
            switch result {
                case -1:
                    throw SLlamaError.invalidVocabulary
                case -2:
                    throw SLlamaError.bufferTooSmall
                case -3:
                    throw SLlamaError.encodingFailure
                default:
                    throw SLlamaError
                        .tokenizationFailed(
                            "Tokenization failed for text of length \(textLength) with error code: \(result)"
                        )
            }
        }

        return Array(tokens.prefix(Int(result)))
    }

    /// Convert a single token to its text representation
    /// - Parameters:
    ///   - token: The token to convert
    ///   - vocab: The vocabulary to use
    ///   - lstrip: Number of leading spaces to skip
    ///   - special: Whether to render special tokens
    ///   - config: Tokenizer configuration (optional, uses defaults if nil)
    /// - Returns: The text representation of the token
    /// - Throws: SLlamaError if conversion fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func tokenToPiece(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?,
        lstrip: Int32 = 0,
        special: Bool = false,
        config: SLlamaTokenizerConfig? = nil
    ) throws
        -> String
    {
        guard let vocab else {
            throw SLlamaError.invalidVocabulary
        }

        let tokenizerConfig = config ?? SLlamaTokenizerConfig()
        
        // Allocate buffer for the piece (worst case: token could be multiple characters)
        let maxLength = tokenizerConfig.maxTokenPieceLength
        var buffer = [CChar](repeating: 0, count: Int(maxLength))

        let result = llama_token_to_piece(
            vocab,
            token,
            &buffer,
            Int32(maxLength),
            lstrip,
            special
        )

        if result < 0 {
            switch result {
                case -1:
                    throw SLlamaError.invalidToken(token)
                case -2:
                    throw SLlamaError.bufferTooSmall
                default:
                    throw SLlamaError
                        .detokenizationFailed("Token to piece failed for token \(token) with error code: \(result)")
            }
        }

        // Convert C string to Swift string, handling null termination
        let bytes = Array(buffer.prefix(Int(result))).map { UInt8(bitPattern: $0) }
        guard let string = String(bytes: bytes, encoding: .utf8) else {
            throw SLlamaError.encodingFailure
        }

        return string
    }

    /// Convert tokens back to text
    /// - Parameters:
    ///   - tokens: Array of tokens to convert
    ///   - vocab: The vocabulary to use
    ///   - removeSpecial: Whether to remove special tokens (BOS/EOS) if model is configured to do so
    ///   - unparseSpecial: Whether to render special tokens in output
    ///   - config: Tokenizer configuration (optional, uses defaults if nil)
    /// - Returns: The text representation
    /// - Throws: SLlamaError if conversion fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func detokenize(
        tokens: [SLlamaToken],
        vocab: SLlamaVocabPointer?,
        removeSpecial: Bool = true,
        unparseSpecial: Bool = true,
        config: SLlamaTokenizerConfig? = nil
    ) throws
        -> String
    {
        guard let vocab else {
            throw SLlamaError.invalidVocabulary
        }

        guard !tokens.isEmpty else {
            return "" // Empty token array produces empty string
        }

        let tokenizerConfig = config ?? SLlamaTokenizerConfig()
        
        // Allocate buffer for the text (worst case: each token could be multiple characters)
        let maxLength = tokens.count * Int(tokenizerConfig.detokenizeBufferMultiplier) // Estimate: each token averages N characters
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
            switch result {
                case -1:
                    throw SLlamaError.invalidVocabulary
                case -2:
                    throw SLlamaError.bufferTooSmall
                case -3:
                    throw SLlamaError.encodingFailure
                default:
                    throw SLlamaError
                        .detokenizationFailed(
                            "Detokenization failed for \(tokens.count) tokens with error code: \(result)"
                        )
            }
        }

        // Convert C string to Swift string, handling null termination
        let bytes = Array(buffer.prefix(Int(result))).map { UInt8(bitPattern: $0) }
        guard let string = String(bytes: bytes, encoding: .utf8) else {
            throw SLlamaError.encodingFailure
        }

        return string
    }

    // MARK: - PLlamaTokenizer Protocol Methods

    /// Detokenize tokens into text (PLlamaTokenizer protocol method)
    /// - Parameters:
    ///   - tokens: Array of tokens to convert
    ///   - vocab: The vocabulary to use
    ///   - renderSpecialTokens: Whether to render special tokens in output
    /// - Returns: The text representation
    /// - Throws: SLlamaError if conversion fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func detokenize(
        tokens: [SLlamaToken],
        vocab: SLlamaVocabPointer?,
        renderSpecialTokens: Bool
    ) throws
        -> String
    {
        // Delegate to the existing implementation
        try detokenize(
            tokens: tokens,
            vocab: vocab,
            removeSpecial: !renderSpecialTokens,
            unparseSpecial: renderSpecialTokens
        )
    }

    /// Get token text representation
    /// - Parameters:
    ///   - token: The token to get text for
    ///   - vocab: The vocabulary to use
    /// - Returns: The text representation of the token, or nil if invalid
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getTokenText(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?
    )
        -> String?
    {
        guard let vocab else { return nil }

        // Allocate buffer for token text
        let maxLength: Int32 = 256 // Maximum token length
        var buffer = [CChar](repeating: 0, count: Int(maxLength))

        let result = llama_token_to_piece(vocab, token, &buffer, maxLength, 0, false)

        guard result > 0 else { return nil }

        // Convert C string to Swift string
        let bytes = Array(buffer.prefix(Int(result))).map { UInt8(bitPattern: $0) }
        return String(bytes: bytes, encoding: .utf8)
    }

    /// Get token type
    /// - Parameters:
    ///   - token: The token to get type for
    ///   - vocab: The vocabulary to use
    /// - Returns: The type of the token
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getTokenType(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?
    )
        -> SLlamaTokenType
    {
        guard let vocab else { return LLAMA_TOKEN_TYPE_UNDEFINED }

        // For now, determine type based on token attributes
        let attr = llama_vocab_get_attr(vocab, token)

        // Map attributes to types (this is a simplified mapping)
        if (attr.rawValue & LLAMA_TOKEN_ATTR_CONTROL.rawValue) != 0 {
            return LLAMA_TOKEN_TYPE_CONTROL
        } else if (attr.rawValue & LLAMA_TOKEN_ATTR_UNKNOWN.rawValue) != 0 {
            return LLAMA_TOKEN_TYPE_UNDEFINED
        } else {
            return LLAMA_TOKEN_TYPE_NORMAL
        }
    }

    /// Get token attributes
    /// - Parameters:
    ///   - token: The token to get attributes for
    ///   - vocab: The vocabulary to use
    /// - Returns: The attributes of the token
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getTokenAttributes(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?
    )
        -> SLlamaTokenAttribute
    {
        guard let vocab else { return LLAMA_TOKEN_ATTR_UNDEFINED }
        return llama_vocab_get_attr(vocab, token)
    }

    /// Check if token is a control token
    /// - Parameters:
    ///   - token: The token to check
    ///   - vocab: The vocabulary to use
    /// - Returns: True if the token is a control token
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func isControlToken(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?
    )
        -> Bool
    {
        guard let vocab else { return false }
        return llama_vocab_is_control(vocab, token)
    }

    /// Apply chat template to messages
    /// - Parameters:
    ///   - template: The template to use (nil for model default)
    ///   - messages: Array of chat messages
    ///   - addAssistant: Whether to end with assistant message tokens
    ///   - config: Tokenizer configuration (optional, uses defaults if nil)
    /// - Returns: The formatted prompt
    /// - Throws: SLlamaError if template application fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func applyChatTemplate(
        template: String?,
        messages: [SLlamaChatMessage],
        addAssistant: Bool = true,
        config: SLlamaTokenizerConfig? = nil
    ) throws
        -> String
    {
        guard !messages.isEmpty else {
            throw SLlamaError.invalidParameters("Messages array cannot be empty")
        }

        let tokenizerConfig = config ?? SLlamaTokenizerConfig()
        
        // Use a configurable buffer size for chat template application
        let maxLength = tokenizerConfig.chatTemplateBufferSize
        var buffer = [CChar](repeating: 0, count: Int(maxLength))

        let result = llama_chat_apply_template(
            template,
            messages,
            messages.count,
            addAssistant,
            &buffer,
            Int32(maxLength)
        )

        if result < 0 {
            switch result {
                case -1:
                    throw SLlamaError.invalidParameters("Invalid template or messages")
                case -2:
                    throw SLlamaError.bufferTooSmall
                default:
                    throw SLlamaError.operationFailed("Chat template application failed with error code: \(result)")
            }
        }

        // Convert C string to Swift string, handling null termination
        let bytes = Array(buffer.prefix(Int(result))).map { UInt8(bitPattern: $0) }
        guard let string = String(bytes: bytes, encoding: .utf8) else {
            throw SLlamaError.encodingFailure
        }

        return string
    }

    /// Get list of built-in chat templates
    /// - Parameters:
    ///   - config: Tokenizer configuration (optional, uses defaults if nil)
    /// - Returns: Array of template names
    /// - Throws: SLlamaError if operation fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getBuiltinTemplates(config: SLlamaTokenizerConfig? = nil) throws -> [String] {
        let tokenizerConfig = config ?? SLlamaTokenizerConfig()
        let maxTemplates = Int(tokenizerConfig.maxBuiltinTemplates)
        var templates = [UnsafePointer<CChar>?](repeating: nil, count: maxTemplates)

        let result = llama_chat_builtin_templates(&templates, maxTemplates)

        if result < 0 {
            throw SLlamaError.operationFailed("Failed to get builtin templates with error code: \(result)")
        }

        var templateNames: [String] = []
        for i in 0 ..< Int(result) {
            if let template = templates[i] {
                templateNames.append(String(cString: template))
            }
        }

        return templateNames
    }

    // MARK: - Legacy Methods (Deprecated)

    /// Legacy tokenize method that returns nil (deprecated)
    @available(*, deprecated, message: "Use tokenize(text:vocab:addSpecial:parseSpecial:) throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _tokenize(
        text: String,
        vocab: SLlamaVocabPointer?,
        addSpecial: Bool = true,
        parseSpecial: Bool = true
    )
        -> [SLlamaToken]?
    {
        try? tokenize(text: text, vocab: vocab, addSpecial: addSpecial, parseSpecial: parseSpecial)
    }

    /// Legacy tokenToPiece method that returns nil (deprecated)
    @available(*, deprecated, message: "Use tokenToPiece(token:vocab:lstrip:special:) throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _tokenToPiece(
        token: SLlamaToken,
        vocab: SLlamaVocabPointer?,
        lstrip: Int32 = 0,
        special: Bool = false
    )
        -> String?
    {
        try? tokenToPiece(token: token, vocab: vocab, lstrip: lstrip, special: special)
    }

    /// Legacy detokenize method that returns nil (deprecated)
    @available(*, deprecated, message: "Use detokenize(tokens:vocab:removeSpecial:unparseSpecial:) throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _detokenize(
        tokens: [SLlamaToken],
        vocab: SLlamaVocabPointer?,
        removeSpecial: Bool = true,
        unparseSpecial: Bool = true
    )
        -> String?
    {
        try? detokenize(tokens: tokens, vocab: vocab, removeSpecial: removeSpecial, unparseSpecial: unparseSpecial)
    }

    /// Legacy applyChatTemplate method that returns nil (deprecated)
    @available(*, deprecated, message: "Use applyChatTemplate(template:messages:addAssistant:) throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _applyChatTemplate(
        template: String?,
        messages: [SLlamaChatMessage],
        addAssistant: Bool = true
    )
        -> String?
    {
        try? applyChatTemplate(template: template, messages: messages, addAssistant: addAssistant)
    }

    /// Legacy getBuiltinTemplates method that returns nil (deprecated)
    @available(*, deprecated, message: "Use getBuiltinTemplates() throws instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func _getBuiltinTemplates() -> [String]? {
        try? getBuiltinTemplates()
    }
}

/// Extension to SLlamaVocab for tokenization convenience
public extension SLlamaVocab {
    /// Tokenize text using this vocabulary
    /// - Parameters:
    ///   - text: The text to tokenize
    ///   - addSpecial: Whether to add special tokens
    ///   - parseSpecial: Whether to parse special tokens
    /// - Returns: Array of tokens
    /// - Throws: SLlamaError if tokenization fails
    func tokenize(
        text: String,
        addSpecial: Bool = true,
        parseSpecial: Bool = true
    ) throws
        -> [SLlamaToken]
    {
        try SLlamaTokenizer.tokenize(
            text: text,
            vocab: pointer,
            addSpecial: addSpecial,
            parseSpecial: parseSpecial
        )
    }

    /// Convert a token to its text representation
    /// - Parameters:
    ///   - token: The token to convert
    ///   - lstrip: Number of leading spaces to skip
    ///   - special: Whether to render special tokens
    /// - Returns: The text representation
    /// - Throws: SLlamaError if conversion fails
    func tokenToPiece(
        token: SLlamaToken,
        lstrip: Int32 = 0,
        special: Bool = false
    ) throws
        -> String
    {
        try SLlamaTokenizer.tokenToPiece(
            token: token,
            vocab: pointer,
            lstrip: lstrip,
            special: special
        )
    }

    /// Convert tokens back to text
    /// - Parameters:
    ///   - tokens: Array of tokens to convert
    ///   - removeSpecial: Whether to remove special tokens
    ///   - unparseSpecial: Whether to render special tokens in output
    /// - Returns: The text representation
    /// - Throws: SLlamaError if conversion fails
    func detokenize(
        tokens: [SLlamaToken],
        removeSpecial: Bool = true,
        unparseSpecial: Bool = true
    ) throws
        -> String
    {
        try SLlamaTokenizer.detokenize(
            tokens: tokens,
            vocab: pointer,
            removeSpecial: removeSpecial,
            unparseSpecial: unparseSpecial
        )
    }

    // MARK: - Legacy Methods (Deprecated)

    /// Legacy tokenize method that returns nil (deprecated)
    /// - Parameters:
    ///   - text: The text to tokenize
    ///   - addSpecial: Whether to add special tokens
    ///   - parseSpecial: Whether to parse special tokens
    /// - Returns: Array of tokens, or nil if tokenization failed
    @available(*, deprecated, message: "Use tokenize(text:addSpecial:parseSpecial:) throws instead")
    func _tokenize(
        text: String,
        addSpecial: Bool = true,
        parseSpecial: Bool = true
    )
        -> [SLlamaToken]?
    {
        try? tokenize(text: text, addSpecial: addSpecial, parseSpecial: parseSpecial)
    }

    /// Legacy tokenToPiece method that returns nil (deprecated)
    /// - Parameters:
    ///   - token: The token to convert
    ///   - lstrip: Number of leading spaces to skip
    ///   - special: Whether to render special tokens
    /// - Returns: The text representation, or nil if conversion failed
    @available(*, deprecated, message: "Use tokenToPiece(token:lstrip:special:) throws instead")
    func _tokenToPiece(
        token: SLlamaToken,
        lstrip: Int32 = 0,
        special: Bool = false
    )
        -> String?
    {
        try? tokenToPiece(token: token, lstrip: lstrip, special: special)
    }

    /// Legacy detokenize method that returns nil (deprecated)
    /// - Parameters:
    ///   - tokens: Array of tokens to convert
    ///   - removeSpecial: Whether to remove special tokens
    ///   - unparseSpecial: Whether to render special tokens in output
    /// - Returns: The text representation, or nil if conversion failed
    @available(*, deprecated, message: "Use detokenize(tokens:removeSpecial:unparseSpecial:) throws instead")
    func _detokenize(
        tokens: [SLlamaToken],
        removeSpecial: Bool = true,
        unparseSpecial: Bool = true
    )
        -> String?
    {
        try? detokenize(tokens: tokens, removeSpecial: removeSpecial, unparseSpecial: unparseSpecial)
    }
}
