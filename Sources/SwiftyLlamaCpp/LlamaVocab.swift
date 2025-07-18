import Foundation
import llama

/// A wrapper for llama vocabulary
public class SLlamaVocab {
    private let vocab: SLlamaVocabPointer?
    
    public init(vocab: SLlamaVocabPointer?) {
        self.vocab = vocab
    }
    
    /// Get the vocabulary pointer for direct C API access
    public var pointer: SLlamaVocabPointer? {
        return vocab
    }
    
    /// Get number of tokens in vocabulary
    public var tokenCount: Int32 {
        guard let vocab else { return 0 }
        return llama_vocab_n_tokens(vocab)
    }
    
    /// Get vocabulary type
    public var type: SLlamaVocabType {
        guard let vocab else { return .none }
        return llama_vocab_type(vocab)
    }
    
    /// Get text for a token
    /// - Parameter token: The token ID
    /// - Returns: The text representation of the token, or nil if not found
    public func getText(for token: SLlamaToken) -> String? {
        guard let vocab else { return nil }
        let text = llama_vocab_get_text(vocab, token)
        return text != nil ? String(cString: text!) : nil
    }
    
    /// Get score for a token
    /// - Parameter token: The token ID
    /// - Returns: The score of the token, or 0.0 if not found
    public func getScore(for token: SLlamaToken) -> Float {
        guard let vocab else { return 0.0 }
        return llama_vocab_get_score(vocab, token)
    }
    
    /// Get attribute for a token
    /// - Parameter token: The token ID
    /// - Returns: The attribute of the token
    public func getAttribute(for token: SLlamaToken) -> SLlamaTokenAttribute {
        guard let vocab else { return .undefined }
        return llama_vocab_get_attr(vocab, token)
    }
    
    /// Check if token is end-of-generation
    /// - Parameter token: The token ID
    /// - Returns: true if token is EOG, false otherwise
    public func isEOG(_ token: SLlamaToken) -> Bool {
        guard let vocab else { return false }
        return llama_vocab_is_eog(vocab, token)
    }
    
    /// Check if token is control token
    /// - Parameter token: The token ID
    /// - Returns: true if token is control, false otherwise
    public func isControl(_ token: SLlamaToken) -> Bool {
        guard let vocab else { return false }
        return llama_vocab_is_control(vocab, token)
    }
    
    /// Get beginning-of-sentence token
    /// - Returns: The BOS token ID
    public var bosToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_bos(vocab)
    }
    
    /// Get end-of-sentence token
    /// - Returns: The EOS token ID
    public var eosToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_eos(vocab)
    }
    
    /// Get end-of-turn token
    /// - Returns: The EOT token ID
    public var eotToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_eot(vocab)
    }
    
    /// Get sentence separator token
    /// - Returns: The SEP token ID
    public var sepToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_sep(vocab)
    }
    
    /// Get newline token
    /// - Returns: The NL token ID
    public var nlToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_nl(vocab)
    }
    
    /// Get padding token
    /// - Returns: The PAD token ID
    public var padToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_pad(vocab)
    }
    
    /// Get mask token
    /// - Returns: The MASK token ID
    public var maskToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_mask(vocab)
    }
    
    /// Check if vocabulary adds BOS token
    /// - Returns: true if BOS token is added, false otherwise
    public var addsBOS: Bool {
        guard let vocab else { return false }
        return llama_vocab_get_add_bos(vocab)
    }
    
    /// Check if vocabulary adds EOS token
    /// - Returns: true if EOS token is added, false otherwise
    public var addsEOS: Bool {
        guard let vocab else { return false }
        return llama_vocab_get_add_eos(vocab)
    }
    
    /// Check if vocabulary adds SEP token
    /// - Returns: true if SEP token is added, false otherwise
    public var addsSEP: Bool {
        guard let vocab else { return false }
        return llama_vocab_get_add_sep(vocab)
    }
    
    /// Get FIM prefix token
    /// - Returns: The FIM prefix token ID
    public var fimPrefixToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_fim_pre(vocab)
    }
    
    /// Get FIM suffix token
    /// - Returns: The FIM suffix token ID
    public var fimSuffixToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_fim_suf(vocab)
    }
    
    /// Get FIM middle token
    /// - Returns: The FIM middle token ID
    public var fimMiddleToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_fim_mid(vocab)
    }
    
    /// Get FIM padding token
    /// - Returns: The FIM padding token ID
    public var fimPaddingToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_fim_pad(vocab)
    }
    
    /// Get FIM replacement token
    /// - Returns: The FIM replacement token ID
    public var fimReplacementToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_fim_rep(vocab)
    }
    
    /// Get FIM separator token
    /// - Returns: The FIM separator token ID
    public var fimSeparatorToken: SLlamaToken {
        guard let vocab else { return SLlamaTokenNull }
        return llama_vocab_fim_sep(vocab)
    }
}
