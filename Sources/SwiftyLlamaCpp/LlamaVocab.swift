import Foundation
import llama

/// A wrapper for llama vocabulary
public class LlamaVocab {
    private var vocab: LlamaVocabPointer?
    
    public init?(vocab: LlamaVocabPointer?) {
        guard let vocab = vocab else { return nil }
        self.vocab = vocab
    }
    
    /// Get vocabulary type
    public var type: LlamaVocabType {
        guard let vocab = vocab else { return .none }
        return llama_vocab_type(vocab)
    }
    
    /// Get number of tokens
    public var tokenCount: Int32 {
        guard let vocab = vocab else { return 0 }
        return llama_vocab_n_tokens(vocab)
    }
    
    /// Get text for a token
    public func getText(for token: LlamaToken) -> String? {
        guard let vocab = vocab else { return nil }
        guard let text = llama_vocab_get_text(vocab, token) else { return nil }
        return String(cString: text)
    }
    
    /// Get score for a token
    public func getScore(for token: LlamaToken) -> Float {
        guard let vocab = vocab else { return 0.0 }
        return llama_vocab_get_score(vocab, token)
    }
    
    /// Get attribute for a token
    public func getAttribute(for token: LlamaToken) -> LlamaTokenAttribute {
        guard let vocab = vocab else { return .undefined }
        return llama_vocab_get_attr(vocab, token)
    }
    
    /// Check if token is end of generation
    public func isEndOfGeneration(token: LlamaToken) -> Bool {
        guard let vocab = vocab else { return false }
        return llama_vocab_is_eog(vocab, token)
    }
    
    /// Check if token is control token
    public func isControl(token: LlamaToken) -> Bool {
        guard let vocab = vocab else { return false }
        return llama_vocab_is_control(vocab, token)
    }
    
    /// Get beginning of sentence token
    public var bosToken: LlamaToken {
        guard let vocab = vocab else { return LlamaTokenNull }
        return llama_vocab_bos(vocab)
    }
    
    /// Get end of sentence token
    public var eosToken: LlamaToken {
        guard let vocab = vocab else { return LlamaTokenNull }
        return llama_vocab_eos(vocab)
    }
    
    /// Get end of turn token
    public var eotToken: LlamaToken {
        guard let vocab = vocab else { return LlamaTokenNull }
        return llama_vocab_eot(vocab)
    }
    
    /// Get separator token
    public var sepToken: LlamaToken {
        guard let vocab = vocab else { return LlamaTokenNull }
        return llama_vocab_sep(vocab)
    }
    
    /// Get newline token
    public var newlineToken: LlamaToken {
        guard let vocab = vocab else { return LlamaTokenNull }
        return llama_vocab_nl(vocab)
    }
    
    /// Get padding token
    public var padToken: LlamaToken {
        guard let vocab = vocab else { return LlamaTokenNull }
        return llama_vocab_pad(vocab)
    }
    
    /// Get mask token
    public var maskToken: LlamaToken {
        guard let vocab = vocab else { return LlamaTokenNull }
        return llama_vocab_mask(vocab)
    }
} 