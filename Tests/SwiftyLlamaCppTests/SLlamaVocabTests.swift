import Testing
@testable import SwiftyLlamaCpp

struct SLlamaVocabTests {
    @Test("SLlamaVocab initialization with nil pointer")
    func sLlamaVocabInitWithNilPointer() throws {
        // Test that vocabulary creation works with nil pointer (non-failable init)
        let vocab = SLlamaVocab(vocab: nil)
        #expect(vocab.pointer == nil, "Vocabulary pointer should be nil")
    }

    @Test("SLlamaVocab properties with nil pointer")
    func sLlamaVocabPropertiesWithNilPointer() throws {
        let vocab = SLlamaVocab(vocab: nil)

        // Test properties with nil vocab
        #expect(vocab.tokenCount == 0, "Nil vocab should have 0 token count")
        #expect(vocab.type == .none, "Nil vocab should have .none type")
        #expect(vocab.bosToken == SLlamaTokenNull, "Nil vocab should have null BOS token")
        #expect(vocab.eosToken == SLlamaTokenNull, "Nil vocab should have null EOS token")
        #expect(vocab.eotToken == SLlamaTokenNull, "Nil vocab should have null EOT token")
        #expect(vocab.sepToken == SLlamaTokenNull, "Nil vocab should have null SEP token")
        #expect(vocab.nlToken == SLlamaTokenNull, "Nil vocab should have null newline token")
        #expect(vocab.padToken == SLlamaTokenNull, "Nil vocab should have null pad token")
        #expect(vocab.maskToken == SLlamaTokenNull, "Nil vocab should have null mask token")
    }

    @Test("SLlamaVocab methods with nil pointer")
    func sLlamaVocabMethodsWithNilPointer() throws {
        let vocab = SLlamaVocab(vocab: nil)

        // Test methods with nil vocab
        #expect(vocab.getText(for: 0) == nil, "getText should return nil for nil vocab")
        #expect(vocab.getScore(for: 0) == 0.0, "getScore should return 0.0 for nil vocab")
        #expect(vocab.getAttribute(for: 0) == .undefined, "getAttribute should return .undefined for nil vocab")
        #expect(vocab.isEOG(0) == false, "isEOG should return false for nil vocab")
        #expect(vocab.isControl(0) == false, "isControl should return false for nil vocab")
    }

    @Test("SLlamaVocab FIM tokens with nil pointer")
    func sLlamaVocabFIMTokensWithNilPointer() throws {
        let vocab = SLlamaVocab(vocab: nil)

        // Test FIM tokens with nil vocab
        #expect(vocab.fimPrefixToken == SLlamaTokenNull, "Nil vocab should have null FIM prefix token")
        #expect(vocab.fimSuffixToken == SLlamaTokenNull, "Nil vocab should have null FIM suffix token")
        #expect(vocab.fimMiddleToken == SLlamaTokenNull, "Nil vocab should have null FIM middle token")
        #expect(vocab.fimPaddingToken == SLlamaTokenNull, "Nil vocab should have null FIM padding token")
        #expect(vocab.fimReplacementToken == SLlamaTokenNull, "Nil vocab should have null FIM replacement token")
        #expect(vocab.fimSeparatorToken == SLlamaTokenNull, "Nil vocab should have null FIM separator token")
    }

    @Test("SLlamaVocab token addition flags with nil pointer")
    func sLlamaVocabTokenAdditionFlagsWithNilPointer() throws {
        let vocab = SLlamaVocab(vocab: nil)

        // Test token addition flags with nil vocab
        #expect(vocab.addsBOS == false, "Nil vocab should not add BOS")
        #expect(vocab.addsEOS == false, "Nil vocab should not add EOS")
        #expect(vocab.addsSEP == false, "Nil vocab should not add SEP")
    }
}
