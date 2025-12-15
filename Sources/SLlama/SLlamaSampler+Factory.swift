import Foundation
import llama

// MARK: - Built-in Sampler Factory Methods

public extension SLlamaSampler {
    /// Initialize a greedy sampler (always picks the highest probability token)
    /// - Returns: A new greedy sampler instance
    static func greedy(context: SLlamaContext) -> SLlamaSampler? {
        guard let samplerPtr = llama_sampler_init_greedy() else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Initialize a distribution sampler with seed
    /// - Parameters:
    ///   - context: The llama context
    ///   - seed: Random seed for the sampler
    /// - Returns: A new distribution sampler instance
    static func distribution(context: SLlamaContext, seed: UInt32) -> SLlamaSampler? {
        guard let samplerPtr = llama_sampler_init_dist(seed) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a temperature sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - temperature: Temperature for sampling
    /// - Returns: A temperature sampler, or nil if initialization failed
    static func temperature(
        context: SLlamaContext,
        temperature: Float
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_temp(temperature) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a top-k sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - k: Number of top tokens to consider
    /// - Returns: A top-k sampler, or nil if initialization failed
    static func topK(
        context: SLlamaContext,
        k: Int32
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_top_k(k) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a top-p (nucleus) sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - p: Cumulative probability threshold
    ///   - minKeep: Minimum number of tokens to keep
    /// - Returns: A top-p sampler, or nil if initialization failed
    static func topP(
        context: SLlamaContext,
        p: Float,
        minKeep: Int = 1
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_top_p(p, minKeep) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a min-p sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - p: Minimum probability threshold
    ///   - minKeep: Minimum number of tokens to keep
    /// - Returns: A min-p sampler, or nil if initialization failed
    static func minP(
        context: SLlamaContext,
        p: Float,
        minKeep: Int = 1
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_min_p(p, minKeep) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a typical sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - p: Typical probability threshold
    ///   - minKeep: Minimum number of tokens to keep
    /// - Returns: A typical sampler, or nil if initialization failed
    static func typical(
        context: SLlamaContext,
        p: Float,
        minKeep: Int = 1
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_typical(p, minKeep) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create an extended temperature sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - temperature: Temperature value
    ///   - delta: Temperature delta
    ///   - exponent: Temperature exponent
    /// - Returns: An extended temperature sampler, or nil if initialization failed
    static func temperatureExtended(
        context: SLlamaContext,
        temperature: Float,
        delta: Float,
        exponent: Float
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_temp_ext(temperature, delta, exponent) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create an XTC sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - p: Probability threshold
    ///   - temperature: Temperature value
    ///   - minKeep: Minimum number of tokens to keep
    ///   - seed: Random seed
    /// - Returns: An XTC sampler, or nil if initialization failed
    static func xtc(
        context: SLlamaContext,
        p: Float,
        temperature: Float,
        minKeep: Int = 1,
        seed: UInt32
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_xtc(p, temperature, minKeep, seed) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a top-n-sigma sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - n: Sigma value
    /// - Returns: A top-n-sigma sampler, or nil if initialization failed
    static func topNSigma(
        context: SLlamaContext,
        n: Float
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_top_n_sigma(n) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a mirostat sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - nVocab: Vocabulary size
    ///   - seed: Random seed
    ///   - tau: Target entropy
    ///   - eta: Learning rate
    ///   - m: Number of tokens for estimation
    /// - Returns: A mirostat sampler, or nil if initialization failed
    static func mirostat(
        context: SLlamaContext,
        nVocab: Int32,
        seed: UInt32,
        tau: Float,
        eta: Float,
        m: Int32
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_mirostat(nVocab, seed, tau, eta, m) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a mirostat v2 sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - seed: Random seed
    ///   - tau: Target entropy
    ///   - eta: Learning rate
    /// - Returns: A mirostat v2 sampler, or nil if initialization failed
    static func mirostatV2(
        context: SLlamaContext,
        seed: UInt32,
        tau: Float,
        eta: Float
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_mirostat_v2(seed, tau, eta) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a repetition penalty sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - penalty: Repetition penalty factor
    /// - Returns: A repetition penalty sampler, or nil if initialization failed
    static func repetitionPenalty(
        context: SLlamaContext,
        penalty: Float
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_penalties(0, penalty, 0.0, 0.0) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a logit bias sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - nVocab: Vocabulary size
    ///   - logitBias: Dictionary mapping token IDs to bias values
    /// - Returns: A logit bias sampler, or nil if initialization failed
    static func logitBias(
        context: SLlamaContext,
        nVocab: Int32,
        logitBias: [SLlamaToken: Float]
    )
        -> SLlamaSampler?
    {
        // Convert dictionary to array of logit bias structures
        var biasArray = [SLlamaLogitBias]()
        for (token, bias) in logitBias {
            biasArray.append(SLlamaLogitBias(
                token: token,
                bias: bias
            ))
        }

        guard let samplerPtr = llama_sampler_init_logit_bias(
            nVocab,
            Int32(biasArray.count),
            biasArray.withUnsafeBufferPointer { $0.baseAddress }
        ) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a grammar sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - vocab: The vocabulary to use
    ///   - grammarString: The grammar rules as a string
    ///   - grammarRoot: The root symbol name
    /// - Returns: A grammar sampler, or nil if initialization failed
    static func grammar(
        context: SLlamaContext,
        vocab: SLlamaVocabPointer,
        grammarString: String,
        grammarRoot: String
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_grammar(vocab, grammarString, grammarRoot) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create a penalties sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - penaltyLastN: Number of last tokens to penalize (0 = disable, -1 = context size)
    ///   - penaltyRepeat: Repetition penalty (1.0 = disabled)
    ///   - penaltyFreq: Frequency penalty (0.0 = disabled)
    ///   - penaltyPresent: Presence penalty (0.0 = disabled)
    /// - Returns: A penalties sampler, or nil if initialization failed
    static func penalties(
        context: SLlamaContext,
        penaltyLastN: Int32,
        penaltyRepeat: Float,
        penaltyFreq: Float,
        penaltyPresent: Float
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_penalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent)
        else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Create an infill sampler
    /// - Parameters:
    ///   - context: The llama context
    ///   - vocab: The vocabulary to use
    /// - Returns: An infill sampler, or nil if initialization failed
    static func infill(
        context: SLlamaContext,
        vocab: SLlamaVocabPointer
    )
        -> SLlamaSampler?
    {
        guard let samplerPtr = llama_sampler_init_infill(vocab) else { return nil }

        let sampler = SLlamaSampler(context: context)
        sampler.sampler = samplerPtr
        return sampler
    }

    /// Get the seed used by the sampler
    /// - Returns: The seed value, or LLAMA_DEFAULT_SEED if not applicable
    func getSeed() -> UInt32 {
        guard let sampler else { return LLAMA_DEFAULT_SEED }
        return llama_sampler_get_seed(sampler)
    }

    /// Sample and accept a token from the specified output index
    /// - Parameters:
    ///   - context: The llama context
    ///   - index: The output index to sample from
    /// - Returns: The sampled token, or nil if sampling failed
    func sampleFromIndex(_ index: Int32) -> SLlamaToken? {
        guard
            let sampler,
            let ctx = context.pointer else { return nil }

        let token = llama_sampler_sample(sampler, ctx, index)
        return token
    }
}
