import Foundation
import llama

// MARK: - SLlamaSampler

/// A wrapper for llama sampling operations
public class SLlamaSampler {
    // MARK: Properties

    var sampler: SLlamaSamplerPointer?

    private let context: SLlamaContext

    // MARK: Computed Properties

    /// Get the underlying C sampler pointer for direct API access
    public var cSampler: SLlamaSamplerPointer? {
        sampler
    }

    /// Get the sampler name
    /// - Returns: The name of the sampler, or nil if not available
    public var name: String? {
        guard let sampler else { return nil }
        return String(cString: llama_sampler_name(sampler))
    }

    // MARK: Lifecycle

    /// Initialize with a context
    /// - Parameter context: The llama context to use for sampling
    public init(context: SLlamaContext) {
        self.context = context
    }

    deinit {
        if let sampler {
            llama_sampler_free(sampler)
        }
    }

    // MARK: Functions

    /// Accept a token (updates internal state of certain samplers)
    /// - Parameter token: The token to accept
    public func accept(_ token: SLlamaToken) {
        guard let sampler else { return }
        llama_sampler_accept(sampler, token)
    }

    /// Apply the sampler to token data array
    /// - Parameter tokenDataArray: The token data array to apply sampling to
    public func apply(to tokenDataArray: SLlamaTokenDataArrayPointer) {
        guard let sampler else { return }
        llama_sampler_apply(sampler, tokenDataArray)
    }

    /// Reset the sampler state
    public func reset() {
        guard let sampler else { return }
        llama_sampler_reset(sampler)
    }

    /// Clone the sampler
    /// - Returns: A new sampler instance, or nil if cloning failed
    public func clone() -> SLlamaSampler? {
        guard let sampler else { return nil }
        guard let clonedSampler = llama_sampler_clone(sampler) else { return nil }

        let newSampler = SLlamaSampler(context: context)
        newSampler.sampler = clonedSampler
        return newSampler
    }

    /// Sample a token from the context
    /// - Parameter lastTokens: Array of last tokens (negative indices for reverse order)
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sample(lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }

        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }

        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0

        if vocabSize == 0 { return nil }

        // Create token data array
        var candidates = [SLlamaTokenData]()
        for i in 0 ..< vocabSize {
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: logits[Int(i)],
                p: 0.0
            ))
        }

        var tokenDataArray = SLlamaTokenDataArray(
            data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: candidates.count,
            selected: 0,
            sorted: false
        )

        // Apply sampling
        apply(to: &tokenDataArray)

        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }

        return SLlamaToken(maxIndex)
    }

    /// Sample with temperature
    /// - Parameters:
    ///   - temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
    ///   - lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleWithTemperature(_ temperature: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }

        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }

        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0

        if vocabSize == 0 { return nil }

        // Create token data array with temperature scaling
        var candidates = [SLlamaTokenData]()
        for i in 0 ..< vocabSize {
            let scaledLogit = temperature > 0 ? logits[Int(i)] / temperature : logits[Int(i)]
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: scaledLogit,
                p: 0.0
            ))
        }

        var tokenDataArray = SLlamaTokenDataArray(
            data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: candidates.count,
            selected: 0,
            sorted: false
        )

        // Apply sampling
        apply(to: &tokenDataArray)

        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }

        return SLlamaToken(maxIndex)
    }

    /// Sample with top-k filtering
    /// - Parameters:
    ///   - k: Number of top tokens to consider
    ///   - lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleTopK(_ k: Int, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }

        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }

        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0

        if vocabSize == 0 { return nil }

        // Create token data array
        var candidates = [SLlamaTokenData]()
        for i in 0 ..< vocabSize {
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: logits[Int(i)],
                p: 0.0
            ))
        }

        // Sort by logit value and take top-k
        candidates.sort { $0.logit > $1.logit }
        candidates = Array(candidates.prefix(k))

        var tokenDataArray = SLlamaTokenDataArray(
            data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: candidates.count,
            selected: 0,
            sorted: true
        )

        // Apply sampling
        apply(to: &tokenDataArray)

        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }

        return candidates[maxIndex].id
    }

    /// Sample with top-p (nucleus) filtering
    /// - Parameters:
    ///   - p: Cumulative probability threshold (0.0 to 1.0)
    ///   - lastTokens: Array of last tokens
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleTopP(_ p: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }

        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }

        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0

        if vocabSize == 0 { return nil }

        // Create token data array
        var candidates = [SLlamaTokenData]()
        for i in 0 ..< vocabSize {
            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: logits[Int(i)],
                p: 0.0
            ))
        }

        // Sort by logit value
        candidates.sort { $0.logit > $1.logit }

        // Calculate cumulative probabilities
        var cumulativeProb = 0.0
        var selectedCandidates: [SLlamaTokenData] = []

        for candidate in candidates {
            let prob = exp(Double(candidate.logit))
            cumulativeProb += prob
            selectedCandidates.append(candidate)

            if cumulativeProb >= Double(p) {
                break
            }
        }

        var tokenDataArray = SLlamaTokenDataArray(
            data: selectedCandidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: selectedCandidates.count,
            selected: 0,
            sorted: true
        )

        // Apply sampling
        apply(to: &tokenDataArray)

        // Find the token with highest probability
        guard let maxIndex = selectedCandidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }

        return selectedCandidates[maxIndex].id
    }

    /// Sample with repetition penalty
    /// - Parameters:
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, >1.0 = penalty)
    ///   - lastTokens: Array of last tokens to penalize
    /// - Returns: The sampled token ID, or nil if sampling failed
    public func sampleWithRepetitionPenalty(_ penalty: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        guard let ctx = context.pointer else { return nil }

        // Get logits from the last token
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }

        // Get vocabulary size
        guard let model = context.associatedModel else { return nil }
        let vocabSize = model.vocab != nil ? llama_vocab_n_tokens(model.vocab!) : 0

        if vocabSize == 0 { return nil }

        // Create token data array with repetition penalty
        var candidates = [SLlamaTokenData]()
        for i in 0 ..< vocabSize {
            var logit = logits[Int(i)]

            // Apply repetition penalty
            for lastToken in lastTokens {
                if SLlamaToken(i) == lastToken {
                    logit *= penalty
                    break
                }
            }

            candidates.append(SLlamaTokenData(
                id: SLlamaToken(i),
                logit: logit,
                p: 0.0
            ))
        }

        var tokenDataArray = SLlamaTokenDataArray(
            data: candidates.withUnsafeMutableBufferPointer { $0.baseAddress },
            size: candidates.count,
            selected: 0,
            sorted: false
        )

        // Apply sampling
        apply(to: &tokenDataArray)

        // Find the token with highest probability
        guard let maxIndex = candidates.enumerated().max(by: { $0.element.p < $1.element.p })?.offset else {
            return nil
        }

        return SLlamaToken(maxIndex)
    }
}

// MARK: - Built-in Sampler Initializers

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

        guard let samplerPtr = llama_sampler_init_logit_bias(nVocab, Int32(biasArray.count), biasArray.withUnsafeBufferPointer { $0.baseAddress }) else { return nil }

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
        guard let samplerPtr = llama_sampler_init_penalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent) else { return nil }

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

// MARK: - Extension to SLlamaContext for Sampling

public extension SLlamaContext {
    /// Create a sampler for this context
    /// - Returns: A SLlamaSampler instance
    func sampler() -> SLlamaSampler {
        SLlamaSampler(context: self)
    }

    /// Sample a token using greedy sampling
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleGreedy() -> SLlamaToken? {
        guard let sampler = SLlamaSampler.greedy(context: self) else { return nil }
        return sampler.sample()
    }

    /// Sample a token using distribution sampling
    /// - Parameter seed: Random seed for the sampler
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleDistribution(seed: UInt32) -> SLlamaToken? {
        guard let sampler = SLlamaSampler.distribution(context: self, seed: seed) else { return nil }
        return sampler.sample()
    }

    /// Sample a token with temperature
    /// - Parameter temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithTemperature(_ temperature: Float) -> SLlamaToken? {
        sampler().sampleWithTemperature(temperature)
    }

    /// Sample a token with top-k filtering
    /// - Parameter k: Number of top tokens to consider
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleTopK(_ k: Int) -> SLlamaToken? {
        sampler().sampleTopK(k)
    }

    /// Sample a token with top-p (nucleus) filtering
    /// - Parameter p: Cumulative probability threshold (0.0 to 1.0)
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleTopP(_ p: Float) -> SLlamaToken? {
        sampler().sampleTopP(p)
    }

    /// Sample a token with repetition penalty
    /// - Parameters:
    ///   - penalty: Repetition penalty factor (1.0 = no penalty, higher = more penalty)
    ///   - lastTokens: Array of last tokens to penalize
    /// - Returns: The sampled token ID, or nil if sampling failed
    func sampleWithRepetitionPenalty(_ penalty: Float, lastTokens: [SLlamaToken] = []) -> SLlamaToken? {
        sampler().sampleWithRepetitionPenalty(penalty, lastTokens: lastTokens)
    }
}
