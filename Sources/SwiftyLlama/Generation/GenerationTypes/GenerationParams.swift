

/// Parameters for text generation
public struct GenerationParams: Hashable, Sendable {
    /// Random seed for reproducible generation. Same seed with same input produces identical output.
    public let seed: UInt32

    /// Limits the number of highest probability tokens to consider during sampling.
    /// Lower values (1-10) make output more focused, higher values (40-100) increase diversity.
    public let topK: Int32

    /// Nucleus sampling parameter. Only considers tokens whose cumulative probability exceeds this value.
    /// Range: 0.0-1.0. Lower values (0.1-0.5) make output more focused, higher values (0.8-1.0) increase diversity.
    public let topP: Float

    /// Controls randomness in token selection. Higher values (0.8-1.2) increase creativity and randomness,
    /// lower values (0.1-0.5) make output more deterministic and focused.
    public let temperature: Float

    /// Penalty applied to repeated tokens to reduce repetition. Values > 1.0 reduce repetition,
    /// values < 1.0 increase repetition. Typical range: 1.0-1.2.
    public let repeatPenalty: Float

    /// Number of tokens to look back when applying repeat penalty. Higher values consider more context
    /// for repetition detection, but may be computationally more expensive.
    public let repetitionLookback: Int32

    /// Maximum number of tokens to generate in a single generation step.
    /// This limits the length of the model's response, not the context window size.
    public let maxTokens: Int32

    /// Number of CPU threads to use for generation. Higher values may improve performance
    /// on multi-core systems, but may also increase memory usage.
    public let threads: Int32

    /// Number of CPU threads to use for batch processing. This affects how many threads
    /// are used when processing multiple tokens simultaneously.
    public let batchThreads: Int32

    /// Whether to enable embedding generation. When true, the model will also generate
    /// embeddings for the input text, which can be useful for semantic analysis.
    public let enableEmbeddings: Bool

    /// Whether to enable causal attention. When true, tokens can only attend to previous tokens
    /// (standard for text generation). When false, tokens can attend to all tokens (bidirectional).
    public let enableCausalAttention: Bool

    /// Initialize generation parameters with sensible defaults for text generation.
    ///
    /// - Parameters:
    ///   - seed: Random seed for reproducible generation (default: 42)
    ///   - topK: Number of top tokens to consider (default: 40)
    ///   - topP: Nucleus sampling parameter (default: 0.9)
    ///   - temperature: Controls randomness (default: 0.7)
    ///   - repeatPenalty: Penalty for repeated tokens (default: 1.1)
    ///   - repetitionLookback: Tokens to look back for repetition (default: 64)
    ///   - maxTokens: Maximum tokens to generate (default: 100)
    ///   - threads: CPU threads for generation (default: 4)
    ///   - batchThreads: CPU threads for batch processing (default: 4)
    ///   - enableEmbeddings: Enable embedding generation (default: false)
    ///   - enableCausalAttention: Enable causal attention (default: true)
    public init(
        seed: UInt32 = 42,
        topK: Int32 = 40,
        topP: Float = 0.9,
        temperature: Float = 0.7,
        repeatPenalty: Float = 1.1,
        repetitionLookback: Int32 = 64,
        maxTokens: Int32 = 100,
        threads: Int32 = 4,
        batchThreads: Int32 = 4,
        enableEmbeddings: Bool = false,
        enableCausalAttention: Bool = true
    ) {
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temperature = temperature
        self.repeatPenalty = repeatPenalty
        self.repetitionLookback = repetitionLookback
        self.maxTokens = maxTokens
        self.threads = threads
        self.batchThreads = batchThreads
        self.enableEmbeddings = enableEmbeddings
        self.enableCausalAttention = enableCausalAttention
    }
}
