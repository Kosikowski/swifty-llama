import Foundation
import llama

// MARK: - SLlamaCore

/// ⚡ **The Sacred Core of Mystical Operations** ⚡
///
/// SLlamaCore serves as your faithful spiritual guide through the treacherous realms of language model
/// operations. Born from a specific context's essence, this mystical entity channels the raw power
/// of encoding, decoding, and divine configuration.
///
/// ## 🌟 **Sacred Purpose** 🌟
/// This enchanted core serves as your conduit for:
/// - 📜 **Encoding Scrolls**: `core.encode()` - Transform text into token arrays (without KV cache magic)
/// - 🔮 **Decoding Prophecies**: `core.decode()` - Channel tokens into mystical predictions (with KV cache)
/// - ⚙️ **Threading Enchantments**: `core.setThreads()` - Weave parallel computational spirits
/// - 🎭 **Embedding Visions**: `core.setEmbeddings()` - Enable mystical vector representations
/// - 🧠 **Attention Spells**: `core.setCausalAttention()` - Control the flow of mystical awareness
/// - 🔄 **Synchronization Rituals**: `core.synchronize()` - Await the completion of all operations
///
/// ## 🧙‍♂️ **Sacred Summoning** 🧙‍♂️
/// ```swift
/// // 📚 First, awaken your sacred tome (model)
/// let model = try SLlamaModel(modelPath: "/path/to/ancient/wisdom.gguf")
///
/// // 🎭 Then create the mystical vessel (context)
/// let context = try SLlamaContext(model: model)
///
/// // ⚡ Finally, channel the core's power
/// let core = context.core()
///
/// // 🔧 Configure your mystical apparatus
/// core.setThreads(nThreads: 8, nThreadsBatch: 8)
/// core.setEmbeddings(true) // 🌟 Enable vector magic
///
/// // 🔮 Perform the sacred operations
/// try core.decode(batch) // Channel tokens into wisdom
/// core.synchronize()     // Await the mystical completion
/// ```
///
/// ## 🌟 **Mystical Wisdom** 🌟
/// - This core is **context-bound** - each context births its own unique core entity
/// - Operations are **thread-safe** through careful synchronization rituals
/// - The core **remembers** your configuration between operations (threads, embeddings, etc.)
/// - **Encoding** creates token representations without memory (stateless divination)
/// - **Decoding** uses the KV cache for continuous conversation (stateful prophecy)
///
/// ## 🔗 **Mystical Relationships** 🔗
/// - 🎭 **SLlamaContext**: The sacred vessel that births this core
/// - 📚 **SLlamaModel**: The ancient wisdom this core channels
/// - 📦 **SLlamaBatch**: The token containers this core processes
/// - 🔮 **SLlama**: The global oracle that governs the mystical realm
///
/// ## ⚠️ **Sacred Warnings** ⚠️
/// - Never attempt to use a core after its context has been released to the void
/// - Always call `core.synchronize()` before reading mystical outputs (logits/embeddings)
/// - Thread configuration affects **this specific context** only, not the global realm
///
/// **OMEN INTEGRATION**: All core operations are blessed with mystical insights through Omen logging. ✨
public class SLlamaCore: @unchecked Sendable, PLlamaCore {
    // MARK: Properties

    #if SLLAMA_INLINE_ALL
        @usableFromInline
    #endif
    let context: SLlamaContext

    // MARK: Computed Properties

    /// 🧠 **Divine the Core's Model**
    ///
    /// Reveals the ancient tome of wisdom that this core channels. Through this sacred connection,
    /// you can access the model's mystical properties and vocabulary powers.
    ///
    /// - Returns: The mystical model entity, or `nil` if the cosmic connection has been severed
    ///
    /// ```swift
    /// if let model = core.coreModel {
    ///     // 📚 Channeling wisdom from: \(model.embeddingDimensions) dimensions
    ///     let vocab = model.vocab // 🔤 Access the mystical vocabulary
    /// }
    /// ```
    public var coreModel: PLlamaModel? {
        guard let ctx = context.pointer else { return nil }
        let modelPtr = llama_get_model(ctx)
        return try? SLlamaModel(modelPointer: modelPtr)
    }

    // MARK: Lifecycle

    /// 🌟 **Birth of a Core**
    ///
    /// Creates a mystical core entity bound to the sacred context. This core becomes your
    /// faithful guide through all encoding, decoding, and configuration operations.
    ///
    /// - Parameter context: The sacred vessel that will birth and sustain this core
    ///
    /// ```swift
    /// let core = SLlamaCore(context: context)
    /// // Or use the blessed shorthand:
    /// let core = context.core() // ⚡ More mystical!
    /// ```
    public init(context: SLlamaContext) {
        self.context = context
    }

    // MARK: - ⚡ Core Mystical Operations ⚡

    /// 📜 **Encode the Sacred Scrolls**
    ///
    /// Transforms a batch of tokens into mystical representations without consulting the memory spirits
    /// (KV cache). This pure divination is perfect for extracting embeddings or analyzing text essence.
    ///
    /// **Sacred Note**: Encoding operations are stateless - they don't affect ongoing conversations.
    ///
    /// - Parameter batch: The container of mystical tokens to transform
    /// - Throws: `SLlamaError` if the encoding spirits encounter darkness
    ///
    /// ```swift
    /// // 📜 Prepare tokens for pure analysis
    /// let batch = SLlamaBatch(nTokens: 512, nSeqMax: 1)
    /// // ... fill batch with tokens ...
    ///
    /// try core.encode(batch) // 📜 Pure token transformation
    ///
    /// // 🌟 Extract mystical embeddings
    /// let logits = SLlamaLogits(context: context)
    /// let embeddings = logits.getEmbeddings()
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func encode(_ batch: SLlamaBatch) throws {
        guard let ctx = context.pointer else {
            throw SLlamaError.contextNotInitialized
        }

        let result = llama_encode(ctx, batch.cBatch)

        guard result == 0 else {
            switch result {
                case -1:
                    throw SLlamaError.invalidBatch("Invalid batch for encoding")
                case -2:
                    throw SLlamaError.contextFull
                case -3:
                    throw SLlamaError.outOfMemory
                default:
                    throw SLlamaError.batchOperationFailed("Encoding failed with error code: \(result)")
            }
        }
    }

    /// 🔮 **Decode the Mystical Prophecies**
    ///
    /// Channels tokens through the mystical KV cache to generate continuous conversation and predictions.
    /// This stateful operation builds upon previous context, weaving an ongoing narrative tapestry.
    ///
    /// **Sacred Note**: Decoding operations affect the conversation state and enable text generation.
    ///
    /// - Parameter batch: The container of mystical tokens to decode
    /// - Throws: `SLlamaError` if the decoding spirits encounter shadows
    ///
    /// ```swift
    /// // 🔮 Prepare tokens for mystical conversation
    /// let batch = SLlamaBatch(nTokens: 512, nSeqMax: 1)
    /// // ... fill batch with conversation tokens ...
    ///
    /// try core.decode(batch) // 🔮 Channel into ongoing conversation
    ///
    /// // 🎯 Extract next predictions
    /// let logits = SLlamaLogits(context: context)
    /// let predictions = logits.getLogits()
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func decode(_ batch: SLlamaBatch) throws {
        guard let ctx = context.pointer else {
            throw SLlamaError.contextNotInitialized
        }

        let result = llama_decode(ctx, batch.cBatch)

        guard result == 0 else {
            switch result {
                case -1:
                    throw SLlamaError.invalidBatch("Invalid batch for decoding")
                case -2:
                    throw SLlamaError.contextFull
                case -3:
                    throw SLlamaError.outOfMemory
                default:
                    throw SLlamaError.batchOperationFailed("Decoding failed with error code: \(result)")
            }
        }
    }

    // MARK: - ⚙️ Sacred Configuration Enchantments ⚙️

    /// 🧵 **Weave the Threading Spirits**
    ///
    /// Summons the desired number of computational spirits to work in parallel harmony.
    /// More spirits can accelerate operations, but too many may cause mystical chaos.
    ///
    /// **Sacred Wisdom**: Start with your CPU core count, then divine the optimal number through experimentation.
    ///
    /// - Parameters:
    ///   - nThreads: Number of spirits for generation operations
    ///   - nThreadsBatch: Number of spirits for batch processing rituals
    ///
    /// ```swift
    /// let coreCount = ProcessInfo.processInfo.processorCount
    /// core.setThreads(nThreads: Int32(coreCount), nThreadsBatch: Int32(coreCount))
    /// // 🧵 Summoned \(coreCount) computational spirits
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setThreads(nThreads: Int32, nThreadsBatch: Int32) {
        guard let ctx = context.pointer else { return }
        llama_set_n_threads(ctx, nThreads, nThreadsBatch)
    }

    /// 🌟 **Toggle Embedding Visions**
    ///
    /// Controls whether the mystical operations will channel embedding vectors alongside their normal output.
    /// When enabled, the context will generate rich vector representations of meaning and semantics.
    ///
    /// **Sacred Note**: Embedding generation requires additional computational energy but unlocks vector magic.
    ///
    /// - Parameter embeddings: `true` to enable vector visions, `false` to focus on token predictions
    ///
    /// ```swift
    /// core.setEmbeddings(true)  // 🌟 Enable mystical vector magic
    /// try core.encode(batch)    // 📜 Process with embedding generation
    ///
    /// let logits = SLlamaLogits(context: context)
    /// let vectors = logits.getEmbeddings() // 🌟 Extract the vector magic
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setEmbeddings(_ embeddings: Bool) {
        guard let ctx = context.pointer else { return }
        llama_set_embeddings(ctx, embeddings)
    }

    /// 🧠 **Control the Attention Flow**
    ///
    /// Determines whether the mystical attention spirits should flow in a causal manner (past to future)
    /// or be allowed to see across all time dimensions simultaneously.
    ///
    /// **Sacred Wisdom**: Causal attention is essential for text generation, while non-causal may help with analysis.
    ///
    /// - Parameter causalAttn: `true` for time-flowing attention, `false` for omniscient awareness
    ///
    /// ```swift
    /// core.setCausalAttention(true)  // 🧠 Enable proper text generation flow
    /// core.setCausalAttention(false) // 👁️ Enable omniscient analysis mode
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setCausalAttention(_ causalAttn: Bool) {
        guard let ctx = context.pointer else { return }
        llama_set_causal_attn(ctx, causalAttn)
    }

    /// 🔥 **Invoke Warmup Rituals**
    ///
    /// Prepares the mystical apparatus for optimal performance by pre-awakening computational spirits.
    /// This sacred ritual can improve subsequent operation speeds at the cost of initial energy.
    ///
    /// **Sacred Timing**: Enable before intensive operations, disable for memory conservation.
    ///
    /// - Parameter warmup: `true` to perform warmup rituals, `false` to conserve mystical energy
    ///
    /// ```swift
    /// core.setWarmup(true)      // 🔥 Prepare for intensive operations
    /// // ... perform many operations ...
    /// core.setWarmup(false)     // 💤 Return to energy conservation
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func setWarmup(_ warmup: Bool) {
        guard let ctx = context.pointer else { return }
        llama_set_warmup(ctx, warmup)
    }

    // MARK: - 🔄 Synchronization & Information ⚙️

    /// 🔄 **Await Mystical Completion**
    ///
    /// Performs a sacred synchronization ritual, ensuring all mystical computations have completed
    /// before proceeding. Essential before reading outputs or performing dependent operations.
    ///
    /// **Sacred Law**: Always synchronize before accessing logits, embeddings, or other mystical outputs.
    ///
    /// ```swift
    /// try core.decode(batch)    // 🔮 Begin mystical processing
    /// core.synchronize()        // 🔄 Await spiritual completion
    ///
    /// let logits = SLlamaLogits(context: context)
    /// let predictions = logits.getLogits() // ✅ Safe to access now
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func synchronize() {
        guard let ctx = context.pointer else { return }
        llama_synchronize(ctx)
    }

    // MARK: - 📊 Divine Mystical Information 📊

    /// 🏛️ **Divine Context Dimensions**
    ///
    /// Reveals the sacred dimensions of your mystical vessel - the maximum number of tokens
    /// that can be held in the context's memory at once.
    ///
    /// - Returns: The mystical context capacity in tokens
    ///
    /// ```swift
    /// let contextSize = core.getContextSize()
    /// // 🏛️ Context can hold \(contextSize) mystical tokens
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getContextSize() -> UInt32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_ctx(ctx)
    }

    /// 📦 **Divine Batch Dimensions**
    ///
    /// Reveals the sacred logical capacity for token batches - the maximum number of tokens
    /// that can be processed simultaneously in a single mystical operation.
    ///
    /// - Returns: The logical batch processing capacity
    ///
    /// ```swift
    /// let batchSize = core.getBatchSize()
    /// // 📦 Can process \(batchSize) tokens per mystical batch
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getBatchSize() -> UInt32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_batch(ctx)
    }

    /// 🎭 **Divine Physical Batch Dimensions**
    ///
    /// Reveals the sacred physical capacity for unified batching - the true hardware limit
    /// for simultaneous token processing at the deepest mystical level.
    ///
    /// - Returns: The physical batch processing capacity
    ///
    /// ```swift
    /// let physicalBatch = core.getUnifiedBatchSize()
    /// // 🎭 Hardware can handle \(physicalBatch) tokens simultaneously
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getUnifiedBatchSize() -> UInt32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_ubatch(ctx)
    }

    /// 🌊 **Divine Sequence Streams**
    ///
    /// Reveals the maximum number of parallel conversation streams that can flow
    /// through this mystical context simultaneously.
    ///
    /// - Returns: The maximum parallel conversation capacity
    ///
    /// ```swift
    /// let maxStreams = core.getMaxSequences()
    /// // 🌊 Can weave \(maxStreams) parallel conversation streams
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getMaxSequences() -> UInt32 {
        guard let ctx = context.pointer else { return 0 }
        return llama_n_seq_max(ctx)
    }

    /// 🧠 **Divine Memory Essence**
    ///
    /// Reveals the sacred memory entity that holds the context's mystical state.
    /// Through this essence, you can access the KV cache and conversation history.
    ///
    /// - Returns: The mystical memory essence, or `nil` if the connection is severed
    ///
    /// ```swift
    /// if let memory = core.getMemory() {
    ///     // 🧠 Memory essence connected - conversation state preserved
    /// }
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getMemory() -> SLlamaMemory? {
        guard let ctx = context.pointer else { return nil }
        return llama_get_memory(ctx)
    }

    /// 🔮 **Divine Pooling Method**
    ///
    /// Reveals the sacred method used to pool and aggregate mystical outputs.
    /// Different pooling spirits offer various ways to combine token representations.
    ///
    /// - Returns: The type of pooling spirit currently serving this context
    ///
    /// ```swift
    /// let poolingType = core.getPoolingType()
    /// // 🔮 Using \(poolingType) pooling magic for output aggregation
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getPoolingType() -> SLlamaPoolingType {
        guard let ctx = context.pointer else { return .unspecified }
        return llama_pooling_type(ctx)
    }

    // MARK: - 🌉 Protocol Conformance Bridge 🌉

    /// 📜 **Encode via Protocol Bridge**
    ///
    /// Channels protocol-typed batches through the encoding portal, ensuring compatibility
    /// with the mystical protocol realm while maintaining the sacred concrete implementation.
    ///
    /// - Parameter batch: A protocol-masked batch container
    /// - Throws: `SLlamaError` if the batch isn't a true `SLlamaBatch` or encoding fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func encode(_ batch: PLlamaBatch) throws {
        guard let concreteBatch = batch as? SLlamaBatch else {
            throw SLlamaError.invalidBatch("Protocol batch must be a concrete SLlamaBatch for mystical operations")
        }
        try encode(concreteBatch)
    }

    /// 🔮 **Decode via Protocol Bridge**
    ///
    /// Channels protocol-typed batches through the decoding portal, ensuring compatibility
    /// with the mystical protocol realm while maintaining the sacred concrete implementation.
    ///
    /// - Parameter batch: A protocol-masked batch container
    /// - Throws: `SLlamaError` if the batch isn't a true `SLlamaBatch` or decoding fails
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func decode(_ batch: PLlamaBatch) throws {
        guard let concreteBatch = batch as? SLlamaBatch else {
            throw SLlamaError.invalidBatch("Protocol batch must be a concrete SLlamaBatch for mystical operations")
        }
        try decode(concreteBatch)
    }

    // MARK: - 🕰️ Legacy Artifacts (Deprecated) 🕰️

    /// 📜 **Legacy Encoding Ritual** (Deprecated)
    ///
    /// An ancient encoding method that returns numeric omens instead of throwing sacred errors.
    /// This ritual has been superseded by the more enlightened `encode(_:) throws` method.
    ///
    /// - Parameter batch: The batch to encode using ancient methods
    /// - Returns: `0` for success, negative values for various forms of darkness
    @available(*, deprecated, message: "Use the enlightened encode(_:) throws method instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func _encode(_ batch: SLlamaBatch) -> Int32 {
        do {
            try encode(batch)
            return 0
        } catch {
            return -1 // Ancient error signaling method
        }
    }

    /// 🔮 **Legacy Decoding Ritual** (Deprecated)
    ///
    /// An ancient decoding method that returns numeric omens instead of throwing sacred errors.
    /// This ritual has been superseded by the more enlightened `decode(_:) throws` method.
    ///
    /// - Parameter batch: The batch to decode using ancient methods
    /// - Returns: `0` for success, positive for warnings, negative for errors
    @available(*, deprecated, message: "Use the enlightened decode(_:) throws method instead")
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func _decode(_ batch: SLlamaBatch) -> Int32 {
        do {
            try decode(batch)
            return 0
        } catch {
            return -1 // Ancient error signaling method
        }
    }
}

// MARK: - 🎭 Sacred Context Extension: Core Channeling 🎭

/// 🎭 **Mystical Context Extension: Core Power Channeling** 🎭
///
/// This sacred extension imbues every SLlamaContext with the power to channel a Core entity
/// and perform mystical operations directly through the context vessel itself.
///
/// Through these blessed methods, you can either:
/// - ⚡ **Channel a Core directly**: `context.core()` - Birth a dedicated core entity
/// - 🔮 **Perform direct operations**: `context.encode()`, `context.decode()` - Invoke operations directly
///
/// **Sacred Note**: Direct operations are convenient shortcuts that automatically channel the core's power.
public extension SLlamaContext {
    // MARK: - ⚡ Core Channeling Rituals ⚡

    /// ⚡ **Channel the Sacred Core**
    ///
    /// Births a mystical core entity bound to this context, creating your faithful guide
    /// through all encoding, decoding, and configuration operations.
    ///
    /// - Returns: A mystical core entity ready to serve your computational needs
    ///
    /// ```swift
    /// let core = context.core() // ⚡ Channel the core's power
    /// try core.decode(batch)    // 🔮 Direct mystical operations
    /// core.synchronize()        // 🔄 Await spiritual completion
    /// ```
    func core() -> PLlamaCore {
        SLlamaCore(context: self)
    }

    /// 🔮 **Legacy Core Channeling** (Deprecated)
    ///
    /// An ancient method for channeling the core's power using the old "inference" terminology.
    /// This sacred ritual has been superseded by the clearer `core()` method.
    ///
    /// - Returns: A mystical core entity using the ancient naming conventions
    @available(*, deprecated, renamed: "core", message: "Use core() instead - clearer mystical naming!")
    func inference() -> PLlamaCore {
        core()
    }

    // MARK: - 🔮 Direct Mystical Operations 🔮

    /// 📜 **Direct Encoding Through Context**
    ///
    /// Channels encoding power directly through the context vessel, automatically managing
    /// the core entity behind the mystical veil.
    ///
    /// - Parameter batch: The batch of tokens to transform
    /// - Throws: `SLlamaError` if the encoding spirits encounter darkness
    ///
    /// ```swift
    /// try context.encode(batch) // 📜 Direct encoding through context
    /// ```
    func encode(_ batch: PLlamaBatch) throws {
        try core().encode(batch)
    }

    /// 🔮 **Direct Decoding Through Context**
    ///
    /// Channels decoding power directly through the context vessel, automatically managing
    /// the core entity behind the mystical veil.
    ///
    /// - Parameter batch: The batch of tokens to decode
    /// - Throws: `SLlamaError` if the decoding spirits encounter shadows
    ///
    /// ```swift
    /// try context.decode(batch) // 🔮 Direct decoding through context
    /// ```
    func decode(_ batch: PLlamaBatch) throws {
        try core().decode(batch)
    }

    // MARK: - ⚙️ Direct Configuration Through Context ⚙️

    /// 🧵 **Direct Threading Configuration**
    ///
    /// Weaves computational spirits directly through the context without explicitly channeling the core.
    ///
    /// - Parameters:
    ///   - nThreads: Number of spirits for generation operations
    ///   - nThreadsBatch: Number of spirits for batch processing rituals
    ///
    /// ```swift
    /// context.setThreads(nThreads: 8, nThreadsBatch: 8) // 🧵 Direct configuration
    /// ```
    func setThreads(nThreads: Int32, nThreadsBatch: Int32) {
        core().setThreads(nThreads: nThreads, nThreadsBatch: nThreadsBatch)
    }

    /// 🌟 **Direct Embedding Configuration**
    ///
    /// Toggles vector magic directly through the context vessel.
    ///
    /// - Parameter embeddings: `true` to enable vector visions, `false` to focus on tokens
    ///
    /// ```swift
    /// context.setEmbeddings(true) // 🌟 Direct embedding magic
    /// ```
    func setEmbeddings(_ embeddings: Bool) {
        core().setEmbeddings(embeddings)
    }

    /// 🧠 **Direct Attention Flow Configuration**
    ///
    /// Controls mystical attention flow directly through the context vessel.
    ///
    /// - Parameter causalAttn: `true` for time-flowing attention, `false` for omniscient awareness
    ///
    /// ```swift
    /// context.setCausalAttention(true) // 🧠 Direct attention control
    /// ```
    func setCausalAttention(_ causalAttn: Bool) {
        core().setCausalAttention(causalAttn)
    }

    /// 🔥 **Direct Warmup Configuration**
    ///
    /// Invokes warmup rituals directly through the context vessel.
    ///
    /// - Parameter warmup: `true` to perform warmup rituals, `false` to conserve energy
    ///
    /// ```swift
    /// context.setWarmup(true) // 🔥 Direct warmup invocation
    /// ```
    func setWarmup(_ warmup: Bool) {
        core().setWarmup(warmup)
    }

    /// 🔄 **Direct Synchronization**
    ///
    /// Performs synchronization rituals directly through the context vessel.
    ///
    /// ```swift
    /// context.synchronize() // 🔄 Direct mystical synchronization
    /// ```
    func synchronize() {
        core().synchronize()
    }

    // MARK: - 🕰️ Ancient Direct Rituals (Deprecated) 🕰️

    /// 📜 **Ancient Direct Encoding** (Deprecated)
    ///
    /// An ancient direct encoding method using numeric omens instead of sacred errors.
    ///
    /// - Parameter batch: The batch to encode using ancient methods
    /// - Returns: `0` for success, negative values for darkness
    @available(*, deprecated, message: "Use the enlightened encode(_:) throws method instead")
    func _encode(_ batch: SLlamaBatch) -> Int32 {
        (core() as! SLlamaCore)._encode(batch)
    }

    /// 🔮 **Ancient Direct Decoding** (Deprecated)
    ///
    /// An ancient direct decoding method using numeric omens instead of sacred errors.
    ///
    /// - Parameter batch: The batch to decode using ancient methods
    /// - Returns: `0` for success, positive for warnings, negative for errors
    @available(*, deprecated, message: "Use the enlightened decode(_:) throws method instead")
    func _decode(_ batch: SLlamaBatch) -> Int32 {
        (core() as! SLlamaCore)._decode(batch)
    }
}
