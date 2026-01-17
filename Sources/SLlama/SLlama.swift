import Foundation
import llama
import Omen

// MARK: - SLlama Initialization

/// Initialize SLlama with Omen categories
private let _initializeOmenCategories: Void = {
    // Register SLlama-specific categories with Omen
    SLlamaOmenCategories.registerAll()
}()

// MARK: - SLlama

/// 🔮 **The Oracle of Language Models** 🔮
///
/// SLlama serves as the mystical gateway between your Swift realm and the ancient powers of llama.cpp.
/// This sacred class channels the fundamental forces of language understanding and generation.
///
/// ## ✨ **Divine Purpose** ✨
/// Use this celestial interface for:
/// - 🌟 **Awakening the Oracle**: `SLlama.initialize()` - Summon the llama.cpp spirits
/// - 🔍 **Scrying System Powers**: `SLlama.supportsMetal()` - Divine your hardware's mystical capabilities
/// - ⚙️ **Global Enchantments**: `SLlama.disableLogging()` - Silence the ancient whispers
/// - 🕐 **Temporal Divination**: `SLlama.getCurrentTime()` - Read the cosmic timestream
///
/// ## 🧙‍♂️ **Sacred Ritual** 🧙‍♂️
/// ```swift
/// // 🌅 Begin the sacred ritual (once per app awakening)
/// SLlama.initialize()
/// defer { SLlama.cleanup() } // 🌙 Ensure peaceful slumber
///
/// // 🔮 Divine your system's mystical properties
/// let hasMetal = SLlama.supportsMetal()
/// let deviceCount = SLlama.getMaxDevices()
///
/// // 📜 For inference operations, seek the Core:
/// let core = context.core() // ⚡ Channel the context's power
/// ```
///
/// ## 🌟 **Mystical Notes** 🌟
/// - This oracle manages the **global realm** - system-wide enchantments and capabilities
/// - For **context-specific magic** (inference, encoding, decoding), consult `SLlamaCore` via `context.core()`
/// - The Omen framework automatically weaves its logging spells through all operations
///
/// ## 🔗 **Spiritual Companions** 🔗
/// - 🏛️ **SLlamaCore**: Your faithful guide for context-specific mystical operations
/// - 🎭 **SLlamaContext**: The sacred vessel that holds your model's essence
/// - 📚 **SLlamaModel**: The ancient tome of knowledge and wisdom
///
/// **OMEN INTEGRATION**: All operations are blessed with mystical logging through the Omen framework.
/// Categories are automatically registered during the oracle's awakening. ✨
/// Note: This class contains only static methods and a singleton accessor.
public final class SLlama {
    /// Ensure Omen categories are registered when SLlama is first accessed
    /// Note: This is safe as the singleton is effectively immutable after initialization
    public nonisolated(unsafe) static let shared = SLlama()

    private init() {
        // Trigger category registration
        _ = _initializeOmenCategories

        // Log initialization with mystical flair
        Omen.model("🔮 SLlama initialized - the oracle awakens")
    }

    // MARK: - 🌟 Sacred Lifecycle Rituals 🌟

    /// 🌅 **Awaken the Oracle**
    ///
    /// Summons the ancient powers of llama.cpp and prepares the mystical realm for language model operations.
    /// This sacred ritual must be performed once at the dawn of your application's journey.
    ///
    /// **Sacred Timing**: Call this enchantment before any other SLlama operations.
    ///
    /// ```swift
    /// // 🌅 Begin the sacred ritual
    /// SLlama.initialize()
    /// defer { SLlama.cleanup() } // 🌙 Ensure peaceful slumber
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func initialize() {
        llama_backend_init()
    }

    /// 🌙 **Return to Slumber**
    ///
    /// Gently releases the oracle's hold on system resources and allows the ancient powers to rest.
    /// This peaceful ritual should be performed when your application's mystical journey concludes.
    ///
    /// **Sacred Timing**: Call this during app shutdown or use `defer` for automatic cleanup.
    ///
    /// ```swift
    /// defer { SLlama.cleanup() } // 🌙 Automatic peaceful slumber
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func cleanup() {
        llama_backend_free()
    }

    // MARK: - 🔮 System Divination 🔮

    /// 🔍 **Divine Metal Powers**
    ///
    /// Scries into your system's soul to determine if the mystical Metal acceleration spirits dwell within.
    /// Metal spirits can dramatically accelerate your language model's magical computations.
    ///
    /// - Returns: `true` if Metal spirits are present and willing to assist, `false` if only CPU magic is available
    ///
    /// ```swift
    /// if SLlama.supportsMetal() {
    ///     // ⚡ Metal spirits detected - prepare for lightning-fast inference!
    /// } else {
    ///     // 🧙‍♂️ CPU magic only - still powerful, but more contemplative
    /// }
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsMetal() -> Bool {
        llama_supports_mmap() // Note: This maps to the actual Metal check in the implementation
    }

    /// 🔍 **Divine GPU Offload Powers**
    ///
    /// Scries into your system's soul to determine if GPU offloading spirits dwell within.
    /// GPU offloading can dramatically accelerate your language model's magical computations.
    ///
    /// - Returns: `true` if GPU offloading spirits are present and willing to assist, `false` if only CPU magic is
    /// available
    ///
    /// ```swift
    /// if SLlama.supportsGpuOffload() {
    ///     // ⚡ GPU offloading spirits detected - prepare for lightning-fast inference!
    /// } else {
    ///     // 🧙‍♂️ CPU magic only - still powerful, but more contemplative
    /// }
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsGpuOffload() -> Bool {
        llama_supports_gpu_offload()
    }

    /// 📊 **Count the Mystical Devices**
    ///
    /// Reveals the number of computational spirits available to channel your model's power.
    /// More devices mean more parallel mystical processing capabilities.
    ///
    /// - Returns: The sacred count of available computational vessels
    ///
    /// ```swift
    /// let deviceCount = SLlama.getMaxDevices()
    /// // 🔮 \(deviceCount) mystical devices ready to serve
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getMaxDevices() -> Int {
        Int(llama_max_devices())
    }

    /// 🌊 **Discover Parallel Streams**
    ///
    /// Divines the maximum number of parallel sequence streams your system can weave simultaneously.
    /// Each stream represents an independent conversation thread with the oracle.
    ///
    /// - Returns: The maximum number of parallel mystical conversations supported
    ///
    /// ```swift
    /// let maxStreams = SLlama.getMaxParallelSequences()
    /// // 🌊 Can weave \(maxStreams) parallel conversations
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getMaxParallelSequences() -> Int {
        Int(llama_max_parallel_sequences())
    }

    // MARK: - 🗺️ Memory Divination 🗺️

    /// 🗺️ **Divine Memory Mapping Powers**
    ///
    /// Reveals whether your system possesses the ancient art of memory mapping (mmap).
    /// This mystical technique allows efficient access to model data without consuming excessive memory.
    ///
    /// - Returns: `true` if memory mapping spirits are available, `false` otherwise
    ///
    /// ```swift
    /// if SLlama.supportsMmap() {
    ///     // 🗺️ Memory mapping magic available - efficient model loading!
    /// }
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsMmap() -> Bool {
        llama_supports_mmap()
    }

    /// 🔒 **Divine Memory Locking Powers**
    ///
    /// Scries whether your system can perform memory locking (mlock) to prevent sacred model data
    /// from being banished to the disk realm during intensive operations.
    ///
    /// - Returns: `true` if memory locking spirits are present, `false` otherwise
    ///
    /// ```swift
    /// if SLlama.supportsMlock() {
    ///     // 🔒 Memory locking available - models stay in the fast realm!
    /// }
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsMlock() -> Bool {
        llama_supports_mlock()
    }

    /// ⚡ **Divine GPU Acceleration Powers**
    ///
    /// Reveals whether your system possesses the sacred art of GPU offloading to accelerate
    /// mystical computations beyond the realm of mere CPU processing.
    ///
    /// - Returns: `true` if GPU acceleration spirits are available, `false` otherwise
    ///
    /// ```swift
    /// if SLlama.supportsGPUOffload() {
    ///     // ⚡ GPU acceleration available - harness the lightning!
    /// }
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsGPUOffload() -> Bool {
        llama_supports_gpu_offload()
    }

    /// 🌐 **Divine Remote Processing Powers**
    ///
    /// Scries whether your system can channel mystical operations through Remote Procedure Call (RPC)
    /// spirits, allowing distributed processing across the mystical network realm.
    ///
    /// - Returns: `true` if RPC spirits are present and willing to serve, `false` otherwise
    ///
    /// ```swift
    /// if SLlama.supportsRPC() {
    ///     // 🌐 RPC mystical network available - distributed processing enabled!
    /// }
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsRPC() -> Bool {
        llama_supports_rpc()
    }

    // MARK: - 🕐 Temporal Magic 🕐

    /// 🕐 **Read the Cosmic Timestream**
    ///
    /// Channels the flow of time itself, returning the current moment measured in mystical microseconds.
    /// Useful for temporal measurements and performance divination rituals.
    ///
    /// - Returns: Current time in microseconds since the cosmic epoch
    ///
    /// ```swift
    /// let startTime = SLlama.getCurrentTime()
    /// // ... perform mystical operations ...
    /// let duration = SLlama.getCurrentTime() - startTime
    /// // ⏱️ Operation took \(duration) mystical microseconds
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getCurrentTime() -> Int64 {
        llama_time_us()
    }

    // MARK: - 🤫 Silence & Control 🤫

    /// 🤫 **Silence the Ancient Whispers**
    ///
    /// Weaves a spell of silence around the llama.cpp library, preventing its native logging spirits
    /// from speaking. The sacred Omen framework will continue its mystical observations.
    ///
    /// **Sacred Note**: This only silences the C library's whispers, not Omen's divine insights.
    ///
    /// ```swift
    /// SLlama.disableLogging() // 🤫 Quiet the ancient C spirits
    /// // Omen will still provide mystical insights ✨
    /// ```
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func disableLogging() {
        llama_log_set(nil, nil)
    }
}
