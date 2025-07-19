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

/// Swift wrapper for the llama.cpp library
///
/// **OMEN INTEGRATION**: SLlama uses Omen for structured logging with mystical theming.
/// Categories are automatically registered during module initialization.
public final class SLlama: @unchecked Sendable {
    /// Ensure Omen categories are registered when SLlama is first accessed
    public static let shared = SLlama()

    private init() {
        // Trigger category registration
        _ = _initializeOmenCategories

        // Log initialization with mystical flair
        Omen.model("🔮 SLlama initialized - the oracle awakens")
    }

    /// Initialize the llama backend
    public static func initialize() {
        llama_backend_init()
    }

    /// Free the llama backend
    public static func cleanup() {
        llama_backend_free()
    }

    /// Disable all logging output
    public static func disableLogging() {
        llama_log_set(nil, nil)
    }

    /// Get the current time in microseconds
    /// - Returns: Current time in microseconds
    public static func getCurrentTime() -> Int64 {
        llama_time_us()
    }

    /// Get maximum number of devices
    /// - Returns: Maximum number of devices
    public static func getMaxDevices() -> Int {
        Int(llama_max_devices())
    }

    /// Get maximum number of parallel sequences
    /// - Returns: Maximum number of parallel sequences
    public static func getMaxParallelSequences() -> Int {
        Int(llama_max_parallel_sequences())
    }

    /// Check if mmap is supported
    /// - Returns: True if mmap is supported
    public static func supportsMmap() -> Bool {
        llama_supports_mmap()
    }

    /// Check if mlock is supported
    /// - Returns: True if mlock is supported
    public static func supportsMlock() -> Bool {
        llama_supports_mlock()
    }

    /// Check if GPU offload is supported
    /// - Returns: True if GPU offload is supported
    public static func supportsGPUOffload() -> Bool {
        llama_supports_gpu_offload()
    }

    /// Check if RPC is supported
    /// - Returns: true if RPC is supported, false otherwise
    public static func supportsRPC() -> Bool {
        llama_supports_rpc()
    }
}
