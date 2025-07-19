import Foundation
import llama

/// Main entry point for SLlama
public enum SLlama {
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
