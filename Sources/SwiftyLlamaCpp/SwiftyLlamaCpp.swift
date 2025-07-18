import Foundation
import llama

/// A Swift wrapper for the llama.cpp library
public class SwiftyLlamaCpp {
    /// Initialize the llama library
    public static func initialize() {
        llama_backend_init()
    }

    /// Free the llama library
    public static func free() {
        llama_backend_free()
    }

    /// Get system information
    public static func getSystemInfo() -> String {
        String(cString: llama_print_system_info())
    }

    /// Check if the library supports mmap
    public static func supportsMmap() -> Bool {
        llama_supports_mmap()
    }

    /// Check if the library supports mlock
    public static func supportsMlock() -> Bool {
        llama_supports_mlock()
    }

    /// Check if the library supports GPU offload
    public static func supportsGpuOffload() -> Bool {
        llama_supports_gpu_offload()
    }

    /// Get maximum number of devices
    public static func maxDevices() -> Int {
        Int(llama_max_devices())
    }

    /// Get maximum number of parallel sequences
    public static func maxParallelSequences() -> Int {
        Int(llama_max_parallel_sequences())
    }
}
