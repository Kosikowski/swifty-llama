import Foundation
import llama

// MARK: - SLlamaSystemInfo

/// A wrapper for llama.cpp system information and utility functions
public class SLlamaSystemInfo {
    /// Log detailed system information to stdout
    public static func logSystemInfo() {
        llama_print_system_info()
    }

    /// Get current time in microseconds
    /// - Returns: Current time in microseconds
    public static func getCurrentTimeMicroseconds() -> Int64 {
        llama_time_us()
    }

    /// Get maximum number of available devices
    /// - Returns: Maximum number of devices
    public static func getMaxDevices() -> Int {
        Int(llama_max_devices())
    }

    /// Get maximum number of parallel sequences
    /// - Returns: Maximum number of parallel sequences
    public static func getMaxParallelSequences() -> Int {
        Int(llama_max_parallel_sequences())
    }

    /// Check if memory mapping is supported
    /// - Returns: True if mmap is supported
    public static func supportsMmap() -> Bool {
        llama_supports_mmap()
    }

    /// Check if memory locking is supported
    /// - Returns: True if mlock is supported
    public static func supportsMlock() -> Bool {
        llama_supports_mlock()
    }

    /// Check if GPU offloading is supported
    /// - Returns: True if GPU offloading is supported
    public static func supportsGpuOffload() -> Bool {
        llama_supports_gpu_offload()
    }

    /// Check if RPC is supported
    /// - Returns: True if RPC is supported
    public static func supportsRpc() -> Bool {
        llama_supports_rpc()
    }
}

// MARK: - SLlamaSystemCapabilities

/// Comprehensive system information
public struct SLlamaSystemCapabilities {
    // MARK: Properties

    public let maxDevices: Int
    public let maxParallelSequences: Int
    public let supportsMmap: Bool
    public let supportsMlock: Bool
    public let supportsGpuOffload: Bool
    public let supportsRpc: Bool
    public let systemInfo: String
    public let currentTimeMicroseconds: Int64

    // MARK: Lifecycle

    public init() {
        maxDevices = SLlamaSystemInfo.getMaxDevices()
        maxParallelSequences = SLlamaSystemInfo.getMaxParallelSequences()
        supportsMmap = SLlamaSystemInfo.supportsMmap()
        supportsMlock = SLlamaSystemInfo.supportsMlock()
        supportsGpuOffload = SLlamaSystemInfo.supportsGpuOffload()
        supportsRpc = SLlamaSystemInfo.supportsRpc()
        systemInfo = "System information available via logSystemInfo()"
        currentTimeMicroseconds = SLlamaSystemInfo.getCurrentTimeMicroseconds()
    }

    // MARK: Functions

    /// Get a formatted description of system capabilities
    /// - Returns: Formatted string describing system capabilities
    public func getFormattedDescription() -> String {
        var description = "System Capabilities:\n"
        description += "  Max Devices: \(maxDevices)\n"
        description += "  Max Parallel Sequences: \(maxParallelSequences)\n"
        description += "  Memory Mapping: \(supportsMmap ? "Supported" : "Not Supported")\n"
        description += "  Memory Locking: \(supportsMlock ? "Supported" : "Not Supported")\n"
        description += "  GPU Offloading: \(supportsGpuOffload ? "Supported" : "Not Supported")\n"
        description += "  RPC: \(supportsRpc ? "Supported" : "Not Supported")\n"
        description += "  Current Time (μs): \(currentTimeMicroseconds)\n"
        return description
    }
}

// MARK: - Convenience Extensions

public extension SwiftyLlamaCpp {
    /// Get comprehensive system information
    /// - Returns: System capabilities information
    static func getSystemCapabilities() -> SLlamaSystemCapabilities {
        SLlamaSystemCapabilities()
    }

    /// Print system information to console
    static func printSystemInfo() {
        let capabilities = getSystemCapabilities()
        print(capabilities.getFormattedDescription())
        print("Detailed System Info:")
        print(capabilities.systemInfo)
    }
}
