import Foundation
import llama
import Omen

// MARK: - Type Aliases

/// Type alias for backwards compatibility
public typealias SSystemCapabilities = SLlamaSystemCapabilities

// MARK: - SLlamaSystemInfo

/// System information and capabilities for SLlama
///
/// **ARCHITECTURAL DECISION**: Using structured logging provides better integration
/// with system debugging tools and allows filtering of system information logs
public struct SLlamaSystemInfo {
    // MARK: - Initialization

    /// Initialize system info
    public init() {}

    // MARK: - System Information Methods

    /// Get system capabilities and hardware information
    /// - Returns: System capabilities structure
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getSystemCapabilities() -> SLlamaSystemCapabilities {
        SLlamaSystemCapabilities()
    }

    // MARK: - Static Utility Methods

    /// Get current time in microseconds
    /// - Returns: Current time in microseconds
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getCurrentTimeMicroseconds() -> Int64 {
        llama_time_us()
    }

    /// Get maximum number of available devices
    /// - Returns: Maximum number of devices
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getMaxDevices() -> Int {
        Int(llama_max_devices())
    }

    /// Check if memory mapping is supported
    /// - Returns: True if mmap is supported
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsMmap() -> Bool {
        llama_supports_mmap()
    }

    /// Check if memory locking is supported
    /// - Returns: True if mlock is supported
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsMlock() -> Bool {
        llama_supports_mlock()
    }

    /// Check if GPU offloading is supported
    /// - Returns: True if GPU offloading is supported
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsGpuOffload() -> Bool {
        llama_supports_gpu_offload()
    }

    /// Check if RPC is supported
    /// - Returns: True if RPC is supported
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func supportsRpc() -> Bool {
        llama_supports_rpc()
    }

    /// Get maximum number of parallel sequences (placeholder implementation)
    /// - Returns: Maximum number of parallel sequences
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public static func getMaxParallelSequences() -> Int {
        // Note: This function may not be available in all llama.cpp builds
        // Providing a reasonable default
        1
    }

    /// Print system capabilities and detailed information with structured logging
    ///
    /// **LOGGING STRATEGY**: System information is logged at INFO level under
    /// .systemInfo category for easy identification and filtering
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func printSystemInfo() {
        let capabilities = getSystemCapabilities()

        Omen.systemInfo(capabilities.getFormattedDescription())
        Omen.systemInfo("Detailed System Info:")
        Omen.systemInfo(capabilities.systemInfo)
    }

    /// Get formatted system information as a string
    /// - Returns: Formatted system information
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
    public func getFormattedSystemInfo() -> String {
        let capabilities = getSystemCapabilities()
        return """
        \(capabilities.getFormattedDescription())

        Detailed System Info:
        \(capabilities.systemInfo)
        """
    }
}

// MARK: - SLlamaSystemCapabilities

/// Detailed system information
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
    #if SLLAMA_INLINE_ALL
        @inlinable
    #endif
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

public extension SLlama {
    /// Get detailed system information
    /// - Returns: System capabilities information
    static func getSystemCapabilities() -> SLlamaSystemCapabilities {
        SLlamaSystemCapabilities()
    }

    /// Print system information to console
    static func printSystemInfo() {
        let capabilities = getSystemCapabilities()
        Omen.info(SLlamaOmenCategories.AI.systemInfo, capabilities.getFormattedDescription())
        Omen.info(SLlamaOmenCategories.AI.systemInfo, "Detailed System Info:")
        Omen.info(SLlamaOmenCategories.AI.systemInfo, capabilities.systemInfo)
    }
}
