import Foundation
import os.log

// MARK: - Omen Category Protocol

/// 🔮 Protocol for defining mystical logging categories
///
/// **ARCHITECTURAL DECISION**: Protocol-based categories allow for easy extension
/// and customization while maintaining the mystical theming and structure.
public protocol OmenCategory: Sendable {
    /// Unique identifier for the category
    var rawValue: String { get }

    /// Mystical description of the category
    var description: String { get }

    /// Emoji symbol representing the category's essence
    var symbol: String { get }
}

// MARK: - Core Mystical Categories

/// 🔮 Core mystical categories - the foundation of all omens
///
/// These represent universal patterns found in all software systems.
/// Users can extend with their own domain-specific categories.
public enum OmenCategories {
    /// Core universal categories available to all users
    public enum Core: String, OmenCategory, CaseIterable {
        case performance = "Performance" // ⚡️ Speed omens
        case memory = "Memory" // 💾 Memory omens
        case network = "Network" // 🌐 Network omens
        case security = "Security" // 🔒 Security omens
        case database = "Database" // 🗄️ Data omens
        case general = "General" // ✨ General omens

        public var description: String {
            switch self {
                case .performance:
                    "⚡️ Performance metrics and timing — speed omens"
                case .memory:
                    "💾 Memory management and allocation — memory omens"
                case .network:
                    "🌐 Network operations and connectivity — network omens"
                case .security:
                    "🔒 Security events and authentication — security omens"
                case .database:
                    "🗄️ Data operations and persistence — data omens"
                case .general:
                    "✨ General purpose logging — universal omens"
            }
        }

        public var symbol: String {
            switch self {
                case .performance: "⚡️"
                case .memory: "💾"
                case .network: "🌐"
                case .security: "🔒"
                case .database: "🗄️"
                case .general: "✨"
            }
        }
    }

    // MARK: - Category Registration (Simplified)

    /// Simple category registration - for demonstration purposes
    /// In a production system, you might want a more sophisticated registry
    public static func register(_: some OmenCategory) {
        // For now, registration is just a no-op since categories work directly
        // This demonstrates the API pattern for extensibility
    }
}

// MARK: - Omen Protocol

/// 🔮 Omen — for logs that reveal the future (or at least the crash)
///
/// **ARCHITECTURAL DECISION**: Protocol-based design allows for dependency injection
/// while @inlinable methods ensure zero overhead when the compiler optimizes.
///
/// "The future leaves traces." - Every action, every event, every warning is a sign.
/// A whisper before the crash. A shadow of what's to come.
public protocol OmenLogger: Sendable {
    /// Log a message with specified level and category
    func log(_ level: Omen.Level, category: some OmenCategory, _ message: @autoclosure () -> String)

    /// Enable or disable logging
    func setEnabled(_ enabled: Bool)

    /// Set minimum log level
    func setMinimumLogLevel(_ level: Omen.Level)
}

// MARK: - Omen

/// 🔮 Omen — A prophetic trace system for Swift
///
/// Omen isn't just a logger — it's a prophetic trace system for Swift.
/// Every action, every event, every warning is a sign.
/// A whisper before the crash. A shadow of what's to come.
///
/// Designed to feel like magic, but built on solid observability,
/// Omen captures the soul of your app — so you can read its fate
/// before the exception even knows it's coming.
///
/// Whether you're debugging live issues or watching quiet patterns,
/// Omen gives you visibility with style.
/// Think structured logs, symbol-rich traces, and just enough drama.
public final class Omen: OmenLogger, @unchecked Sendable {
    // MARK: - Shared Instance

    /// The oracle — shared instance for convenient static access
    public static let shared: OmenLogger = Omen()

    // MARK: - Log Levels

    /// Prophetic log levels — each carries different weight of destiny
    public enum Level: String, CaseIterable, Sendable {
        case debug = "DEBUG" // 🔍 Whispers from the depths
        case info = "INFO" // ℹ️ Signs in plain sight
        case notice = "NOTICE" // 📋 Omens of note
        case warning = "WARNING" // ⚠️ Shadows of trouble ahead
        case error = "ERROR" // ❌ Dark portents
        case fault = "FAULT" // 💥 Apocalyptic visions

        /// Convert to OSLogType for the system oracles
        var osLogType: OSLogType {
            switch self {
                case .debug:
                    .debug
                case .info:
                    .info
                case .notice:
                    .default
                case .warning:
                    .default
                case .error:
                    .error
                case .fault:
                    .fault
            }
        }

        /// Mystical symbols for each level
        var symbol: String {
            switch self {
                case .debug:
                    "🔍"
                case .info:
                    "ℹ️"
                case .notice:
                    "📋"
                case .warning:
                    "⚠️"
                case .error:
                    "❌"
                case .fault:
                    "💥"
            }
        }
    }

    // MARK: - Properties

    private static let subsystem = "com.oracle.Omen"
    private var loggers: [String: OSLog] = [:]
    private var _isEnabled = true
    private var _minimumLogLevel: Level = .debug
    private let queue = DispatchQueue(label: "omen.logger", qos: .utility)

    // MARK: - Initialization

    /// Initialize the oracle
    public init() {
        // Register core categories by default
        for category in OmenCategories.Core.allCases {
            OmenCategories.register(category)
        }
    }

    /// Get logger for category
    private func logger(for category: some OmenCategory) -> OSLog {
        queue.sync {
            let key = category.rawValue
            if let existingLogger = loggers[key] {
                return existingLogger
            }

            let newLogger = OSLog(subsystem: Self.subsystem, category: key)
            loggers[key] = newLogger
            return newLogger
        }
    }

    // MARK: - Configuration

    /// Enable or disable the oracle's sight
    public func setEnabled(_ enabled: Bool) {
        queue.async {
            self._isEnabled = enabled
        }
    }

    /// Set the minimum level of omens to reveal
    public func setMinimumLogLevel(_ level: Level) {
        queue.async {
            self._minimumLogLevel = level
        }
    }

    /// Check if the oracle should speak this omen
    private func shouldLog(level: Level) -> Bool {
        queue.sync {
            guard self._isEnabled else { return false }

            let levelPriority: [Level: Int] = [
                .debug: 0,
                .info: 1,
                .notice: 2,
                .warning: 3,
                .error: 4,
                .fault: 5,
            ]

            let currentPriority = levelPriority[level] ?? 0
            let minimumPriority = levelPriority[self._minimumLogLevel] ?? 0

            return currentPriority >= minimumPriority
        }
    }

    // MARK: - Core Logging

    /// Reveal an omen through the mystical channels
    public func log(
        _ level: Level,
        category: some OmenCategory,
        _ message: @autoclosure () -> String
    ) {
        guard shouldLog(level: level) else { return }

        let logger = logger(for: category)
        let omenMessage = "\(level.symbol) \(category.symbol) \(message())"

        os_log("%{public}@", log: logger, type: level.osLogType, omenMessage)
    }
}

// MARK: - Static Convenience API

public extension Omen {
    // MARK: - Generic Logging Methods (Inlinable for Performance)

    /// 🔍 Debug whispers — for traces from the depths
    @inlinable
    static func debug(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        shared.log(.debug, category: category, message())
    }

    /// ℹ️ Informational signs — for omens in plain sight
    @inlinable
    static func info(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        shared.log(.info, category: category, message())
    }

    /// 📋 Notable omens — for signs that deserve attention
    @inlinable
    static func notice(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        shared.log(.notice, category: category, message())
    }

    /// ⚠️ Warning shadows — for trouble brewing ahead
    @inlinable
    static func warning(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        shared.log(.warning, category: category, message())
    }

    /// ❌ Error portents — for dark omens manifested
    @inlinable
    static func error(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        shared.log(.error, category: category, message())
    }

    /// 💥 Fault visions — for apocalyptic revelations
    @inlinable
    static func fault(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        shared.log(.fault, category: category, message())
    }

    // MARK: - Configuration

    /// Enable or disable the oracle's sight globally
    @inlinable
    static func setEnabled(_ enabled: Bool) {
        shared.setEnabled(enabled)
    }

    /// Set the minimum level of omens to reveal globally
    @inlinable
    static func setMinimumLogLevel(_ level: Level) {
        shared.setMinimumLogLevel(level)
    }
}

// MARK: - Core Category Convenience Extensions

public extension Omen {
    // MARK: - Core Category Shortcuts

    /// ⚡️ Performance omen — speed visions
    @inlinable
    static func performance(_ message: @autoclosure () -> String) {
        info(OmenCategories.Core.performance, message())
    }

    /// 💾 Memory omen — allocation visions
    @inlinable
    static func memory(_ message: @autoclosure () -> String) {
        info(OmenCategories.Core.memory, message())
    }

    /// 🌐 Network omen — connectivity visions
    @inlinable
    static func network(_ message: @autoclosure () -> String) {
        info(OmenCategories.Core.network, message())
    }

    /// 🔒 Security omen — protection visions
    @inlinable
    static func security(_ message: @autoclosure () -> String) {
        info(OmenCategories.Core.security, message())
    }

    /// 🗄️ Database omen — data visions
    @inlinable
    static func database(_ message: @autoclosure () -> String) {
        info(OmenCategories.Core.database, message())
    }

    /// ✨ General omen — universal visions
    @inlinable
    static func general(_ message: @autoclosure () -> String) {
        info(OmenCategories.Core.general, message())
    }
}

// MARK: - Performance Measurement Magic

public extension Omen {
    /// 🔮 Measure time and reveal the duration omen
    @inlinable
    static func measureTime<R>(
        category: some OmenCategory,
        operation: String,
        using logger: OmenLogger = shared,
        _ block: () throws -> R
    ) rethrows
        -> R
    {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime

        logger.log(
            .info,
            category: category,
            "⏱️ \(operation) completed in \(String(format: "%.3f", timeElapsed * 1000))ms"
        )
        return result
    }

    /// 🔮 Measure async time and reveal the duration omen
    @inlinable
    static func measureTime<R>(
        category: some OmenCategory,
        operation: String,
        using logger: OmenLogger = shared,
        _ block: () async throws -> R
    ) async rethrows
        -> R
    {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await block()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime

        logger.log(
            .info,
            category: category,
            "⏱️ \(operation) completed in \(String(format: "%.3f", timeElapsed * 1000))ms"
        )
        return result
    }

    /// ⚡️ Convenience performance measurement
    @inlinable
    static func measurePerformance<R>(
        operation: String,
        using logger: OmenLogger = shared,
        _ block: () throws -> R
    ) rethrows
        -> R
    {
        try measureTime(category: OmenCategories.Core.performance, operation: operation, using: logger, block)
    }
}

// MARK: - Injectable Protocol Extensions

public extension OmenLogger {
    /// Convenience methods for protocol conformers

    @inlinable
    func debug(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        log(.debug, category: category, message())
    }

    @inlinable
    func info(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        log(.info, category: category, message())
    }

    @inlinable
    func warning(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        log(.warning, category: category, message())
    }

    @inlinable
    func error(_ category: some OmenCategory, _ message: @autoclosure () -> String) {
        log(.error, category: category, message())
    }

    @inlinable
    func performance(_ message: @autoclosure () -> String) {
        info(OmenCategories.Core.performance, message())
    }
}
