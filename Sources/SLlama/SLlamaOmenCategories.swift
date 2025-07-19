import Foundation
import Omen

// MARK: - SLlama Omen Categories

/// 🔮 SLlama-specific mystical categories extending Omen
///
/// **EXTENSIBILITY EXAMPLE**: This demonstrates how users can extend Omen's core
/// categories with domain-specific ones while maintaining the mystical theming.
public enum SLlamaOmenCategories {
    /// Machine Learning and AI-specific mystical categories
    public enum AI: String, OmenCategory, CaseIterable {
        case model = "Model" // 🧠 Intelligence omens
        case context = "Context" // 🎯 Contextual omens
        case sampler = "Sampler" // 🎲 Randomness omens
        case inference = "Inference" // 🔍 Prediction omens
        case tokenizer = "Tokenizer" // 📝 Language omens
        case quantization = "Quantization" // ⚖️ Precision omens
        case adapter = "Adapter" // 🔗 Adaptation omens
        case batch = "Batch" // 📦 Batch omens
        case systemInfo = "SystemInfo" // 🔧 System omens

        public var description: String {
            switch self {
                case .model:
                    "🧠 Model loading, validation, and management — intelligence omens"
                case .context:
                    "🎯 Context creation, configuration, and operations — contextual omens"
                case .sampler:
                    "🎲 Token sampling strategies and chains — randomness omens"
                case .inference:
                    "🔍 Inference operations (encode/decode) — prediction omens"
                case .tokenizer:
                    "📝 Text tokenization and vocabulary — language omens"
                case .quantization:
                    "⚖️ Model quantization operations — precision omens"
                case .adapter:
                    "🔗 LoRA adapters and control vectors — adaptation omens"
                case .batch:
                    "📦 Batch processing operations — batch omens"
                case .systemInfo:
                    "🔧 System capabilities and hardware information — system omens"
            }
        }

        public var symbol: String {
            switch self {
                case .model: "🧠"
                case .context: "🎯"
                case .sampler: "🎲"
                case .inference: "🔍"
                case .tokenizer: "📝"
                case .quantization: "⚖️"
                case .adapter: "🔗"
                case .batch: "📦"
                case .systemInfo: "🔧"
            }
        }
    }

    /// Register all SLlama categories with Omen at startup
    ///
    /// **USAGE PATTERN**: Call this once during app initialization to make
    /// SLlama categories available throughout the application.
    public static func registerAll() {
        for category in AI.allCases {
            OmenCategories.register(category)
        }
    }
}

// MARK: - Convenience Extensions for SLlama

public extension Omen {
    // MARK: - AI/ML Category Shortcuts

    /// 🧠 Model omen — intelligence visions
    @inlinable
    static func model(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.model, message())
    }

    /// 🧠 Model debug whisper
    @inlinable
    static func modelDebug(_ message: @autoclosure () -> String) {
        debug(SLlamaOmenCategories.AI.model, message())
    }

    /// 🧠 Model error portent
    @inlinable
    static func modelError(_ message: @autoclosure () -> String) {
        error(SLlamaOmenCategories.AI.model, message())
    }

    /// 🎯 Context omen — contextual visions
    @inlinable
    static func context(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.context, message())
    }

    /// 🎲 Sampler omen — randomness visions
    @inlinable
    static func sampler(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.sampler, message())
    }

    /// 🔍 Inference omen — prediction visions
    @inlinable
    static func inference(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.inference, message())
    }

    /// 📝 Tokenizer omen — language visions
    @inlinable
    static func tokenizer(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.tokenizer, message())
    }

    /// ⚖️ Quantization omen — precision visions
    @inlinable
    static func quantization(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.quantization, message())
    }

    /// 🔗 Adapter omen — adaptation visions
    @inlinable
    static func adapter(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.adapter, message())
    }

    /// 📦 Batch omen — batch visions
    @inlinable
    static func batch(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.batch, message())
    }

    /// 🔧 System info omen — hardware visions
    @inlinable
    static func systemInfo(_ message: @autoclosure () -> String) {
        info(SLlamaOmenCategories.AI.systemInfo, message())
    }

    /// 🔧 System debug whisper
    @inlinable
    static func systemDebug(_ message: @autoclosure () -> String) {
        debug(SLlamaOmenCategories.AI.systemInfo, message())
    }
}
