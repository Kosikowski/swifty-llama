import Foundation
import llama

/// A wrapper for llama LoRA adapter operations
public class SLlamaAdapter {
    private var adapter: SLlamaAdapterLoraPointer?
    
    /// Initialize with a LoRA adapter file path
    /// - Parameters:
    ///   - model: The model to attach the adapter to
    ///   - path: Path to the LoRA adapter file
    public init?(model: SLlamaModel, path: String) {
        guard let modelPtr = model.pointer else { return nil }
        let adapterPtr = llama_adapter_lora_init(modelPtr, path)
        if adapterPtr == nil {
            return nil
        }
        adapter = adapterPtr
    }
    
    deinit {
        if let adapter = adapter {
            llama_adapter_lora_free(adapter)
        }
    }
    
    /// Get the underlying C adapter pointer for direct API access
    public var cAdapter: SLlamaAdapterLoraPointer? {
        return adapter
    }
    
    /// Check if the adapter is valid
    public var isValid: Bool {
        return adapter != nil
    }
}

// MARK: - Extension to SLlamaContext for LoRA Support

public extension SLlamaContext {
    
    /// Add a LoRA adapter to this context
    /// - Parameter adapter: The LoRA adapter to add
    /// - Returns: true if the adapter was successfully added, false otherwise
    func addLoRAAdapter(_ adapter: SLlamaAdapter) -> Bool {
        guard let context = pointer, let adapterPtr = adapter.cAdapter else { return false }
        return llama_set_adapter_lora(context, adapterPtr, 0) == 0
    }
    
    /// Remove a specific LoRA adapter from this context
    /// - Parameter adapter: The LoRA adapter to remove
    /// - Returns: true if the adapter was successfully removed, false otherwise
    func removeLoRAAdapter(_ adapter: SLlamaAdapter) -> Bool {
        guard let context = pointer, let adapterPtr = adapter.cAdapter else { return false }
        return llama_rm_adapter_lora(context, adapterPtr) == 0
    }
    
    /// Remove all LoRA adapters from this context
    func clearLoRAAdapters() {
        guard let context = pointer else { return }
        llama_clear_adapter_lora(context)
    }
    
    /// Load and add a LoRA adapter from file
    /// - Parameter path: Path to the LoRA adapter file
    /// - Returns: The loaded adapter, or nil if loading failed
    func loadLoRAAdapter(from path: String) -> SLlamaAdapter? {
        guard let model = associatedModel else { return nil }
        guard let adapter = SLlamaAdapter(model: model, path: path) else { return nil }
        
        if addLoRAAdapter(adapter) {
            return adapter
        } else {
            return nil
        }
    }
} 