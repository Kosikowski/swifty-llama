import Foundation
import Testing
@testable import SwiftyLlamaCpp

class DummyContext: LlamaContext {
    override init?(model: LlamaModel, contextParams: LlamaContextParams? = nil) {
        super.init(model: model, contextParams: contextParams)
    }
    
    override var pointer: LlamaContextPointer? { nil }
    override var associatedModel: LlamaModel? { nil }
} 