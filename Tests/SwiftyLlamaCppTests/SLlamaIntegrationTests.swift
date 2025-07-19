import Foundation
import Testing
import SwiftyLlamaCpp

@testable import SwiftyLlamaCpp

struct SLlamaIntegrationTests {

    @Test("Real model loading test")
    func testRealModelLoading() async throws {
        do {
            // Use the existing model file
            let modelPath = "Tests/Models/tinystories-gpt-0.1-3m.fp16.gguf"
            
            guard FileManager.default.fileExists(atPath: modelPath) else {
                throw TestError.modelNotFound
            }
            
            // Try to load the model
            guard let model = SLlamaModel(modelPath: modelPath) else {
                throw TestError.modelLoadFailed
            }
            
            // Try to create a context
            guard let context = SLlamaContext(model: model) else {
                throw TestError.contextCreationFailed
            }
            
            // Test model properties
            #expect(model.size > 0, "Model size should be positive")
            #expect(model.parameters > 0, "Model parameters should be positive")
            #expect(model.embeddingDimensions > 0, "Model embedding dimensions should be positive")
            #expect(model.layers > 0, "Model layers should be positive")
            #expect(model.attentionHeads > 0, "Model attention heads should be positive")
            
            // Test vocabulary
            guard let vocabPointer = model.vocab else {
                throw TestError.vocabLoadFailed
            }
            
            let vocab = SLlamaVocab(vocab: vocabPointer)
            #expect(vocab.tokenCount > 0, "Vocabulary should have tokens")
            
            // Test tokenization
            let testText = "Hello, world!"
            guard let tokens = vocab.tokenize(text: testText) else {
                throw TestError.tokenizationFailed
            }
            
            #expect(tokens.count > 0, "Tokenization should produce tokens")
            
            // Test detokenization
            guard let detokenizedText = vocab.detokenize(tokens: tokens) else {
                throw TestError.detokenizationFailed
            }
            
            #expect(detokenizedText == testText, "Detokenization should preserve original text")
            
            // Test basic sampling
            let sampler = SLlamaSampler(context: context)
            let _ = sampler.sample()
            // Note: token might be nil for empty context, which is expected behavior
            // The sampler needs tokens in the context to produce meaningful output
            
        } catch {
            throw error
        }
    }
}

enum TestError: Error {
    case modelNotFound
    case modelLoadFailed
    case contextCreationFailed
    case vocabLoadFailed
    case tokenizationFailed
    case detokenizationFailed
} 