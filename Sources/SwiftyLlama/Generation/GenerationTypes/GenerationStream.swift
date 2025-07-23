import Foundation

/// A stream of generated text tokens
public struct GenerationStream {
    public let id: SwiftyLlamaID
    public let stream: AsyncThrowingStream<String, Error>

    public init(id: SwiftyLlamaID, stream: AsyncThrowingStream<String, Error>) {
        self.id = id
        self.stream = stream
    }
}
