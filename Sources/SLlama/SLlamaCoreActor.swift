

/// A global actor providing isolation for operations related to the SLlama module.
@globalActor
public actor SLlamaActor {
    public static let shared = SLlamaActor()
}
