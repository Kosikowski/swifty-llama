

/// QLoRA configuration
public struct QLoRAConfig: Codable, Sendable {
    public let quantType: String
    public let useDoubleQuant: Bool
    public let computeDtype: String

    public init(
        quantType: String = "nf4",
        useDoubleQuant: Bool = true,
        computeDtype: String = "float16"
    ) {
        self.quantType = quantType
        self.useDoubleQuant = useDoubleQuant
        self.computeDtype = computeDtype
    }
}
