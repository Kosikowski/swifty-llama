// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SLlama",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .watchOS(.v9),
        .tvOS(.v16),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "SwiftyLlama",
            targets: ["SwiftyLlama"]
        ),
        .library(
            name: "Omen",
            targets: ["Omen"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-atomics.git", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "Omen",
            dependencies: [],
            path: "Sources/Omen",
            exclude: ["README.md"]
        ),
        .target(
            name: "SLlama",
            dependencies: ["llama", "Omen"],
            path: "Sources/SLlama"
        ),
        .target(
            name: "TestUtilities",
            path: "Sources/TestUtilities",
            resources: [
                .copy("../../Models/tinystories-gpt-0.1-3m.fp16.gguf"),
            ]
        ),
        .target(
            name: "SwiftyLlama",
            dependencies: ["SLlama", "Omen", .product(name: "Atomics", package: "swift-atomics")],
            path: "Sources/SwiftyLlama"
        ),
        .binaryTarget(
            name: "llama",
            path: "Sources/llama.cpp/llama.xcframework"
        ),
        .testTarget(
            name: "SLlamaTests",
            dependencies: ["SLlama", "TestUtilities"],
            path: "Tests/SLlamaTests",
//            resources: [
//                .copy("../../Models/tinystories-gpt-0.1-3m.fp16.gguf"),
//            ],
            cSettings: [
                .define("LLAMA_LOG_LEVEL", to: "0"),
            ]
        ),
        .testTarget(
            name: "SwiftyLlamaTests",
            dependencies: ["SwiftyLlama", "SLlama", "TestUtilities"],
            path: "Tests/SwiftyLlamaTests",
//            resources: [
//                .copy("../../Models/tinystories-gpt-0.1-3m.fp16.gguf"),
//            ],
            cSettings: [
                .define("LLAMA_LOG_LEVEL", to: "0"),
            ]
        ),
    ]
)
