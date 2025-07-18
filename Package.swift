// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftyLlamaCpp",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .watchOS(.v9),
        .tvOS(.v16),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "SwiftyLlamaCpp",
            targets: ["SwiftyLlamaCpp"]
        ),
    ],
    dependencies: [
        // Add any dependencies here if needed
    ],
    targets: [
        .target(
            name: "SwiftyLlamaCpp",
            dependencies: ["llama"],
            path: "Sources/SwiftyLlamaCpp"
        ),
        .binaryTarget(
            name: "llama",
            path: "Sources/llama.cpp/llama.xcframework"
        ),
        .testTarget(
            name: "SwiftyLlamaCppTests",
            dependencies: ["SwiftyLlamaCpp"],
            path: "Tests/SwiftyLlamaCppTests",
            cSettings: [
                .define("LLAMA_LOG_LEVEL", to: "0")
            ]
        ),
    ]
)
