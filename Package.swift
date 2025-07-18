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
            path: "Sources/SwiftyLlamaCpp",
            cSettings: [
                .define("GGML_LOG_DISABLE"),
                .define("LLAMA_LOG_DISABLE"),
                .define("GGML_METAL_LOG_DISABLE"),
                .define("LLAMA_KV_CACHE_LOG_DISABLE"),
                .define("LLAMA_CONTEXT_LOG_DISABLE"),
                .define("LLAMA_MODEL_LOADER_LOG_DISABLE"),
                .define("LLAMA_TOKENIZER_LOG_DISABLE"),
                .define("LLAMA_GRAPH_LOG_DISABLE"),
                .define("LLAMA_ADAPTER_LOG_DISABLE"),
                .define("LLAMA_BACKEND_LOG_DISABLE"),
                .define("LLAMA_PRINT_INFO_DISABLE"),
                .define("LLAMA_LOAD_TENSORS_LOG_DISABLE"),
                .define("LLAMA_INIT_TOKENIZER_LOG_DISABLE"),
                .define("LLAMA_LOAD_LOG_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_DISABLE"),
                .define("LLAMA_GGML_BACKEND_LOG_DISABLE"),
                .define("LLAMA_GGML_LOG_DISABLE"),
                .define("LLAMA_GGML_METAL_LOG_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_ALLOCATED_SIZE_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_FREE_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_FREE_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_ALLOC_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_RESET_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_GET_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_SET_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_CLEAR_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_DESTROY_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_INIT_DISABLE"),
                .define("GGML_USE_METAL", to: "0"),
                .define("LLAMA_USE_METAL", to: "0"),
                .define("GGML_METAL_DISABLE"),
                .define("LLAMA_METAL_DISABLE"),
                .define("GGML_BACKEND_METAL_DISABLE"),
                .define("LLAMA_BACKEND_METAL_DISABLE"),
            ]
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
                          .define("GGML_LOG_DISABLE"),
                
                .define("GGML_METAL_LOG_DISABLE"),
                .define("LLAMA_KV_CACHE_LOG_DISABLE"),
                .define("LLAMA_CONTEXT_LOG_DISABLE"),
                .define("LLAMA_MODEL_LOADER_LOG_DISABLE"),
                .define("LLAMA_TOKENIZER_LOG_DISABLE"),
                .define("LLAMA_GRAPH_LOG_DISABLE"),
                .define("LLAMA_ADAPTER_LOG_DISABLE"),
                .define("LLAMA_BACKEND_LOG_DISABLE"),
                .define("LLAMA_PRINT_INFO_DISABLE"),
                .define("LLAMA_LOAD_TENSORS_LOG_DISABLE"),
                .define("LLAMA_INIT_TOKENIZER_LOG_DISABLE"),
                .define("LLAMA_LOAD_LOG_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_DISABLE"),
                .define("LLAMA_GGML_BACKEND_LOG_DISABLE"),
                .define("LLAMA_GGML_LOG_DISABLE"),
                .define("LLAMA_GGML_METAL_LOG_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_ALLOCATED_SIZE_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_FREE_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_FREE_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_ALLOC_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_RESET_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_GET_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_SET_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_CLEAR_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_DESTROY_DISABLE"),
                .define("LLAMA_GGML_BACKEND_METAL_LOG_MEM_POOL_INIT_DISABLE"),
                .define("GGML_USE_METAL", to: "0"),
                .define("LLAMA_USE_METAL", to: "0"),
                .define("GGML_METAL_DISABLE"),
                .define("LLAMA_METAL_DISABLE"),
                .define("GGML_BACKEND_METAL_DISABLE"),
                .define("LLAMA_BACKEND_METAL_DISABLE"),
            ]
        ),
    ]
)
