// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "LlamaCppSwift",
    platforms: [
        .macOS(.v15),
        .iOS(.v16),
        .watchOS(.v11),
        .tvOS(.v18),
        .visionOS(.v2)
    ],
    products: [
        .library(name: "LlamaCppSwift", targets: ["LlamaCppSwift"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "LlamaCppSwift",
                dependencies: [
                    .target(name: "llama")
                ]),
        .binaryTarget(name: "llama", path: "Frameworks/llama.xcframework")
    ]
)
