import Foundation
import SLlama
import TestUtilities
import XCTest
@testable import SwiftyLlama

@SwiftyLlamaActor
final class SwiftyCoreLlamaTests: XCTestCase, @unchecked Sendable {
    // MARK: - Test Properties

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Skip if model not available
        guard TestUtilities.isTestModelAvailable() else {
            throw XCTSkip("Test model not available")
        }
    }

    override func tearDown() async throws {
        try await super.tearDown()
    }

    // MARK: - Compilation Tests

    func testCompilation() async throws {
        // Just test that we can reference the type
        let _: SwiftyCoreLlama.Type = SwiftyCoreLlama.self

        XCTAssertTrue(true)
    }

    // MARK: - Initialization Tests

    func testInitialization() async throws {
        // Create SwiftyCoreLlama with real model
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        XCTAssertNotNil(swiftyCore)

        // Test model info
        let modelInfo = swiftyCore.modelInfo
        XCTAssertFalse(modelInfo.name.isEmpty)
        XCTAssertGreaterThan(modelInfo.contextSize, 0)

        // Test vocab info
        let vocabInfo = swiftyCore.vocabInfo
        XCTAssertGreaterThan(vocabInfo.size, 0)
    }

    // MARK: - Basic Functionality Tests

    func testStartGeneration() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let prompt = "Hello"
        let params = GenerationParams(temperature: 0.7, maxTokens: 5)

        // Start generation
        let stream = await swiftyCore.start(prompt: prompt, params: params)

        // Verify stream was created
        XCTAssertNotNil(stream)
        XCTAssertNotNil(stream.id)
        XCTAssertNotNil(stream.stream)

        // Verify generation info exists
        let info = await swiftyCore.getGenerationInfo(stream.id)
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.params, params)

        // Cancel to clean up
        await swiftyCore.cancel(stream.id)
    }

    // MARK: - Multiple Generations Tests

    func testMultipleGenerations() async throws {
        let swiftyCore = try SwiftyCoreLlama(modelPath: TestUtilities.testModelPath)

        let prompts = ["Hello", "How are", "The weather"]
        let params = GenerationParams(temperature: 0.7, maxTokens: 3)

        // Start multiple generations concurrently
        let stream1 = await swiftyCore.start(prompt: prompts[0], params: params)
        let stream2 = await swiftyCore.start(prompt: prompts[1], params: params)
        let stream3 = await swiftyCore.start(prompt: prompts[2], params: params)

        // Verify all streams were created
        XCTAssertNotNil(stream1)
        XCTAssertNotNil(stream2)
        XCTAssertNotNil(stream3)

        // Verify all generation IDs are different
        XCTAssertNotEqual(stream1.id, stream2.id)
        XCTAssertNotEqual(stream2.id, stream3.id)
        XCTAssertNotEqual(stream1.id, stream3.id)

        // Verify all generation info exists
        let info1 = await swiftyCore.getGenerationInfo(stream1.id)
        let info2 = await swiftyCore.getGenerationInfo(stream2.id)
        let info3 = await swiftyCore.getGenerationInfo(stream3.id)

        XCTAssertNotNil(info1)
        XCTAssertNotNil(info2)
        XCTAssertNotNil(info3)

        // Verify active generation IDs
        let activeIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertEqual(activeIDs.count, 3)
        XCTAssertTrue(activeIDs.contains(stream1.id))
        XCTAssertTrue(activeIDs.contains(stream2.id))
        XCTAssertTrue(activeIDs.contains(stream3.id))

        // Cancel all to clean up
        await swiftyCore.cancelAll()

        // Verify all are cancelled
        let finalActiveIDs = await swiftyCore.getActiveGenerationIDs()
        XCTAssertEqual(finalActiveIDs.count, 0)
    }
}
