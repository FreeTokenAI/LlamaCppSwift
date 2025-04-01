import Foundation
import llama
import Combine

public class LlamaCppSwift {
    private let model: LlamaCppSwiftModel
    private let configuration: Configuration
    private var contentStarted = false
    private var sessionSupport = false {
        didSet {
            if !sessionSupport {
                session = nil
            }
        }
    }

    private var session: Session?
    private lazy var resultSubject: CurrentValueSubject<String, Error> = {
        .init("")
    }()
    private var generatedTokenCache = ""

    var maxLengthOfStopToken: Int {
        configuration.stopTokens.map { $0.count }.max() ?? 0
    }

    public init(modelPath: String,
                 modelConfiguration: Configuration = .init()) throws {
        self.model = try LlamaCppSwiftModel(path: modelPath, configuration: modelConfiguration)
        self.configuration = modelConfiguration
    }

    private func prepare(sessionSupport: Bool, for prompt: Prompt) -> Prompt {
        contentStarted = false
        generatedTokenCache = ""
        self.sessionSupport = sessionSupport
        if sessionSupport {
            if session == nil {
                session = Session(lastPrompt: prompt)
            } else {
                session?.lastPrompt = prompt
            }
            return session?.sessionPrompt ?? prompt
        } else {
            return prompt
        }
    }
    
    private func rawPrepare(for prompt: String) -> String {
        contentStarted = false
        generatedTokenCache = ""
        self.sessionSupport = false
        return prompt
    }

    private func isStopToken() -> Bool {
        configuration.stopTokens.reduce(false) { partialResult, stopToken in
            generatedTokenCache.hasSuffix(stopToken)
        }
    }

    private func response(for prompt: Prompt, output: (String) -> Void, finish: () -> Void) {
        func finaliseOutput() {
            configuration.stopTokens.forEach {
                generatedTokenCache = generatedTokenCache.replacingOccurrences(of: $0, with: "")
            }
            output(generatedTokenCache)
            finish()
            generatedTokenCache = ""
        }
        defer { model.clear() }
        do {
            try model.start(for: prompt)
            while model.shouldContinue {
                var delta = try model.continue()
                if contentStarted { // remove the prefix empty spaces
                    if needToStop(after: delta, output: output) {
                        finish()
                        break
                    }
                } else {
                    delta = delta.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !delta.isEmpty {
                        contentStarted = true
                        if needToStop(after: delta, output: output) {
                            finish()
                            break
                        }
                    }
                }
            }
            finaliseOutput()
        } catch {
            finaliseOutput()
        }
    }

    private func rawResponse(for prompt: String, output: (String) -> Void, finish: () -> Void) {
        func finaliseOutput() {
            configuration.stopTokens.forEach {
                generatedTokenCache = generatedTokenCache.replacingOccurrences(of: $0, with: "")
            }
            output(generatedTokenCache)
            finish()
            generatedTokenCache = ""
        }
        defer { model.clear() }
        do {
            try model.rawStart(for: prompt)
            while model.shouldContinue {
                var delta = try model.continue()
                if contentStarted { // remove the prefix empty spaces
                    if needToStop(after: delta, output: output) {
                        finish()
                        break
                    }
                } else {
                    delta = delta.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !delta.isEmpty {
                        contentStarted = true
                        if needToStop(after: delta, output: output) {
                            finish()
                            break
                        }
                    }
                }
            }
            finaliseOutput()
        } catch {
            finaliseOutput()
        }
    }
    
    /// Handling logic of StopToken
    private func needToStop(after delta: String, output: (String) -> Void) -> Bool {
        guard maxLengthOfStopToken > 0 else {
            output(delta)
            return false
        }
        generatedTokenCache += delta
        if generatedTokenCache.count >= maxLengthOfStopToken * 2 {
            if let stopToken = configuration.stopTokens.first(where: { generatedTokenCache.contains($0) }),
               let index = generatedTokenCache.range(of: stopToken) {
                let outputCandidate = String(generatedTokenCache[..<index.lowerBound])
                output(outputCandidate)
                generatedTokenCache = ""
                return true
            } else { // no stop token generated
                let outputCandidate = String(generatedTokenCache.prefix(maxLengthOfStopToken))
                generatedTokenCache.removeFirst(outputCandidate.count)
                output(outputCandidate)
                return false
            }
        }
        return false
    }

    @LlamaCppSwiftActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) -> AsyncThrowingStream<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        return .init { continuation in
            Task {
                response(for: sessionPrompt) { [weak self] delta in
                    continuation.yield(delta)
                    self?.session?.response(delta: delta)
                } finish: { [weak self] in
                    continuation.finish()
                    self?.session?.endResponse()
                }
            }
        }
    }
    
    @LlamaCppSwiftActor
    public func rawStart(for prompt: String) -> AsyncThrowingStream<String, Error> {
        _ = rawPrepare(for: prompt)
        return .init { continuation in
            Task {
                rawResponse(for: prompt) { delta in
                    continuation.yield(delta)
                } finish: {
                    continuation.finish()
                }
            }
        }
    }

    @LlamaCppSwiftActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) -> AnyPublisher<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        Task {
            response(for: sessionPrompt) { delta in
                resultSubject.send(delta)
                session?.response(delta: delta)
            } finish: {
                resultSubject.send(completion: .finished)
                session?.endResponse()
            }
        }
        return resultSubject.eraseToAnyPublisher()
    }

    @LlamaCppSwiftActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) async throws -> String {
        var result = ""
        for try await value in start(for: prompt) {
            result += value
        }
        return result
    }
    
    public func tokenCount(_ text: String, addBos: Bool = false) throws -> Int {
        let tokens = model.tokenize(text: text, addBos: addBos)
        return tokens.count
    }
}
