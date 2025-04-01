import Foundation
import llama

class LlamaCppSwiftModel {
    private let model: Model
    private let configuration: Configuration
    private let context: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private var batch: Batch
    private var tokens: [Token]
    private var temporaryInvalidCChars: [CChar] = []
    private var generatedTokenAccount: Int32 = 0
    private var ended = false
    private let n_len: Int32 = 1024

    var shouldContinue: Bool {
        generatedTokenAccount < configuration.maxTokenCount && !ended
    }

    init(path: String, configuration: Configuration = .init()) throws {
        self.configuration = configuration
        llama_backend_init()
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED)
        var model_params = llama_model_default_params()
        #if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        #endif
        guard let model = llama_model_load_from_file(path, model_params) else {
            throw SwiftLlamaError.others("Cannot load model at path \(path)")
        }
        self.model = model
        guard let context = llama_init_from_model(model, configuration.contextParameters) else {
            throw SwiftLlamaError.others("Cannot load model context")
        }
        self.context = context
        self.tokens = []
        self.batch = llama_batch_init(Int32(configuration.batchSize * Configuration.historySize * 2), 0, 1)
        self.sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(configuration.temperature))
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(configuration.topK)))
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(configuration.topP, 1))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234))
        try checkContextLength(context: context, model: model)
    }

    private func checkContextLength(context: Context, model: Model) throws {
        let n_ctx = llama_n_ctx(context)
        let n_ctx_train = llama_model_n_ctx_train(model)
        if n_ctx > n_ctx_train {
            throw SwiftLlamaError.others("Model was trained on \(n_ctx_train) context but tokens \(n_ctx) specified")
        }
    }

    func start(for prompt: Prompt) throws {
        ended = false
        tokens = tokenize(text: prompt.prompt, addBos: true)
        temporaryInvalidCChars = []
        batch.clear()

        tokens.enumerated().forEach { index, token in
            batch.add(token: token, position: Int32(index), seqIDs: [0], logit: false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        generatedTokenAccount = batch.n_tokens
    }
    
    func rawStart(for prompt: String) throws {
        ended = false
        tokens = tokenize(text: prompt, addBos: true)
        temporaryInvalidCChars = []
        batch.clear()

        tokens.enumerated().forEach { index, token in
            batch.add(token: token, position: Int32(index), seqIDs: [0], logit: false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        generatedTokenAccount = batch.n_tokens
    }

    func `continue`() throws -> String {
        let newToken =  llama_sampler_sample(sampler, context, batch.n_tokens - 1)
        let vocab = llama_model_get_vocab(model)
        
        if llama_vocab_is_eog(vocab, newToken) || generatedTokenAccount == n_len {
            temporaryInvalidCChars.removeAll()
            ended = true
            return ""
        }


        let newTokenCChars = tokenToCChars(token: newToken)
        temporaryInvalidCChars.append(contentsOf: newTokenCChars)

        let newTokenStr: String
        if let validString = String(cString: temporaryInvalidCChars + [0], encoding: .utf8) {
            newTokenStr = validString
            temporaryInvalidCChars.removeAll()
        } else if let suffixIndex = temporaryInvalidCChars.firstIndex(where: { $0 != 0 }),
                  let validSuffix = String(cString: Array(temporaryInvalidCChars.suffix(from: suffixIndex)) + [0],
                                            encoding: .utf8) {
            newTokenStr = validSuffix
            temporaryInvalidCChars.removeAll()
        } else {
            newTokenStr = ""
        }

        batch.clear()
        batch.add(token: newToken, position: generatedTokenAccount, seqIDs: [0], logit: true)
        generatedTokenAccount += 1

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        return newTokenStr
    }

    private func tokenToCChars(token: llama_token) -> [CChar] {
        var length: Int32 = 8
        var piece = Array<CChar>(repeating: 0, count: Int(length))
        let vocab = llama_model_get_vocab(model)

        let nTokens = llama_token_to_piece(vocab, token, &piece, length, 0, false)
        if nTokens >= 0 {
            return Array(piece.prefix(Int(nTokens)))
        } else {
            length = -nTokens
            piece = Array<CChar>(repeating: 0, count: Int(length))
            let nNewTokens = llama_token_to_piece(vocab, token, &piece, length, 0, false)
            return Array(piece.prefix(Int(nNewTokens)))
        }
    }

    func tokenize(text: String, addBos: Bool) -> [Token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (addBos ? 1 : 0) + 1
        let vocab = llama_model_get_vocab(model)
        
        return Array(unsafeUninitializedCapacity: n_tokens) { buffer, initializedCount in
            initializedCount = Int(
                llama_tokenize(vocab, text, Int32(utf8Count), buffer.baseAddress, Int32(n_tokens), addBos, false)
            )
        }
    }

    func clear() {
        tokens.removeAll()
        temporaryInvalidCChars.removeAll()
        llama_kv_self_clear(context)
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }
}
