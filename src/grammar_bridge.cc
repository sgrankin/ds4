// Native XGrammar boundary. Serving owns model/request policy in Zig; this file
// only translates byte vocabularies, immutable grammars and per-generation state.
#include "grammar.h"
#include "grammar_xml.h"

#include <xgrammar/xgrammar.h>

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct ds4_grammar_compiler {
    xgrammar::GrammarCompiler value;
    size_t vocab_size;
    ds4::GrammarControls controls;
};
struct ds4_grammar_compiled {
    xgrammar::CompiledGrammar value;
    size_t vocab_size;
};
struct ds4_grammar_matcher {
    xgrammar::GrammarMatcher value;
    size_t vocab_size;
};

template <class Result, class Function>
static Result checked(Result failure, char *error, size_t size, Function function) {
    if (error && size) error[0] = '\0';
    try {
        return function();
    } catch (const std::exception &e) {
        if (error && size) std::snprintf(error, size, "%s", e.what());
    } catch (...) {
        if (error && size) std::snprintf(error, size, "unknown XGrammar error");
    }
    return failure;
}

extern "C" ds4_grammar_compiler *ds4_grammar_compiler_new(
    const ds4_grammar_piece *pieces, size_t vocab_size,
    const int32_t *stops, size_t stop_count, char *error, size_t error_size) {
    return ds4_grammar_compiler_new_token_aware(pieces, vocab_size, stops, stop_count,
                                               nullptr, 0, error, error_size);
}

extern "C" ds4_grammar_compiler *ds4_grammar_compiler_new_token_aware(
    const ds4_grammar_piece *pieces, size_t vocab_size,
    const int32_t *stops, size_t stop_count,
    const int32_t *controls, size_t control_count, char *error, size_t error_size) {
    return checked<ds4_grammar_compiler *>(nullptr, error, error_size, [&] {
        if (!pieces || !vocab_size || vocab_size > size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("invalid grammar vocabulary size");
        std::vector<std::string> vocabulary;
        vocabulary.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            vocabulary.emplace_back(pieces[i].len ? reinterpret_cast<const char *>(pieces[i].bytes) : "", pieces[i].len);
        }
        ds4::GrammarControls control_tokens;
        for (size_t i = 0; i < control_count; ++i) {
            const int32_t id = controls[i];
            if (id < 0 || size_t(id) >= vocab_size)
                throw std::invalid_argument("grammar control token is outside the vocabulary");
            control_tokens.emplace(vocabulary[id], id);
            // Empty pieces are rejected before native token matching. A nonempty
            // invalid-UTF8 prefix instead keeps explicit Token(id) usable while
            // preventing a control ID from satisfying ordinary text terminals.
            vocabulary[id] = std::string(1, char(0xff)) + "DS4" + std::to_string(id);
        }
        std::vector<int32_t> stop_ids;
        if (stop_count) stop_ids.assign(stops, stops + stop_count);
        for (int32_t id : stop_ids) {
            if (id < 0 || size_t(id) >= vocab_size)
                throw std::invalid_argument("grammar stop token is outside the vocabulary");
        }
        xgrammar::TokenizerInfo tokenizer(vocabulary, xgrammar::VocabType::RAW,
                                          int(vocab_size), stop_ids, false);
        return new ds4_grammar_compiler{
            xgrammar::GrammarCompiler(tokenizer, 2, true, 128 * 1024 * 1024), vocab_size,
            std::move(control_tokens)};
    });
}

extern "C" void ds4_grammar_compiler_free(ds4_grammar_compiler *compiler) { delete compiler; }

extern "C" ds4_grammar_compiled *ds4_grammar_compile(
    ds4_grammar_compiler *compiler, ds4_grammar_kind kind,
    const char *source, size_t source_len, bool strict_json,
    char *error, size_t error_size) {
    return checked<ds4_grammar_compiled *>(nullptr, error, error_size, [&] {
        const std::string text(source_len ? source : "", source_len);
        auto grammar = [&] {
            switch (kind) {
            case DS4_GRAMMAR_STRUCTURAL:
                return compiler->value.CompileStructuralTag(ds4::tokenAwareStructuralTag(text, compiler->controls));
            case DS4_GRAMMAR_JSON_SCHEMA:
                return compiler->value.CompileJSONSchema(text, true, std::nullopt,
                                                         std::nullopt, strict_json);
            case DS4_GRAMMAR_REGEX:
                return compiler->value.CompileRegex(text);
            }
            throw std::invalid_argument("unknown grammar kind");
        }();
        return new ds4_grammar_compiled{std::move(grammar), compiler->vocab_size};
    });
}

extern "C" void ds4_grammar_compiled_free(ds4_grammar_compiled *compiled) { delete compiled; }
extern "C" size_t ds4_grammar_compiled_vocab_size(const ds4_grammar_compiled *compiled) {
    return compiled->vocab_size;
}

extern "C" ds4_grammar_matcher *ds4_grammar_matcher_new(
    const ds4_grammar_compiled *compiled, char *error, size_t error_size) {
    return checked<ds4_grammar_matcher *>(nullptr, error, error_size, [&] {
        return new ds4_grammar_matcher{xgrammar::GrammarMatcher(compiled->value), compiled->vocab_size};
    });
}
extern "C" void ds4_grammar_matcher_free(ds4_grammar_matcher *matcher) { delete matcher; }

extern "C" int ds4_grammar_accept(ds4_grammar_matcher *matcher, int32_t token,
                                  char *error, size_t error_size) {
    return checked(-1, error, error_size, [&] {
        return matcher->value.AcceptToken(token) ? 1 : 0;
    });
}

extern "C" int ds4_grammar_fill(ds4_grammar_matcher *matcher, uint32_t *words,
                                size_t word_count, char *error, size_t error_size) {
    return checked(-1, error, error_size, [&] {
        if (word_count != size_t(xgrammar::GetBitmaskSize(int(matcher->vocab_size))))
            throw std::invalid_argument("grammar mask has the wrong word count");
        int64_t shape = int64_t(word_count);
        DLTensor tensor{};
        tensor.data = words;
        tensor.device = DLDevice{kDLCPU, 0};
        tensor.ndim = 1;
        tensor.dtype = xgrammar::GetBitmaskDLType();
        tensor.shape = &shape;
        return matcher->value.FillNextTokenBitmask(&tensor) ? 1 : 0;
    });
}

extern "C" bool ds4_grammar_rollback(ds4_grammar_matcher *matcher, size_t tokens,
                                     char *error, size_t error_size) {
    return checked(false, error, error_size, [&] {
        if (tokens > size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("grammar rollback exceeds native token count");
        matcher->value.Rollback(int(tokens));
        return true;
    });
}
extern "C" bool ds4_grammar_completed(const ds4_grammar_matcher *matcher) {
    return matcher->value.IsCompleted();
}
extern "C" bool ds4_grammar_terminated(const ds4_grammar_matcher *matcher) {
    return matcher->value.IsTerminated();
}
