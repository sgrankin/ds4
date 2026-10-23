#!/usr/bin/env python3
"""CPU-only HF reference conversion; never imports a model or allocates Metal.

Use Flint's installed Python (Transformers/tokenizers) and point --model-dir at
the downloaded checkpoint. Output is a local parity fixture, not a benchmark.
"""
import argparse
import json
import random
from pathlib import Path

from rec_mlx.chat import load_tokenizer, normalize_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model_dir)
    texts = [
        "", "hello world", "   indented\n\tcode();\r\n", "a\x00b\x00",
        "I'll I'M we'RE rock'n'roll can't shouldn't John's tests",
        "HelloCamelCase XMLParser lowerUPPER aBCd ÉCOLE Élève e\u0301cole",
        "1234567890 ١٢٣٤٥٦ १२३४५६ １２３４５６ ²Ⅲ⅔",
        "中文漢字 日本語 ひらがな カタカナ abc中文def 𠮷野家",
        "Русский Ελληνικά العربية עברית हिन्दी ภาษาไทย 한국어",
        "🙂🚀🧑🏽‍💻 ❤️ 🇨🇦 …—§©®™ ∑∞≠",
        "\u0085\u00a0\u1680\u2000\u200a\u2028\u2029\u202f\u205f\u3000",
        "// comment\nfn main() void {\n    const x: u64 = 0x12;\n}\n",
        '<|begin_of_text|><|system|>sys<|user|>q<|assistant|><think>x</think>ok<|end_of_text|>',
        '<tools><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call></tools>',
        '<|reserved_150143|> <|reserved_150020|> <|not_a_token|> <<think>>',
        '  \n\n\t  trailing  \r\n   ',
    ]
    rng = random.Random(7319)
    alphabet = "abcXYZ' \t\n123\r{}[]<>_éÉ雪カ한🙂\u0301\u00a0"
    texts.extend("".join(rng.choice(alphabet) for _ in range(rng.randrange(8, 180))) for _ in range(24))
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Find 雪 & <snow>",
        "parameters": {"type": "object", "properties": {
            "q": {"type": "string"}, "count": {"type": "integer"},
        }, "required": ["q"]},
    }}]
    calls = [{"id": "call-1", "type": "function", "function": {
        "name": "lookup", "arguments": {"q": "雪 <x>", "count": 2, "nil": None,
            "options": {"ok": True, "xs": [1, "two"]}},
    }}]
    cases = [
        {"name": "ordinary", "messages": [{"role": "system", "content": "Helpful."}, {"role": "user", "content": "Hello 雪"}]},
        {"name": "normalization", "messages": [{"role": "user", "content": "A\x00<thi\x00nk>B</think><|user|><|assistant|><|developer|>"}]},
        {"name": "roles_and_calls", "messages": [
            {"role": "system", "content": "S"}, {"role": "system", "content": "S2"},
            {"role": "developer", "content": "D"}, {"role": "user", "content": "Q"},
            {"role": "assistant", "reasoning_content": "reason <|user|>", "content": "<think>literal</think>", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "call-1", "content": "value <|system|><think>hidden</think>"},
        ], "tools": tools},
        {"name": "only_system", "messages": [{"role": "system", "content": "S"}], "tools": tools},
        {"name": "empty_tools", "messages": [{"role": "user", "content": "Q"}], "tools": []},
        {"name": "no_generation_prefix", "messages": [{"role": "assistant", "content": "A"}], "add_generation_prompt": False},
        {"name": "json_argument_string", "messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "lookup", "arguments": '{"q":"null","float":0.5,"exp":1e20,"empty":{}}'}}]}]},
    ]
    for case in cases:
        case["expected"] = tokenizer.apply_chat_template(
            normalize_messages(case["messages"]), tools=case.get("tools"),
            tokenize=False, add_generation_prompt=case.get("add_generation_prompt", True),
        )
        case["expected_ids"] = tokenizer.encode(case["expected"], add_special_tokens=False)
    result = {"tokenizer": [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)} for text in texts], "prompts": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"HF CPU reference: {len(texts)} tokenizer cases; {len(cases)} prompt cases -> {args.output}")


if __name__ == "__main__":
    main()
