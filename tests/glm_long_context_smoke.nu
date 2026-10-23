#!/usr/bin/env nu
# Optional GLM long-context corruption gate. Run explicitly under pressure_run.py.
# DS4_GLM_LONG_CONTEXT_REPEATS may contain several whitespace-separated sizes.
# Pass additional backend options after MODEL; these preserve argument boundaries.
const ROOT = path self | path dirname | path dirname

def prompt [count: int] {
    let padding = (0..<$count | each {|i|
        let id = $i | into string | fill --width 4 --alignment right --character '0'
        r#'/* audit {{id}}: ggrep searches mmap-backed files with a Boyer-Moore-Horspool skip table. */
static unsigned ggrep_shift_{{id}}(unsigned char c) { return c == (unsigned char)0 ? 1u : 7u; }
static int ggrep_line_{{id}}(const char *p, const char *e) { return p < e && *p != \n\; }
/* note {{id}}: include spellings must remain stdio.h, stdint.h, string.h, errno.h, fcntl.h, unistd.h. */'# | str replace --all '{{id}}' $id
    } | str join (char nl))
    [
        'You are completing a C source file. Return only source text, no prose.

The following audit notes are context padding. Preserve exact code identifiers,
headers, punctuation, and spelling when you later continue the source prefix.'
        $padding
        ((if $count > 0 { char nl } else { '' }) + 'Now continue exactly after the final character of this C source prefix.
The first output character must be ">".

/*
 * ggrep - a small, fast grep.
 *
 * Speed techniques:
 *   - mmap whole regular files and search the page cache directly
 *   - Boyer-Moore-Horspool with a 256-entry bad-character shift table
 *   - line-spanning reporting: on a hit, expand to the enclosing newlines
 *   - one growable output buffer flushed with a single write()
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h
')
    ] | str join (char nl)
}

def last-value [text: string, pattern: string] {
    let values = $text | parse -r $pattern
    if ($values | is-empty) {
        'unknown'
    } else {
        $values | last | get value
    }
}

def --wrapped main [model?: path, ...args: string] {
    let binary = $env.DS4_BIN? | default './ds4'
    let model = $model | default ($env.DS4_GLM_MODEL? | default 'models/GLM-5.2-UD-Q4_K_XL.gguf')
    let ctx = $env.DS4_GLM_LONG_CONTEXT_CTX? | default '100000' | into int
    let counts = (
        $env.DS4_GLM_LONG_CONTEXT_REPEATS?
        | default '130'
        | split row -r '\s+'
        | into int
    )
    let generate = $env.DS4_GLM_LONG_CONTEXT_GEN? | default '32' | into int
    let backend = $env.DS4_GLM_BACKEND? | default 'metal'
    if $backend not-in ['metal' 'cuda' 'cpu'] { error make {msg: $'invalid DS4_GLM_BACKEND: ($backend)'} }
    # Compatibility for the old whitespace-split environment option.
    let legacy = (
        $env.DS4_GLM_EXTRA_ARGS?
        | default ''
        | split row -r '\s+'
        | where {|arg| $arg != '' }
    )
    let parent = $ROOT | path join '.workspaces'
    mkdir $parent
    let output = (mktemp -d -p $parent 'glm-long-context.XXXXXX')
    print $'artifacts: ($output)'
    for count in $counts {
        let stem = $output | path join $'repeat-($count)'
        let prompt_file = $'($stem).prompt.txt'
        prompt $count | save --raw $prompt_file
        print $'glm-long-context-smoke: repeat=($count) ctx=($ctx) gen=($generate)'
        let result = (
            do { ^$binary -m $model $'--($backend)' ...$legacy ...$args --ctx $ctx --nothink --temp 0 -n $generate --prompt-file $prompt_file }
            | complete
        )
        $result.stdout | save --raw $'($stem).stdout'
        $result.stderr | save --raw $'($stem).stderr'
        if $result.exit_code != 0 {
            print --stderr $result.stderr
            exit $result.exit_code
        }
        let generated = $result.stdout | str replace --all --regex '(?m)^ds4: GPU config: [^\n]*\n?' ''
        $generated | save --raw $'($stem).generated.txt'
        let tokens = (last-value $result.stderr 'processing (?<value>[0-9]+) input tokens')
        if not ($generated | str starts-with '>') {
            error make {
                msg: $"glm-long-context-smoke: FAIL repeat=($count) tokens=($tokens): first byte was not '>'; see ($stem).stdout"
            }
        }
        if $generated =~ 'unistdFlush|stdiint|stdintFlush|stdioFlush|errnoFlush' {
            error make {msg: $'glm-long-context-smoke: FAIL repeat=($count) tokens=($tokens): known corruption marker found'}
        }
        let perf = (last-value $result.stderr '(?m)^ds4: prefill: (?<value>[^\n]*)')
        print $'glm-long-context-smoke: PASS repeat=($count) tokens=($tokens) ($perf)'
    }
}
