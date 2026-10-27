#!/usr/bin/env nu
# Preserve actual baseline reasoning and answer token IDs; the probe adds EOS.
def main [answers: path, output: path] {
    mkdir $output
    open --raw $answers | lines | each {|line|
        let answer = ($line | from json)
        $answer.tokens | each { into string } | str join "\n" | save ($output | path join $'($answer.case).txt')
    } | ignore
}
