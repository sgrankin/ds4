#!/usr/bin/env nu

# Fresh uncached requests against one already-loaded model. This measures HTTP
# wall time, including server preparation and head, rather than kernel time.
def main [url: string, requests: path, output: path, --repeats: int = 8] {
    let cases = open --raw $requests | lines | where {|line| $line | str trim | is-not-empty} | each {|line| $line | from json}
    mut rows = []
    for case in ($cases | enumerate) {
        http post --content-type application/json $"($url)/v1/systemone" $case.item | ignore
        mut timings = []
        mut response = {}
        mut answers = []
        for _ in 0..<$repeats {
            let start = date now
            $response = http post --content-type application/json $"($url)/v1/systemone" $case.item
            let seconds = ((date now) - $start | into int) / 1e9
            $timings = $timings | append $seconds
            $answers = $answers | append $response.answers
        }
        $rows = $rows | append {
            case: $case.index,
            tokens: $response.usage.input_tokens,
            question_count: ($case.item.questions | columns | length),
            median_seconds: ($timings | math median),
            seconds: $timings,
            answers: $answers,
            response: $response
        }
    }
    mkdir ($output | path dirname)
    $rows | to json --indent 2 | save --force $output
    $rows | select case tokens question_count median_seconds
}
