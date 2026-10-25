#!/usr/bin/env nu
# Run against a loaded Clef-Flash helper with prefix reuse enabled.
use std/assert

def main [url: string, inputs: path, output: path] {
    mkdir $output
    let requests = open --raw ($inputs | path join requests.jsonl) | lines | each { from json }
    for row in [[index expected]; [0 billing] [1 billing] [4 technical] [0 billing]] {
        let body = $requests | get $row.index | update model clef-flash
        let answer = http post --content-type application/json $'($url)/v1/systemone' $body
        $answer | save -f ($output | path join $'request-($row.index)-($row.expected).json')
        let actual = if $row.index == 1 { $answer.answers.owner.choice } else { $answer.answers.team.choice }
        assert equal $actual $row.expected
    }
    # Actual image/video calls must bypass text caching without retiring its
    # independently owned causal state. The following text edit should hit.
    ^nu tests/test_clef_media.nu --url $url --model clef-flash --vision-only --output ($output | path join media)
    if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Media bypass qualification failed'} }
    let body = $requests | get 1 | update model clef-flash
    let answer = http post --content-type application/json $'($url)/v1/systemone' $body
    $answer | save -f ($output | path join after-media.json)
    assert equal $answer.answers.owner.choice billing
    print 'Prefix HTTP: changed questions, changed fact, image/video bypass and returning text passed.'
}
