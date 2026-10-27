#!/usr/bin/env nu
# HTTP-only: run against an existing qualified server; lifecycle stays external.
use std/assert

def post [base: string, output: path, name: string, body: record] {
    $body | to json | save ($output | path join $'($name).request.json')
    let start = date now
    let reply = http post --full --raw --allow-errors --max-time 300sec --content-type application/json $'($base)/v1/responses' $body
    let seconds = (((date now) - $start) | into int) / 1e9
    $reply.body | save ($output | path join $'($name).response.json')
    assert equal $reply.status 200
    {seconds: $seconds, http_status: $reply.status, response: ($reply.body | from json)}
}

def main [base: string, request_dir: path, output: path] {
    mkdir $output
    let switch = open ($request_dir | path join r0-T0.0-schema-answer.request.json)
    let request = open ($request_dir | path join r0-T0.0-ordinary.request.json)
    assert equal $switch.reasoning.effort high
    assert equal $request.reasoning.effort high
    mut results = []
    for cap in [32 96] {
        # Switch away from the measured ordinary context before each restore.
        # Its request/response is retained, but its latency is excluded.
        post $base $output $'switch-($cap)' $switch | ignore
        let result = post $base $output $'cap-($cap)' ($request | update max_output_tokens $cap)
        let measurement = {
            cap: $cap, seconds: $result.seconds, http_status: $result.http_status,
            status: $result.response.status, usage: $result.response.usage
        }
        $results = ($results | append $measurement)
        $results | to json | save --force ($output | path join results.json)
        print $measurement
        assert ($result.response.status in [completed incomplete])
        assert ($result.response.usage.input_tokens_details.cached_tokens > 0)
        assert equal $result.response.usage.output_tokens $cap
        # Keep complete-output quality separate from these cached cap timings.
    }
}
