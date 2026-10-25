#!/usr/bin/env nu
# Optional live proof: only this public technical query is sent to Exa.
def main [output: path = '.workspaces/mcp-tools-20261025/exa-reproduce'] {
    mkdir $output
    let endpoint = 'https://mcp.exa.ai/mcp'
    let accept = ['--header' 'Content-Type: application/json' '--header' 'Accept: application/json, text/event-stream']
    let initialize = {
        jsonrpc: '2.0', id: 1, method: initialize,
        params: {protocolVersion: '2025-06-18', capabilities: {}, clientInfo: {name: ds4-readonly-probe, version: '0.1'}}
    }
    $initialize | to json | save --force ($output | path join '01-initialize.request.json')
    let status = ^/usr/bin/curl --silent --show-error --max-time 45 --request POST ...$accept --data-binary ('@' + ($output | path join '01-initialize.request.json')) --dump-header ($output | path join '01-initialize.headers') --output ($output | path join '01-initialize.body') --write-out '%{http_code}' $endpoint
    if $status != '200' { error make {msg: $'Exa initialize returned HTTP($status); inspect raw body. No retry performed.'} }
    let session = open --raw ($output | path join '01-initialize.headers') | lines | parse --regex '(?i)^mcp-session-id:\s*(?<value>.+)$' | first | get value | str trim
    let headers = $accept | append ['--header' 'MCP-Protocol-Version: 2025-06-18' '--header' $'Mcp-Session-Id: ($session)']
    let steps = [
        {name: '02-initialized', expected: '202', body: {jsonrpc: '2.0', method: 'notifications/initialized'}},
        {name: '03-tools-list', expected: '200', body: {jsonrpc: '2.0', id: 2, method: 'tools/list', params: {}}},
        {name: '04-search', expected: '200', body: {jsonrpc: '2.0', id: 3, method: 'tools/call', params: {
            name: web_search_exa, arguments: {
                query: 'Official Model Context Protocol Streamable HTTP documentation describing session IDs and MCP-Protocol-Version headers',
                objective: 'Return official modelcontextprotocol.io specification pages about Streamable HTTP session IDs and protocol-version request headers; exclude third-party tutorials.',
                numResults: 2
            }
        }}},
        {name: '05-fetch', expected: '200', body: {jsonrpc: '2.0', id: 4, method: 'tools/call', params: {
            name: web_fetch_exa, arguments: {urls: ['https://modelcontextprotocol.io/specification/2025-06-18/basic/transports'], maxCharacters: 1800}
        }}}
    ]
    for step in $steps {
        let prefix = $output | path join $step.name
        $step.body | to json | save --force ($prefix + '.request.json')
        let status = ^/usr/bin/curl --silent --show-error --max-time 45 --request POST ...$headers --data-binary ('@' + $prefix + '.request.json') --dump-header ($prefix + '.headers') --output ($prefix + '.body') --write-out '%{http_code}' $endpoint
        if $status != $step.expected { error make {msg: $'($step.name) returned HTTP($status); stopped without retry.'} }
        if $status == '200' {
            let messages = open --raw ($prefix + '.body') | lines | where {|line| $line | str starts-with 'data:' } | each {|line| $line | str substring 5.. | str trim | from json }
            $messages | to json --indent 2 | save --force ($prefix + '.json')
            if ($messages | any {|message| ('error' in $message) or ($message.result?.isError? | default false) }) {
                error make {msg: $'($step.name) returned an MCP error; inspect raw output.'}
            }
        }
        print {step: $step.name, status: $status}
    }
}
