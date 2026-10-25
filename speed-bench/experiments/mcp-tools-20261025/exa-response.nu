#!/usr/bin/env nu
# Explicit live integration proof; natural-task comparisons use the fixed fixture.
def main [
    --url: string = 'http://127.0.0.1:19333',
    --model: string = 'deepseek-v4.1-flash',
    --output: path = '.workspaces/mcp-tools-20261025/exa-response'
] {
    mkdir $output
    let code = "const search = await tools.mcp__exa__web_search_exa({
  query:'Official Model Context Protocol Streamable HTTP documentation describing session IDs and MCP-Protocol-Version headers',
  objective:'Return official modelcontextprotocol.io specification pages about Streamable HTTP session IDs and protocol-version request headers; exclude third-party tutorials.',
  numResults:2
});
const url = 'https://modelcontextprotocol.io/specification/2025-06-18/basic/transports';
const fetched = await tools.mcp__exa__web_fetch_exa({urls:[url],maxCharacters:12000});
const content = result => (result.content || []).filter(c => c.type === 'text').map(c => c.text).join(' ');
const searchText = content(search), fetchText = content(fetched);
const terms = ['Mcp-Session-Id','MCP-Protocol-Version','application/json','text/event-stream'];
const seen = terms.map(term => {
  const index = fetchText.indexOf(term);
  return {term, search:searchText.includes(term), fetch: index >= 0,
    excerpt:index < 0 ? null : fetchText.slice(Math.max(0,index-80),index+200)};
});
text({url,search_excerpt:searchText.slice(0,350),seen});"
    let request = {
        model: $model, temperature: 0, store: false,
        max_output_tokens: 4096, max_tool_calls: 8,
        instructions: ('Run ds4_exec exactly once with this code unchanged. Then return its emitted evidence as JSON; do not run another program.' + "\n" + $code),
        input: 'Check the public official MCP transport documentation and return the compact evidence produced by the supplied program.',
        tools: [
            {type: programmatic_tool_calling},
            {
                type: mcp, server_label: exa, server_url: 'https://mcp.exa.ai/mcp',
                allowed_tools: [web_search_exa web_fetch_exa],
                require_approval: never, allowed_callers: [programmatic]
            }
        ]
    }
    $request | to json --indent 2 | save --force ($output | path join request.json)
    let start = date now
    let raw = http post --raw --allow-errors --max-time 10min --content-type application/json ($url + '/v1/responses') $request
    let elapsed = ((date now) - $start | into int) / 1_000_000_000
    $raw | save --force ($output | path join response.json)
    let response = $raw | from json
    let programs = $response.output | where type == program
    let outputs = $response.output | where type == program_output
    if $response.status != completed or ($programs | length) != 1 or ($outputs | length) != 1 {
        error make {msg: 'Live Exa response did not complete with exactly one program; inspect response.json.'}
    }
    let program = $outputs | first
    let result = $program.result | from json
    let audit = $program.ds4.calls
    $audit | to json --indent 2 | save --force ($output | path join audit.json)
    if $program.status != completed or $result.error != null or ($result.output | length) != 1 {
        error make {msg: 'Live Exa program failed; inspect response.json and audit.json.'}
    }
    if ($audit | length) != 2 or ($audit | any {|call| $call.status != 200 or $call.server_label != exa }) or (($audit | get name | sort) != [web_fetch_exa web_search_exa]) {
        error make {msg: 'Expected exactly two successful Exa MCP calls; inspect audit.json.'}
    }
    let emitted = $result.output | first
    let summary = {
        completed: true, wall_seconds: $elapsed,
        main_input_tokens: $response.usage.input_tokens,
        main_output_tokens: $response.usage.output_tokens,
        main_reasoning_tokens: $response.usage.output_tokens_details.reasoning_tokens?,
        main_cached_tokens: $response.usage.input_tokens_details.cached_tokens?,
        program_calls: ($programs | length), mcp_calls: ($audit | length),
        nested_mcp_ms: ($audit | get elapsed_ms | math sum),
        response_bytes: ($raw | into binary | bytes length),
        program_result_bytes: ($program.result | into binary | bytes length),
        seen: $emitted.seen
    }
    $summary | to json --indent 2 | save --force ($output | path join summary.json)
    # Missing strings are reported, not converted into fabricated evidence.
    $summary
}
