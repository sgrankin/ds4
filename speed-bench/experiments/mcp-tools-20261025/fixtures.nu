#!/usr/bin/env nu
# Reuse the earlier policy, notes and oracle; shrink only the favorable queue.
const BASE = path self '../ptc-decisions-20261025/fixtures.nu'

def main [output: path] {
    ^$nu.current-exe $BASE $output | ignore
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    let path = $output | path join 'fixtures.json'
    let cases = open $path | each {|case|
        let records = $case.queue.records | first (if $case.name == 'ordinary' { 4 } else { 8 })
        let ids = $records | get id
        let active = $case.expected.ids | where {|id| $id in $ids }
        $case | merge {
            queue: {policy: $case.queue.policy, records: $records},
            manifest: {policy: $case.queue.policy, records: ($records | select id charge_cents)},
            expected: {
                ids: $active,
                count: ($active | length),
                max_charge_cents: ($records | where {|row| $row.id in $active } | get charge_cents | math max)
            }
        }
    }
    $cases | to json --indent 2 | save --force $path
}
