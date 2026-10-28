#!/usr/bin/env nu
# Fixtures use the same frozen shader sources as the measured model binary.
def --wrapped main [binary: path, ...arguments: string] {
    let binary = $binary | path expand
    let shaders = $binary | path dirname | path dirname | path join shaders
    if not ($shaders | path exists) {
        error make {msg: $'Missing frozen shader directory: ($shaders)'}
    }
    mut settings = {}
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $settings = ($settings | upsert $key $source)
    }
    load-env $settings
    exec $binary ...$arguments
}
