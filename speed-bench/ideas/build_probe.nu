#!/usr/bin/env nu
# C-era diagnostic control: requires an archived checkout's compiled C objects.
# The current Zig build is driven by tools/zig-build instead.
const ROOT = path self ../..

def --wrapped main [...args: string] {
    if ($args | get --optional 0 | default '') in ['-h' '--help'] {
        print 'Usage: build_probe.nu C_CHECKOUT [OUTPUT [POLICY_FLAGS [SOURCE [CORE_OBJECT]]]]'
        return
    }
    let runtime = $args | get --optional 0 | default --empty '.'
    let output = $args | get --optional 1 | default --empty 'speed-bench/ideas/quality-probe'
    let flags = (
        $args | get --optional 2 | default ''
        | split row --regex '\s+'
        | where {|flag| $flag != '' }
    )
    let source = $args | get --optional 3 | default --empty ($ROOT | path join speed-bench/ideas/quality_probe.c)
    let core = $args | get --optional 4 | default --empty ($runtime | path join ds4.o)
    let objects = [
        ds4_image.o
        ds4_distributed.o
        ds4_tp.o
        ds4_ssd.o
        ds4_metal.o
        ds4_layer_pack.o
        ds4_engram.o
    ]
    let objects = $objects | each {|name| $runtime | path join $name }
    exec cc -O3 -Wall -Wextra -std=c11 ...$flags -I $runtime $source $core ...$objects -lm -pthread -framework Foundation -framework Metal -o $output
}
