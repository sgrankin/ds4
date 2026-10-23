#!/usr/bin/env nu
# Retained Linux/L40S deployment example; the supported local build is Metal.
const ROOT = path self | path dirname
const SCRIPT = path self

def fail [message: string] {
    print --stderr $"run-nvidia-tp-server: ($message)"
    exit 1
}

def setting [name: string, fallback: string] {
    let value = $env | get --optional $name
    if $value == null or $value == '' { $fallback } else { $value }
}

def timeout [name: string, fallback: string] {
    let value = setting $name $fallback
    if $value !~ '^[1-9][0-9]*$' { fail $"($name) must be a positive integer, got: ($value)" }
    $value | into int
}

def configuration [] {
    let state = $ROOT | path join .workspaces nvidia-server
    {
        model: (
            setting DS4_MODEL '/home/antirez/models/deepseek-v4-gguf/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf'
        )
        ctx: (setting DS4_CTX '100000')
        host: (setting DS4_SERVER_HOST '0.0.0.0')
        port: (setting DS4_SERVER_PORT '8000')
        kv: (setting DS4_KV_DIR '/data/ds4-kv')
        kv_mb: (setting DS4_KV_SPACE_MB '8192')
        sessions: (setting DS4_BATCHED_SESSIONS '16')
        lock: (setting DS4_LOCK_FILE ($state | path join ds4.lock))
        log: (setting DS4_SERVER_LOG ($state | path join server.log))
    }
}

def prerequisites [config: record] {
    let binary = $ROOT | path join ds4-server
    if (^test -x $binary | complete).exit_code != 0 { fail './ds4-server is not built; provide a CUDA-enabled server on this Linux host' }
    if (^test -f $config.model | complete).exit_code != 0 { fail $"model not found: ($config.model)" }
}

def alive [pid: int] {
    let stat = try { open --raw $"/proc/($pid)/stat" } catch { return false }
    # The parenthesized comm field can contain spaces; the following field is state.
    let state = $stat | parse --regex '^\d+ \(.+\) (?<state>\S)' | get --optional 0.state
    $state != null and $state != Z
}

def managed [pid: int] {
    let cwd = ^readlink -f $"/proc/($pid)/cwd" | complete
    if $cwd.exit_code != 0 or ($cwd.stdout | str trim) != $ROOT { return false }
    let argv = try {
        open --raw $"/proc/($pid)/cmdline" | split row (char nul)
    } catch { return false }
    ($argv | first | default '') in [
        './ds4-server'
        ($ROOT | path join ds4-server)
    ] and '--cuda-tensor-parallel' in $argv
}

def lock-owner [config: record] {
    if not ($config.lock | path exists) { return {kind: none, pid: 0} }
    let readable = (^test -r $config.lock | complete).exit_code == 0
    let writable = (^test -w $config.lock | complete).exit_code == 0
    if not $readable or not $writable { fail $"cannot access ($config.lock); fix its ownership before managing the server" }
    let raw = try {
        open --raw $config.lock | lines | get --optional 0 | default '' | str trim
    } catch { fail $"cannot read ($config.lock); check its ownership" }
    if $raw !~ '^[1-9][0-9]*$' { return {kind: none, pid: 0} }
    let pid = $raw | into int
    if not (alive $pid) { return {kind: none, pid: 0} }
    {
        kind: (if (managed $pid) { 'managed' } else { 'other' })
        pid: $pid
    }
}

def require-owned [owner: record, config: record] {
    if $owner.kind == other {
        fail $"($config.lock) is held by pid ($owner.pid), which is not this CUDA-TP server"
    }
}

def stop-server [config: record] {
    let seconds = timeout DS4_STOP_TIMEOUT '120'
    let owner = lock-owner $config
    require-owned $owner $config
    if $owner.kind == none {
        print 'run-nvidia-tp-server: server is not running'
        return
    }
    print $"run-nvidia-tp-server: stopping pid ($owner.pid)"
    ^kill -TERM ($owner.pid | into string)
    if $env.LAST_EXIT_CODE != 0 { fail $"could not signal pid ($owner.pid)" }
    let deadline = (date now) + ($seconds * 1sec)
    while (alive $owner.pid) {
        if (date now) >= $deadline { fail $"pid ($owner.pid) did not stop within ($seconds)s" }
        sleep 1sec
    }
    print 'run-nvidia-tp-server: stopped'
}

def run-server [config: record, args: list<string>] {
    prerequisites $config
    mkdir ($config.lock | path dirname)
    $env.DS4_LOCK_FILE = $config.lock
    exec ($ROOT | path join ds4-server) --cuda --cuda-tensor-parallel --gpu-devices '0,2,4,6,1,3,5,7' --gpu-vram auto --model $config.model --ctx $config.ctx --host $config.host --port $config.port --batched-session $config.sessions --kv-disk-dir $config.kv --kv-disk-space-mb $config.kv_mb ...$args
}

def start-server [config: record, args: list<string>] {
    let seconds = timeout DS4_START_TIMEOUT '180'
    prerequisites $config
    if (which setsid | is-empty) { fail 'setsid is required for detached server startup' }
    let owner = lock-owner $config
    require-owned $owner $config
    if $owner.kind == managed {
        print $"run-nvidia-tp-server: already running \(pid ($owner.pid)\)"
        return
    }
    mkdir ($config.log | path dirname)
    # setsid forks the foreground command; its inherited log FD outlives this Nu process.
    '' | ^setsid -f $SCRIPT foreground ...$args out+err> $config.log
    if $env.LAST_EXIT_CODE != 0 { fail $"detached startup failed; see ($config.log)" }
    print $"run-nvidia-tp-server: starting; log: ($config.log)"
    let deadline = (date now) + ($seconds * 1sec)
    while (date now) < $deadline {
        let owner = lock-owner $config
        require-owned $owner $config
        if $owner.kind == managed and (alive $owner.pid) {
            let ready = try {
                open --raw $config.log | str contains 'ds4-server: listening on '
            } catch {
                false
            }
            if $ready {
                print $"run-nvidia-tp-server: ready on ($config.host):($config.port) \(pid ($owner.pid)\)"
                return
            }
        }
        sleep 1sec
    }
    open --raw $config.log | lines | last 20 | each {|line| print --stderr $line }
    fail $"server did not become ready within ($seconds)s; see ($config.log)"
}

# Linux/L40S example: foreground [options], start [options], restart [options], stop, status.
# Host-specific model/cache defaults can be set with DS4_MODEL, DS4_KV_DIR, etc.
def --wrapped main [...args: string] {
    let command = $args | get --optional 0 | default foreground
    if $command in ['-h' '--help' 'help'] {
        print 'Usage: run-nvidia-tp-server.nu [foreground|start|restart] [server options...]
       run-nvidia-tp-server.nu stop|status

Retained Linux/L40S deployment example. Override host-specific defaults with
DS4_MODEL, DS4_KV_DIR, DS4_SERVER_HOST, DS4_SERVER_PORT and DS4_CTX.'
        return
    }
    cd $ROOT
    if $nu.os-info.name != linux { fail 'this retained CUDA-TP deployment example requires Linux' }
    let config = configuration
    let rest = $args | skip 1
    match $command {
        foreground | run => { run-server $config $rest }
        start => { start-server $config $rest }
        restart => {
            stop-server $config
            start-server $config $rest
        }
        stop => {
            if not ($rest | is-empty) { fail 'stop does not accept ds4-server options' }
            stop-server $config
        }
        status => {
            if not ($rest | is-empty) { fail 'status does not accept ds4-server options' }
            let owner = lock-owner $config
            require-owned $owner $config
            if $owner.kind == managed {
                print $"run-nvidia-tp-server: running \(pid ($owner.pid)\)"
            } else {
                print 'run-nvidia-tp-server: not running'
                exit 1
            }
        }
        _ => { run-server $config $args }
    }
}
