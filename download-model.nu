#!/usr/bin/env nu

const ROOT = path self .

# Run download tools with their normal terminal output and original failure code.
def --wrapped checked [command: string, ...args: string] {
    ^$command ...$args
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}

def fail [message: string] {
    print --stderr $message
    exit 1
}

def file-size [file: path] {
    ls --directory ($file | path expand --strict) | get size | first | into int
}

def nonempty [file: path] {
    ($file | path exists) and ((file-size $file) > 0)
}

def verify [catalog: record, model: record, file: string, path: path] {
    # Directory bundles share names such as config.json. Their identity belongs
    # to that model, while older standalone artifacts retain the shared table.
    let identity = $model.artifacts? | default {} | get --optional $file
        | default ($catalog.artifacts | get --optional $file)
    if $identity == null { return }
    if (file-size $path) != $identity.bytes {
        fail $"Incorrect file size: ($path). Move the incomplete file aside and retry."
    }
    print $"Verifying SHA-256: ($path)"
    # These tools stream the file rather than bringing a multi-hundred-GB GGUF
    # into shell memory. Only the short digest result enters the pipeline.
    let result = if (which sha256sum | is-not-empty) {
        ^sha256sum -- $path | complete
    } else {
        ^shasum -a 256 -- $path | complete
    }
    if $result.exit_code != 0 {
        print --stderr --no-newline $result.stderr
        exit $result.exit_code
    }
    let digest = $result.stdout | split row --regex '\s+' | first
    if $digest != $identity.sha256 {
        fail $"Checksum mismatch: ($path). The file was not accepted."
    }
}

def find-hf [] {
    let found = (which hf)
    if ($found | is-not-empty) { return ($found | first | get path) }
    let python_bins = try { glob ($env.HOME | path join 'Library/Python/*/bin/hf') } catch { [] }
    let candidates = $python_bins | append ($env.HOME | path join .local/bin/hf)
    for candidate in $candidates {
        if (which $candidate | is-not-empty) { return $candidate }
    }
    fail 'Model downloads require the official Hugging Face CLI. Install it with:
  python3 -m pip install -U huggingface_hub hf_xet'
}

def download-one [
    catalog: record
    model: record
    file: string
    directory: path
    token: string
] {
    let local_name = if ($model.flatten? | default false) {
        $file | path basename
    } else {
        $file
    }
    let out = $directory | path join $local_name
    mkdir ($out | path dirname)
    if (nonempty $out) {
        verify $catalog $model $file $out
        print $"Already downloaded: ($out)"
        return
    }
    print $"Downloading ($file)\nfrom https://huggingface.co/($model.repo)"
    let part = $"($out).part"
    if ($part | path exists) {
        fail $"Found curl partial download: ($part). The Hugging Face downloader cannot resume curl .part files. Move or remove it before retrying."
    }
    let command = (find-hf)
    print $"using ($command) download\nIf the download stops, run the same command again to resume it."
    let auth = if $token == '' { [] } else { [--token $token] }
    let revision = if ($model.revision? | default '') == '' { [] } else { [--revision $model.revision] }
    (checked
        $command
        download
        $model.repo
        $file
        --repo-type
        model
        --local-dir
        $directory
        ...$revision
        ...$auth
    )
    let hf_out = $directory | path join $file
    if $hf_out != $out and (nonempty $hf_out) {
        mv $hf_out $out
        try { ^rmdir ($hf_out | path dirname) err> /dev/null } catch { }
    }
    if not (nonempty $out) { fail $"Hugging Face download finished but expected file is missing: ($out)" }
    verify $catalog $model $file $out
}

def download-split [
    catalog: record
    model: record
    directory: path
    token: string
] {
    let file = $catalog.split.file
    let out = $directory | path join $file
    if ($out | path exists) {
        verify $catalog $model $file $out
        print $"Already downloaded: ($out)"
        return
    }
    if not ($"($out).assembling" | path exists) {
        download-one $catalog $model $"($file).part1" $directory $token
    }
    download-one $catalog $model $"($file).part2" $directory $token
    let full = $catalog.artifacts | get $file
    let first = $catalog.artifacts | get $"($file).part1"
    (checked
        python3
        ($ROOT | path join tools/join-model-parts.py)
        $out
        ($first.bytes | into string)
        ($full.bytes | into string)
        $full.sha256
    )
}

def download-mxfp4 [catalog: record, model: record, directory: path, token: string] {
    let out = $directory | path join $model.file
    if ($out | path exists) {
        verify $catalog $model $model.file $out
        print $"Already downloaded: ($out)"
        return
    }
    let merger = $ROOT | path join gguf-tools/merge_deepseek41_shards.py
    if not ($merger | path exists) { fail $"Missing ($merger); use a complete checkout." }
    let full = $catalog.artifacts | get $model.file
    let pending = $"($out).assembling"
    # Retain the input shards and allow an interrupted output to resume. HF
    # partial caches can require additional space beyond this 1 GiB reserve.
    mut needed = 1gib | into int
    let assembled = if ($pending | path exists) { file-size $pending } else { 0 }
    $needed += ([0 ($full.bytes - $assembled)] | math max)
    for file in $model.files {
        let path = $directory | path join $file
        let present = if ($path | path exists) { file-size $path } else { 0 }
        $needed += ([0 (($catalog.artifacts | get $file | get bytes) - $present)] | math max)
    }
    let location = $directory | path expand --strict
    let disk = sys disks
        | where { |disk| $disk.mount == '/' or $location == $disk.mount or ($location | str starts-with $"($disk.mount)/") }
        | sort-by { |disk| $disk.mount | str length }
        | last
    if ($disk.free | into int) < $needed {
        fail $"Not enough disk space for MXFP4 shards and merged GGUF: need ($needed | into filesize) free. Existing files were kept."
    }
    for file in $model.files { download-one $catalog $model $file $directory $token }
    let shards = $model.files | each { |file| $directory | path join $file }
    (checked
        python3
        $merger
        --output
        $out
        --expected-bytes
        ($full.bytes | into string)
        --sha256
        $full.sha256
        ...$shards
    )
}

def usage [catalog: record] {
    print 'DwarfStar model downloader
Usage: ./download-model.nu TARGET [--token TOKEN]
DS4_GGUF_DIR selects the download directory (default: ./gguf).
Flint and Clef targets download complete safetensors directories under ./gguf.
Main GGUF downloads update ./ds4flash.gguf after successful download; directory bundles leave it unchanged.
Pinned artifacts also receive size and SHA-256 checks.
'
    for model in ($catalog.targets | transpose name configuration) {
        print $"($model.name): ($model.configuration.description? | default $model.configuration.file)"
    }
    print "\nAliases: qwen38-iq2 -> qwen38-q2; flint -> flint-int4; clef/clef-flash/clef-omni -> corresponding -int4 targets"
}

# Download model files with resumable HF transfers and pinned size/SHA checks.
# Use target "help" for detailed model descriptions.
#
# Targets:
#   ds41f-q2, ds41f-q4, ds41f-mxfp4, ds41f-vision, ds41f-1017-vision
#   pro-q2-imatrix, pro-q4-layers00-30, pro-q4-layers31-output, pro-q4-split
#   glm53-full-q2, glm53-q2, glm53-q4, glm53-fp8, glm53-vision
#   qwen38-q2 (alias qwen38-iq2), qwen38-q4k, qwen38-vision
#   flint-int4 (alias flint): complete native affine INT4 safetensors directory
#   clef-int4, clef-flash-int4, clef-omni-int4: decision backbones and joint heads
#
# HF_TOKEN or the local HF token cache is used when --token is absent.
# DS4_GGUF_DIR chooses the output directory (default: repository ./gguf).
def main [target?: string, --token: string] {
    let catalog = open ($ROOT | path join tools/model-downloads.json)
    if $target == null {
        usage $catalog
        exit 1
    }
    if $target == help {
        usage $catalog
        return
    }
    let selected = $catalog.aliases | get --optional $target | default $target
    let model = $catalog.targets | get --optional $selected
    if $model == null { fail $"Unknown model: ($target). Use ./download-model.nu help for available targets." }
    let base = $ROOT | path join ($env.DS4_GGUF_DIR? | default --empty gguf) | path expand
    let directory = $base | path join ($model.directory? | default '')
    let token_file = $env.HOME | path join .cache/huggingface/token
    let auth = $token | default ($env.HF_TOKEN? | default '')
    let auth = if $auth == '' and (nonempty $token_file) {
        open --raw $token_file | str trim
    } else {
        $auth
    }
    mkdir $directory
    if $selected == $catalog.split.target {
        download-split $catalog $model $directory $auth
    } else if $selected == ds41f-mxfp4 {
        download-mxfp4 $catalog $model $directory $auth
    } else {
        let files = $model.files? | default [$model.file]
        for file in $files { download-one $catalog $model $file $directory $auth }
    }
    if ($model.link? | default true) {
        let pending = $ROOT | path join $"ds4flash.gguf.($nu.pid).new"
        checked ln -s ($directory | path join $model.file) $pending
        checked mv -f $pending ($ROOT | path join ds4flash.gguf)
        print $"Linked ./ds4flash.gguf -> ($directory | path join $model.file)"
    }
    print 'Done.'
    if $selected == flint-int4 {
        print $"Run ./ds4-local --server flint --model '($directory)'"
    } else if $selected == qwen38-q2 {
        print 'Run ./ds4 --ctx 8192 --prefill-chunk 1024. Add --mtp to enable speculation.'
    } else if $selected == qwen38-q4k {
        print 'Run ./ds4. Add --mtp to enable speculation.'
    } else if $selected == qwen38-vision {
        print $"Pass the vision encoder with --vision ($directory | path join $model.file)."
    }
}
