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

def verify [catalog: record, file: string, path: path] {
    let identity = $catalog.artifacts | get --optional $file
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
    fail 'Large GGUF downloads require the official Hugging Face CLI. Install it with:
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
    let part = $"($out).part"
    mkdir ($out | path dirname)
    if $model.transport == curl and ($"($out).aria2" | path exists) {
        fail $"Found incomplete aria2 download sidecar: ($out).aria2. Finish or remove it before using this curl downloader."
    }
    if (nonempty $out) {
        verify $catalog $file $out
        print $"Already downloaded: ($out)"
        return
    }
    print $"Downloading ($file)\nfrom https://huggingface.co/($model.repo)"
    if $model.transport == hf {
        if ($part | path exists) {
            fail $"Found curl partial download: ($part). The Hugging Face downloader cannot resume curl .part files. Move or remove it before retrying."
        }
        let command = (find-hf)
        print $"using ($command) download\nIf the download stops, run the same command again to resume it."
        let auth = if $token == '' { [] } else { [--token $token] }
        checked $command download $model.repo $file --repo-type model --local-dir $directory ...$auth
        let hf_out = $directory | path join $file
        if $hf_out != $out and (nonempty $hf_out) {
            mv $hf_out $out
            try { ^rmdir ($hf_out | path dirname) err> /dev/null } catch { }
        }
        if not (nonempty $out) { fail $"Hugging Face download finished but expected file is missing: ($out)" }
    } else {
        print 'If the download stops, run the same command again to resume it.'
        let auth = if $token == '' { [] } else { [-H $"Authorization: Bearer ($token)"] }
        (checked
            curl
            -fL
            --progress-meter
            -C
            -
            ...$auth
            -o
            $part
            $"https://huggingface.co/($model.repo)/resolve/main/($file)"
        )
        verify $catalog $file $part
        mv $part $out
        return
    }
    verify $catalog $file $out
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
        verify $catalog $file $out
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

def usage [catalog: record] {
    print 'DwarfStar GGUF downloader
Usage: ./download-model.nu TARGET [--token TOKEN]
DS4_GGUF_DIR selects the download directory (default: ./gguf).
Main-model downloads update ./ds4flash.gguf after successful download.
Pinned artifacts also receive size and SHA-256 checks.
'
    for model in ($catalog.targets | transpose name configuration) {
        print $"($model.name): ($model.configuration.description? | default $model.configuration.file)"
    }
    print "\nAlias: qwen38-iq2 -> qwen38-q2"
}

# Download a GGUF with resumable HF/curl transfers and pinned size/SHA checks.
# Use target "help" for detailed model descriptions.
#
# Targets:
#   ds4f-q2, ds4f-q2-q4, ds4f-q4, ds4f-mxfp4, ds4f-dspark
#   ds4f-vision-q2, ds4f-vision-q2-q4, ds4f-vision-mxfp4
#   ds4f-vision-encoder, ds4f-vision-dspark
#   ds41f-q2, ds41f-q4, ds41f-vision, ds41f-1017-vision
#   pro-q2-imatrix, pro-q4-layers00-30, pro-q4-layers31-output, pro-q4-split
#   glm-unsloth-q4, glm-antirez-iq2xxs, glm-antirez-q2, glm-antirez-q4
#   glm53-full-q2, glm53-q2, glm53-q4, glm53-fp8, glm53-vision
#   qwen38-q2 (alias qwen38-iq2), qwen38-q4k, qwen38-vision
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
    let directory = $ROOT | path join ($env.DS4_GGUF_DIR? | default --empty gguf) | path expand
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
    if $selected == ds4f-dspark or $selected == ds4f-vision-dspark {
        print $"Enable DSpark explicitly with --dspark --mtp-model ($directory | path join $model.file); use only its matching language checkpoint."
    } else if $selected == qwen38-q2 {
        print 'Run ./ds4 --ctx 8192 --prefill-chunk 1024. Add --mtp to enable speculation.'
    } else if $selected == qwen38-q4k {
        print 'Run ./ds4. Add --mtp to enable speculation.'
    } else if $selected == qwen38-vision {
        print $"Pass the vision encoder with --vision ($directory | path join $model.file)."
    }
}
