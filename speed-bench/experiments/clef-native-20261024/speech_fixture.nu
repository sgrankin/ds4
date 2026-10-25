#!/usr/bin/env nu

# A private, generated audio fixture. The spoken label belongs in the test's
# expected answer, never in the model request's state or question text.
const ROOT = path self ../../..

def --wrapped checked [command: string, ...args: string] {
    ^$command ...$args
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}

def main [--work: path = '.workspaces/clef-media-20261024'] {
    cd $ROOT
    mkdir $work
    for clip in [
        {name: spoken-code, text: 'The access code is seven.'}
        {name: spoken-code-three, text: 'The access code is three.'}
    ] {
        let source = $work | path join $'($clip.name).aiff'
        let output = $work | path join $'($clip.name).wav'
        checked /usr/bin/say -r 180 -o $source $clip.text
        (checked ffmpeg -hide_banner -loglevel error -i $source
            -ac 1 -ar 16000 -c:a pcm_s16le -y $output)
        # Sandboxed speech services can return success with an empty file.
        let probe = ^ffprobe -v error -show_entries format=duration -of json $output | from json
        if (($probe.format | get --optional duration | default '0' | into float) < 0.5) {
            error make {msg: $'Empty speech fixture: ($output); rerun with access to the macOS speech service.'}
        }
        print $output
    }
}
