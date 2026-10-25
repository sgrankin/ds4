#!/usr/bin/env nu
# Small contrasting inputs for the real-model driver; no answers enter requests.
const ROOT = path self ../../..

def --wrapped checked [command: string, ...args: string] {
    ^$command ...$args
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}

def main [--work: path = '.workspaces/clef-media-20261024'] {
    cd $ROOT
    mkdir $work
    for color in [red blue] {
        (checked ffmpeg -hide_banner -loglevel error -f lavfi
            -i $'color=c=($color):s=128x128:r=2:d=1' -frames:v 1 -threads 1
            -y ($work | path join $'($color).png'))
    }
    (checked ffmpeg -hide_banner -loglevel error
        -f lavfi -i 'color=c=red:s=128x128:r=2:d=1'
        -f lavfi -i 'color=c=blue:s=128x128:r=2:d=1'
        -filter_complex '[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]'
        -map '[v]' -c:v libx264 -threads 1 -y ($work | path join red-blue.mp4))
    checked $nu.current-exe speed-bench/experiments/clef-native-20261024/speech_fixture.nu --work $work
    for label in [three seven] {
        let audio = if $label == seven { 'spoken-code.wav' } else { 'spoken-code-three.wav' }
        (checked ffmpeg -hide_banner -loglevel error -i ($work | path join red-blue.mp4)
            -i ($work | path join $audio) -map 0:v -map 1:a -c:v copy -c:a aac
            -af apad -t 2 -y ($work | path join $'audiovisual-($label).mp4'))
    }
}
