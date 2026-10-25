#!/usr/bin/env nu

# Reuse reference math and the actual production Zig frontend, without a model.
const ROOT = path self ../../..

def --wrapped checked [command: string, ...args: string] {
    ^$command ...$args
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}

def main [--test: path] {
    cd $ROOT
    let work = '.workspaces/clef-media-20261024'
    let reference = '.workspaces/clef-references/cleffa'
    mkdir $work
    (checked clang -dynamiclib -O3 -fno-fast-math -I $reference
        ($reference | path join clef_image.c) -lz -o ($work | path join libcleffa-image.dylib))
    (checked .workspaces/clef-artifacts-20261024/venv/bin/python
        speed-bench/experiments/clef-native-20261024/media_oracle.py)
    (checked ffmpeg -hide_banner -loglevel error -f lavfi -i 'testsrc=size=64x64:rate=24:duration=2'
        -f lavfi -i 'sine=frequency=440:sample_rate=44100:duration=2' -c:v ffv1 -pix_fmt bgr0
        -c:a pcm_s16le -y ($work | path join video.mkv))
    (checked ffmpeg -hide_banner -loglevel error -i ($work | path join video.mkv) -map '0:v:0'
        -vf "select='not(mod(n,12))'" -fps_mode passthrough -pix_fmt rgb24 -f rawvideo
        -y ($work | path join decoded-video.rgb))
    (checked ffmpeg -hide_banner -loglevel error -i ($work | path join video.mkv) -map '0:a:0'
        -ac 1 -ar 16000 -f f32le -y ($work | path join decoded-audio.f32))
    (checked ffmpeg -hide_banner -loglevel error -itsoffset 2 -i ($work | path join video.mkv)
        -map 0 -c copy -y ($work | path join video-offset.mkv))
    let executable = $test | default '.workspaces/clef-native-20261024/build/bin/test-clef-media'
    checked $executable $work
}
