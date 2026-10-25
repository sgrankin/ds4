#!/usr/bin/env nu
# Real-model media checks. Generate the private fixtures with speech_fixture.nu
# and provide red.png, blue.png and red-blue.mp4 in the same directory.
use std/assert

def data-url [file: path, mime: string] {
    $'data:($mime);base64,(open --raw $file | encode base64)'
}

def post [url: string, body: record] {
    let started = date now
    let result = http post --full --allow-errors --content-type application/json $'($url)/v1/systemone' $body
    assert equal $result.status 200
    $result.body | insert qualification_wall_ms (((date now) - $started) / 1ms)
}

def main [
    --url: string = 'http://127.0.0.1:19311',
    --model: string = 'clef-omni',
    --fixtures: path = '.workspaces/clef-media-20261024',
    --output: path = '.workspaces/clef-media-serving',
    --vision-only
] {
    mkdir $output
    let red = data-url ($fixtures | path join red.png) image/png
    let blue = data-url ($fixtures | path join blue.png) image/png
    let colors = {red: 'Red', blue: 'Blue', green: 'Green'}
    let image_question = {color: {type: choice, instructions: 'What is the dominant color in the image?', criteria: $colors}}
    for sample in [[expected, image]; [red, $red] [blue, $blue]] {
        let result = post $url {model: $model, state: 'Inspect the attached image.', images: [$sample.image], questions: $image_question}
        $result | save -f ($output | path join $'image-($sample.expected).json')
        assert equal $result.answers.color.choice $sample.expected
    }
    let question_image = http post --full --allow-errors --content-type application/json $'($url)/v1/decisions' {
        model: $model, input: 'Answer the visual question.',
        questions: [{type: choice, instructions: [
            {type: input_text, text: 'What is the dominant color in this image?'},
            {type: input_image, image_url: $red}
        ], choices: [{value: red, description: Red}, {value: blue, description: Blue}]}]
    }
    $question_image.body | save -f ($output | path join question-image.json)
    assert equal $question_image.status 200
    assert equal $question_image.body.answers.0.choice red

    let video_question = {last: {type: choice, instructions: 'What color is visible at the end of the video?', criteria: $colors}}
    let frames = {frames: [$red $red $blue $blue], timestamps: [0 0.5 1 1.5]}
    let video = post $url {model: $model, state: 'Inspect the attached video.', videos: [$frames], questions: $video_question}
    $video | save -f ($output | path join video-frames.json)
    assert equal $video.answers.last.choice blue
    let encoded = data-url ($fixtures | path join red-blue.mp4) video/mp4
    let decoded = post $url {model: $model, state: 'Inspect the attached video.', videos: [$encoded], questions: $video_question}
    $decoded | save -f ($output | path join video-encoded.json)
    assert equal $decoded.answers.last.choice blue

    if not $vision_only {
        let codes = {three: 'The spoken access code is three', seven: 'The spoken access code is seven', nine: 'The spoken access code is nine'}
        let question = {code: {type: choice, instructions: 'What access code does the speaker say?', criteria: $codes}}
        for label in [three seven] {
            let file = if $label == seven { 'spoken-code.wav' } else { 'spoken-code-three.wav' }
            let audio = data-url ($fixtures | path join $file) audio/wav
            let result = post $url {model: $model, state: 'Listen to the recording.', audio: [$audio], questions: $question}
            $result | save -f ($output | path join $'audio-($label).json')
            assert equal $result.answers.code.choice $label
            let combined = post $url {
                model: $model, state: 'Inspect the video and listen to its soundtrack.',
                videos: [($frames | insert audio $audio)],
                questions: ($question | insert last $video_question.last)
            }
            $combined | save -f ($output | path join $'audiovisual-($label).json')
            assert equal $combined.answers.code.choice $label
            assert equal $combined.answers.last.choice blue
            let encoded_av = data-url ($fixtures | path join $'audiovisual-($label).mp4') video/mp4
            let muxed = post $url {
                model: $model, state: 'Inspect the video and listen to its soundtrack.',
                videos: [$encoded_av], questions: ($question | insert last $video_question.last)
            }
            $muxed | save -f ($output | path join $'audiovisual-encoded-($label).json')
            assert equal $muxed.answers.code.choice $label
            assert equal $muxed.answers.last.choice blue
        }
    }
    print $'($model): image contrast, frame/encoded video and selected audio checks pass.'
}
