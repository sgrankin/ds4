#!/usr/bin/env nu
# Recreate AC-refinement ZRL fixtures with libjpeg-turbo cjpeg.
const HERE = path self | path dirname

const ROOT = (
    path self
    | path dirname
    | path dirname
    | path dirname
    | path dirname
)

def pixels [width: int, height: int, color: bool] {
    0..<$height | each {|y|
        0..<$width | each {|x|
            let value = $x * 1103515245 + $y * 12345 + 42 | bits and 0xFFFFFFFF
            let red = ($value | bits shr 16) | bits and 255
            if $color { [$red (($value | bits shr 8) | bits and 255) ($value | bits and 255)] } else { [$red] }
        }
    } | flatten | flatten
}

def main [--output: path = $HERE] {
    if (which cjpeg | is-empty) { error make {msg: 'generate.nu: cjpeg not found (install libjpeg-turbo)'} }
    mkdir $output
    let parent = $ROOT | path join '.workspaces'
    mkdir $parent
    let work = (mktemp -d -p $parent 'jpeg-fixtures.XXXXXX')
    let gray = $work | path join 'gray.pgm'
    let color = $work | path join 'color.ppm'
    bytes build ("P5\n24 16\n255\n" | into binary) ...(pixels 24 16 false) | save --raw $gray
    bytes build ("P6\n64 48\n255\n" | into binary) ...(pixels 64 48 true) | save --raw $color
    let gray_scans = $work | path join 'gray.scans'
    let color_scans = $work | path join 'color.scans'
    '0: 0-0, 0, 1 ;
0: 1-63, 0, 2 ;
0: 1-63, 2, 1 ;
0: 1-63, 1, 0 ;
0: 0-0, 1, 0 ;
' | save --raw $gray_scans
    '0,1,2: 0-0, 0, 1 ;
0: 1-63, 0, 2 ;
1: 1-63, 0, 1 ;
2: 1-63, 0, 1 ;
0: 1-63, 2, 1 ;
0: 1-63, 1, 0 ;
0,1,2: 0-0, 1, 0 ;
1: 1-63, 1, 0 ;
2: 1-63, 1, 0 ;
' | save --raw $color_scans
    ^cjpeg -grayscale -quality 75 -restart 1 -scans $gray_scans -outfile ($output | path join 'prog_ac_refine_zrl_gray.jpg') $gray
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    ^cjpeg -quality 75 -restart 1 -scans $color_scans -outfile ($output | path join 'prog_ac_refine_zrl_420.jpg') $color
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    print $'wrote JPEG fixtures under ($output); inputs: ($work)'
}
