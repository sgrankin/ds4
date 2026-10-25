#!/usr/bin/env python3
"""Offline Omni audio-tower oracle from already prepared mel features."""
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('mel', type=Path)
    parser.add_argument('frames', type=int)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fp32', action='store_true')
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    from mlx_vlm.utils import load_model

    model = load_model(args.model)
    tower = model.thinker.audio_tower
    if args.fp32:
        tower.set_dtype(mx.float32)
    dtype = tower.conv2d1.weight.dtype
    # The published processor emits FP32 mel values, and Thinker passes them
    # directly to the audio tower. Keep that boundary in both weight modes.
    features = mx.array(np.fromfile(args.mel, dtype=np.float32).reshape(128, args.frames))
    lengths = mx.array([args.frames], dtype=mx.int32)
    hidden = tower(features, feature_lens=lengths)
    mx.eval(hidden)
    args.output.mkdir(parents=True, exist_ok=True)
    np.asarray(hidden.astype(mx.float32)).tofile(args.output / 'features.f32')

    times = []
    for _ in range(8):
        start = time.perf_counter()
        value = tower(features, feature_lens=lengths)
        mx.eval(value)
        times.append(time.perf_counter() - start)
    result = {
        'frames': args.frames,
        'shape': list(hidden.shape),
        'weight_dtype': str(dtype),
        'input_dtype': str(features.dtype),
        'median_seconds': statistics.median(times),
        'seconds': times,
    }
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
