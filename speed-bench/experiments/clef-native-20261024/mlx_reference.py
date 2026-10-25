"""Offline MLX oracle adapter for the existing Decision Index runner."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import mlx.core as mx
from decision_index.engines.base import Engine


class Clef(Engine):
    name = 'clef-mlx-reference'
    latency = 'In-process record construction plus backbone and FP32 head; excludes model load and HTTP.'

    def __init__(self, model, vision=False, image_root=None, max_pixels=1638400, **options):
        super().__init__(**options)
        path = Path(model)
        spec = importlib.util.spec_from_file_location('clef_mlx_reference', path / 'clef_mlx.py')
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
        config = json.loads((path / 'config.json').read_text())
        self.omni = config.get('model_type') == 'qwen3_omni_moe'
        self.model = (ref.load(path, head_dtype=mx.float32) if self.omni else
                      ref.load(path, head_dtype=mx.float32, backend='vlm' if vision else 'lm'))
        self.vision, self.image_root, self.max_pixels = vision, image_root, max_pixels
        self.name = str(path)
        self.provenance = {'kind': 'offline-mlx-oracle', 'model': str(path), 'head_dtype': 'float32',
                           'vision': vision, 'max_pixels': max_pixels if vision else None,
                           'policy': 'Exact state/questions, no truncation, all option probabilities unrounded.'}

    def __call__(self, state, questions, images=(), prompt=None):
        record = {'state': state, 'questions': questions}
        if images:
            if not self.vision:
                raise ValueError('Set vision=true to load the published vision processor/tower.')
            from decision_index.vision import images as image_io
            record['images'] = [image_io.pil(image, self.image_root, self.max_pixels).convert('RGB')
                                for image in images]
        enc, logits = self.model.logits(record, truncate=False)
        answers = {}
        for q, lg in zip(enc.questions, logits):
            probabilities = dict(zip(q.option_ids, mx.softmax(lg.astype(mx.float32)).tolist()))
            choice = max(probabilities, key=probabilities.get)
            if questions[q.question_id]['type'] != 'choice':
                raise ValueError('This offline sample adapter only serves its categorical benchmark tasks.')
            answers[q.question_id] = {'type': 'choice', 'choice': choice,
                                      'confidence': probabilities[choice], 'probabilities': probabilities}
        result = {'model': self.name, 'answers': answers,
                  'usage': {'input_tokens': len(enc.input_ids), 'output_tokens': 0}}
        return result, result

    def synchronize(self):
        mx.synchronize()

    def runtime(self):
        return {name: importlib.metadata.version(name) for name in ('mlx', 'mlx-lm', 'transformers')}
