#!/usr/bin/env python3
"""Offline CPU media fixtures: frozen native image oracle, NumPy HF layout and Whisper frontend.

Compile the reference-only libcleffa-image.dylib first; raw output stays local.
"""
import ctypes as c
import io, wave, json
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import WhisperFeatureExtractor
root=Path('.workspaces/clef-media-20261024')
class RGB(c.Structure):_fields_=[('width',c.c_int),('height',c.c_int),('rgb',c.POINTER(c.c_uint8))]
class Params(c.Structure):_fields_=[('min_pixels',c.c_long),('max_pixels',c.c_long),('patch',c.c_int),('merge',c.c_int),('temporal',c.c_int)]
class Patches(c.Structure):_fields_=[('width',c.c_int),('height',c.c_int),('grid_h',c.c_int),('grid_w',c.c_int),('n_patch',c.c_int),('n_tokens',c.c_int),('patch_dim',c.c_int),('patches',c.POINTER(c.c_float))]
lib=c.CDLL(str(root/'libcleffa-image.dylib'))
lib.clef_image_preprocess.argtypes=[c.POINTER(RGB),c.POINTER(Params),c.POINTER(Patches),c.c_char_p,c.c_size_t]
lib.clef_image_patches_free.argtypes=[c.POINTER(Patches)]
y,x=np.mgrid[:64,:64]
rgb=np.stack(((x*7+y*3)%256,(x*2+y*11)%256,(x*13+y*5)%256),axis=-1).astype(np.uint8)
other=np.stack((255-rgb[...,0],rgb[...,2],rgb[...,1]),axis=-1)
Image.fromarray(rgb).save(root/'image.png'); Image.fromarray(other).save(root/'second.png');rgb.tofile(root/'image.rgb')
source=RGB(64,64,rgb.ctypes.data_as(c.POINTER(c.c_uint8)));params=Params(65536,16777216,16,2,2);out=Patches();error=c.create_string_buffer(256)
assert lib.clef_image_preprocess(c.byref(source),c.byref(params),c.byref(out),error,256),error.value
np.ctypeslib.as_array(out.patches,shape=(out.n_patch*out.patch_dim,)).tofile(root/'image-patches.f32');lib.clef_image_patches_free(c.byref(out))
# Independent NumPy HF patch reshaping, including two different temporal taps.
frames=np.stack([rgb,other,rgb,rgb]).transpose(0,3,1,2).astype(np.float32)
frames=(frames-np.float32(127.5))/np.float32(127.5)
patches=frames.reshape(2,2,3,2,2,16,2,2,16).transpose(0,3,6,4,7,2,1,5,8).reshape(-1,1536)
patches.tofile(root/'video-patches.f32')
# Bilinear learned position table, same float32 endpoint arithmetic as torch.
axis=np.linspace(0,47,4,dtype=np.float32);indices=[];weights=[]
for by in range(2):
 for bx in range(2):
  for my in range(2):
   for mx in range(2):
    h,w=axis[by*2+my],axis[bx*2+mx];h0,w0=int(h),int(w);h1,w1=min(h0+1,47),min(w0+1,47);hf,wf=h-h0,w-w0
    indices.append([h0*48+w0,h0*48+w1,h1*48+w0,h1*48+w1]);weights.append([(1-hf)*(1-wf),(1-hf)*wf,hf*(1-wf),hf*wf])
np.asarray(indices,dtype=np.int32).tofile(root/'position-indices.i32');np.asarray(weights,dtype=np.float32).tofile(root/'position-weights.f32')
# Quantized synthetic signal gives exact PCM16 sample identity, then a real
# independent Whisper frontend oracle without requiring a model or GPU.
t=np.arange(20000,dtype=np.float64)/16000
signal=(.35*np.sin(2*np.pi*(220*t+100*t*t))+.1*np.cos(2*np.pi*997*t))
pcm=np.clip(np.rint(signal*32768),-32768,32767).astype('<i2')
with wave.open(str(root/'audio.wav'),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(pcm.tobytes())
samples=pcm.astype(np.float32)/32768;samples.tofile(root/'audio-samples.f32')
extractor=WhisperFeatureExtractor(feature_size=128,n_fft=400,hop_length=160,sampling_rate=16000)
features=extractor(samples,sampling_rate=16000,padding='longest',truncation=False,return_tensors='np',return_attention_mask=True)
features.input_features[0].astype(np.float32).tofile(root/'audio-mel.f32')
(root/'fixture-info.json').write_text(json.dumps({'audio_mel_shape':list(features.input_features[0].shape),'image_reference':'cleffa24fc91e2 uint8 antialiased bicubic','audio_reference':'transformers WhisperFeatureExtractor numpy','video_reference':'NumPy canonical HF reshape/transpose'},indent=2)+'\n')
print('Wrote CPU media fixtures',features.input_features.shape)
