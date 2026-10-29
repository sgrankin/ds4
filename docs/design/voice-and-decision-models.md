# Voice and decision model direction

The October 2026 design discussion broadened the intended local runtime beyond generative models to typed decisions, extraction and voice. These were candidate capabilities and references in the October discussion. The Clef family and both decision endpoints have since been implemented; the remaining voice and helper candidates are still design options. The selected product direction is recorded in [project status](../project.md); [model composition](model-composition.md) explains engine ownership and shared scheduling.

## Decisions and extraction

The Clef family and both decision endpoints are now implemented; see the
[Clef manuals](../models/clef/README.md) for the current contract and qualification.
The other candidates and broader constellation below remain design options.

A decision model may consume text, images, video or audio and return a predicate, choice or score rather than generate a token stream. Its encoder, joint head, tokenizer and media preprocessing are part of the implementation. It should not need conversational KV or token sampling just to fit a generative interface. Representations can be reused when the model permits, but KV is not transferable across models.

The intended protocol surfaces are TypeSafe-compatible `/v1/systemone` and OpenAI-compatible `/v1/decisions`, with a common native decision execution layer and separate request/result adapters. Their probability, confidence, refusal and modality semantics need individual treatment; similar concepts do not make their wire contracts equivalent. Extraction and voice compatibility surfaces remained under discussion. These endpoints were proposed, not implemented by the October discussion. The generic server-hosted tools have since been removed; this design does not reinstate them. References examined then include [llama.cpp system-one](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api), [Decisions API](https://developers.openai.com/api/docs/guides/decisions) and the distinct [Clairic extraction contract](https://developers.cloudflare.com/workers-ai/models/clairic/).

The October 24 shortlist included [Clef 27B](https://huggingface.co/Cloudflare/clef) and [Clef-Flash 9B](https://huggingface.co/Cloudflare/clef-flash) for text, images and video frames; [Clef-Omni 30B-A3B](https://huggingface.co/Cloudflare/clef-omni) adds audio and video with sound. Smaller text candidates were [Laya](https://huggingface.co/convaiinnovations/laya), [GLiNER2](https://github.com/Andrew-Chen-Wang/gliner2-mlx) for schema extraction, [GLiNER2.5-Decide](https://github.com/souuzaa/gliner2-mlx) for dynamic labels, and [GLiClass Edge](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml) for a small classifier. These roles reflect the source review, not local qualification. [Cleffa](https://github.com/zknpr/cleffa) is a C/Objective-C/Metal reference for Clef including media and heads; its BF16 footprint and unfinished long-context Omni performance are not requirements for a native port.

Production inference should be native, without a Python sidecar or embedded Python implementation. Reference algorithms, packing and precision boundaries can be adapted; Python remains useful for offline conversion and numerical oracles. C/C++ libraries, Metal, CPU/Accelerate and Core ML/Neural Engine are options, not mandatory boundaries. Repacking, quantization and mixed precision should be judged on task accuracy, calibration, escalation, working memory, conversion and load cost, and interference with the main model. An isolated short-input timing does not establish long-conversation performance.

## Voice sessions and work that outlives speech

A continuous voice frontend needs separate microphone/playback transport, speech processing, conversation state and backend task state. Duplex only means listening and speaking can overlap; it does not guarantee tool use or continued free-form conversation during backend work. Interrupting playback, canceling speech generation and canceling or revising a backend job are separate actions. The application should retain task IDs, progress, completed actions and stale-result handling so an interrupted session does not discard or repeat useful work; an explicit correction can supersede it.

The October 24 source review distinguished [Realtime](https://developers.openai.com/api/docs/guides/realtime), a voice/reasoning/tool session, from [Live](https://developers.openai.com/api/docs/guides/live), which separates spoken interaction from delegated work. WebRTC carries audio tracks and a JSON data channel; a primary WebSocket carries audio and events. A sideband controls the same voice session, not a separate reasoning connection. In [client delegation](https://developers.openai.com/api/docs/guides/live-delegation), the application builds a backend request from its own transcript and task state and returns progress or results; the delegation event itself is not the completed request. Managed Responses delegation has different backend selection semantics and is not an arbitrary local base URL.

Recognition and synthesis might use CPU or ANE while a main model uses the GPU, but actual placement, shared memory bandwidth, audio frame deadlines, backlog, first audio and main-model throughput must be measured together. GPU voice models also remain candidates. The [VoiceChat native Apple reference](https://github.com/soniqo/speech-swift/blob/main/docs/inference/voicechat.md) reported isolated INT5 M5 Pro compute around 73–75 ms per 80 ms frame; that is neither concurrent DeepSeek qualification nor first-audio latency. Its documented shared language KV prevents free-form speech during function-channel work even though perception continues, illustrating why duplex alone is insufficient.

## Voice implementation references

The following shortlist records the October 24 source review. Checkpoint and
port details are dated observations, not newly verified performance or local
model qualification. Native references are algorithm and kernel sources; their
languages are not required linkage boundaries for this runtime.

| Workload | Candidate and reason to examine it | Existing Apple/native references and limits |
| --- | --- | --- |
| Fast transcription | Parakeet Ultra 0.6B, 25 European languages | [FluidAudio Core ML/ANE](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/ASR/ParakeetUltra.md); file/window throughput is not live first-partial latency |
| Streaming recognition | Nemotron streaming ASR | [Native cached streaming/latency tiers](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md); language/accuracy varies by checkpoint and chunk size |
| Multilingual recognition | Qwen3-ASR 0.6B/1.7B | [Models](https://github.com/QwenLM/Qwen3-ASR), [MLX encoder/decoder/alignment](https://github.com/moona3k/mlx-qwen3-asr); examined streaming path uses chunking and revisable text |
| Mature recognition baseline | Whisper | [whisper.cpp Metal/Core ML](https://github.com/ggml-org/whisper.cpp), [WhisperKit](https://github.com/argmaxinc/argmax-oss-swift); repeated-window transcription is not a cached transducer |
| Lightweight speech synthesis | PocketTTS 100M and October 20 tiny English 18M/7M/2M releases | [Official models](https://github.com/kyutai-labs/pocket-tts), [FluidAudio CPU/Core ML/ANE implementation](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/TTS/PocketTTS.md); accelerator split differs by variant, tiny-model cloning is not wired in that port |
| Expressive synthesis / cloning | Qwen3-TTS 0.6B/1.7B | [Models](https://github.com/QwenLM/Qwen3-TTS), [native MLX model/codec/streaming](https://github.com/Blaizzy/mlx-audio-swift/tree/main/Sources/MLXAudioTTS/Models/Qwen3TTS); Base, CustomVoice and VoiceDesign have different capabilities |
| Long-form and streaming synthesis | VibeVoice | [Microsoft releases](https://github.com/microsoft/VibeVoice), [community fork](https://github.com/vibevoice-community/VibeVoice), [Apple implementation](https://github.com/soniqo/speech-swift/blob/main/docs/models/vibevoice.md); original long-form code was withdrawn September 2025, separate official Realtime-0.5B followed December 2025; examined native port still returns whole waveforms from its stream API |
| Expressive synthesis comparison | Fish Audio S2 Pro | [Weights and architecture](https://huggingface.co/fishaudio/s2-pro); published headline latency is H200 evidence, not Apple performance |
| Duplex conversation | Moshi / PersonaPlex | [Official MLX and Rust/Metal](https://github.com/kyutai-labs/moshi), [native PersonaPlex work](https://github.com/soniqo/speech-swift/blob/main/docs/models/personaplex.md); native duplex does not itself provide backend delegation |
| Duplex with backend delegation | Realtime-Venus | [Project and models](https://realtime-venus.github.io/); architecture matches the frontend/backend split, Apple execution not verified |

[0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) is another useful native
C++/ggml/Metal model and codec reference, including Qwen/Pocket/Vibe. Its remote
release and published CUDA speed table were not equated to the installed
`../audio.cpp` checkout or to Apple model qualification.

For protocol research, [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech#realtime-api)
demonstrates a tested core OpenAI Realtime subset over WebSocket/WebRTC on a
cascaded ASR → language model → TTS pipeline. Original Moshi/PersonaPlex protocols
are different; a `/v1/realtime` path alone does not establish client compatibility.
Our existing Responses WebSocket and a future voice session adapter are distinct
protocols over shared application state. Keep heard speech/playback state apart
from generated-but-unplayed output and durable backend progress.

Voice qualification should measure first useful audible response, interruption
response, recognition errors on names and numbers, relevant languages,
naturalness, frame p95/p99 and backlog alongside main-model prefill and decode.
Real-time factor alone is insufficient. None of these voice models has been
qualified in this runtime.
