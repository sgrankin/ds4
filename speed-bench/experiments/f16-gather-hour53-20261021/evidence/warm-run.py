import hashlib,json,os,subprocess
from pathlib import Path
env={k:v for k,v in os.environ.items() if not k.startswith(("DS4_", "PT_"))}
env.update(DS4_V41_COMPRESSED_KV_F16="1",DS4_V41_DECODER_REPLAY_ROWS="0", DS4_METAL_V41_RESIDENT_ROUTE_GAP="0", DS4_METAL_V41_RESIDENT_PREFILL_GAP="0", DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS="0", DS4_METAL_V41_SELECTED_TILE="64", DS4_METAL_V41_SELECTED_LIMIT="1024")
cmd=["speed-bench/metal_decode_schedule_bench","--ssd-streaming","--model","gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf","--prefix-tokens","8192","--ctx","16384","--warmup","16","--tokens","256","--candidate-env","DS4_METAL_V41_PACKED_F16_GATHER","--include-selection"]
Path(".workspaces/f16-gather-hour53/warm-execution.json").write_text(json.dumps({"command":cmd,"env":{k:v for k,v in env.items() if k.startswith("DS4_")},"base_revision":"a2c47483ffd2ffac58726da0f07da5df4cf9accd","binary_sha256":hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest(),"prompt_file_sha256":hashlib.sha256(Path("ds4.c").read_bytes()).hexdigest()},indent=2)+"\n")
raise SystemExit(subprocess.run(cmd,env=env).returncode)
