# Apple Silicon Metal build. Zig owns compilation and incremental dependencies;
# these aliases keep the existing ./ds4 and make test workflows convenient.
.DEFAULT_GOAL := all
.SUFFIXES:
.NOTPARALLEL:

ZIG_FLAGS ?=
CC ?= cc
PROGRAMS := ds4 ds4-server ds4-agent ds4-bench ds4-eval

.PHONY: all $(PROGRAMS) help clean test test-scripts test-frontends test-web-recovery \
	test-state-cache test-response-store test-metal check-mxfp4-half-lut \
	q4k-dot-test mxfp4-dot-test test-quality-api test-download-model \
	quantize quants-shared quality-score ds4_test ds4_agent_test

# One Zig invocation builds all five programs, including the retained C Pro/full
# GLM 5.3 paths selected from GGUF metadata. Copy binaries for ./ds4-local.
all:
	./tools/zig-build $(ZIG_FLAGS)
	cp $(addprefix zig-out/bin/,$(PROGRAMS)) .

$(PROGRAMS):
	./tools/zig-build $(ZIG_FLAGS) $@
	cp zig-out/bin/$@ $@

help:
	@echo 'make                         Build the five Metal executables in the repo root'
	@echo 'make ds4-server              Build one executable'
	@echo 'make ZIG_FLAGS=-Doptimize=Debug  Select Debug host modules'
	@echo 'make test                    Run model-free host, protocol and script checks'
	@echo 'make test-scripts            Run fast Nushell launcher/downloader contracts'
	@echo 'make test-metal              Run DeepSeek, Qwen and GLM Metal kernel checks'
	@echo 'make zig-<step>              Build any named step from tools/zig-build --help'
	@echo 'make quantize quants-shared  Build the standalone GGUF quantization tools'
	@echo 'make quality-score           Build the scorer against the production Zig engine'
	@echo 'make clean                   Remove installed outputs; retain incremental caches'

# Pass through optional fixtures/benchmarks without maintaining a second build
# graph here. Build-only: run model-dependent fixtures explicitly afterwards.
zig-%:
	./tools/zig-build $(ZIG_FLAGS) $*

# The default suite does not load model weights or run GPU/latency experiments.
test:
	nu tests/verify.nu $(ZIG_FLAGS)

test-scripts:
	nu tests/verify.nu scripts

# Keep common individual aliases. Each fixture uses the production Zig objects.
DIRECT_TESTS := test-sampling test-engram test-image-decode \
	test-deepseek4-vision-image test-vocab test-tokenizer-piece test-pretokenizer \
	test-gguf-cursor test-ssd-cache test-prompt-prefix test-session-state \
	test-serving-boundary test-websocket-frames test-deepseek41-gguf \
	test-qwen4-ngrams test-deepseek41-metal test-deepseek41-graph \
	test-metal-dense-mpp test-metal-shared-bf16 test-metal-ssd-experts \
	test-metal-command-memory test-metal-norm-bf16 test-qwen4-kernels test-glm53-kda
.PHONY: $(DIRECT_TESTS)
test-sampling: ZIG_STEP = sampling-test
test-engram: ZIG_STEP = engram-test
test-image-decode: ZIG_STEP = image-test
test-deepseek4-vision-image: ZIG_STEP = vision-image-test
test-vocab: ZIG_STEP = vocab-test
test-tokenizer-piece: ZIG_STEP = tokenizer-test
test-pretokenizer: ZIG_STEP = pretokenizer-test
test-gguf-cursor: ZIG_STEP = gguf-test
test-ssd-cache: ZIG_STEP = ssd-test
test-prompt-prefix: ZIG_STEP = prompt-prefix-test
test-session-state: ZIG_STEP = session-state-test
test-serving-boundary: ZIG_STEP = serving-boundary-test
test-websocket-frames: ZIG_STEP = websocket-frames-test
test-deepseek41-gguf: ZIG_STEP = gguf-model-test
test-qwen4-ngrams: ZIG_STEP = qwen-ngram-test
test-deepseek41-metal: ZIG_STEP = metal-test
test-deepseek41-graph: ZIG_STEP = graph-test
test-metal-dense-mpp: ZIG_STEP = dense-mpp-test
test-metal-shared-bf16: ZIG_STEP = shared-bf16-test
test-metal-ssd-experts: ZIG_STEP = ssd-experts-test
test-metal-command-memory: ZIG_STEP = command-memory-test
test-metal-norm-bf16: ZIG_STEP = norm-metal-test
test-qwen4-kernels: ZIG_STEP = qwen-kernels-test
test-glm53-kda: ZIG_STEP = glm53-kda-test
$(DIRECT_TESTS):
	./tools/zig-build $(ZIG_FLAGS) $(ZIG_STEP)
	./zig-out/bin/$@

test-state-cache:
	./tools/zig-build $(ZIG_FLAGS) state-cache-test
	@mkdir -p .workspaces
	@set -eu; work=$$(mktemp -d .workspaces/make-tests.XXXXXX); \
		./zig-out/bin/test-state-cache "$$work/state"

test-response-store:
	./tools/zig-build $(ZIG_FLAGS) response-store-test
	@mkdir -p .workspaces
	@set -eu; work=$$(mktemp -d .workspaces/make-tests.XXXXXX); \
		./zig-out/bin/test-response-store "$$work/responses"

test-frontends:
	./tools/zig-build $(ZIG_FLAGS) server-test agent-test web-test
	./zig-out/bin/ds4_test --server
	./zig-out/bin/ds4_agent_test
	python3 tests/test_web_recovery.py ./zig-out/bin/test-web-recovery

test-web-recovery:
	./tools/zig-build $(ZIG_FLAGS) web-test
	python3 tests/test_web_recovery.py ./zig-out/bin/test-web-recovery

ds4_test:
	./tools/zig-build $(ZIG_FLAGS) server-test
	cp zig-out/bin/ds4_test $@

ds4_agent_test:
	./tools/zig-build $(ZIG_FLAGS) agent-test
	cp zig-out/bin/ds4_agent_test $@

test-metal: test-deepseek41-metal test-metal-dense-mpp test-metal-shared-bf16 \
	test-metal-ssd-experts test-metal-command-memory test-metal-norm-bf16 \
	test-qwen4-kernels test-glm53-kda

check-mxfp4-half-lut:
	python3 src/metal/generate_mxfp4_half_lut.py --check

# Standalone numerical references and tooling tests do not link an engine.
q4k-dot-test:
	@mkdir -p zig-out/bin
	$(CC) -O2 -Wall -Wextra -std=c99 -o zig-out/bin/test-q4k-dot tests/test_q4k_dot.c -lm -pthread
	./zig-out/bin/test-q4k-dot

mxfp4-dot-test:
	@mkdir -p zig-out/bin
	$(CC) -O2 -Wall -Wextra -std=c99 -o zig-out/bin/test-mxfp4-dot tests/test_mxfp4_dot.c -lm
	./zig-out/bin/test-mxfp4-dot

test-quality-api:
	@mkdir -p zig-out/bin
	$(CC) -O3 -Wall -Wextra -std=c11 -I. -Isrc -ffunction-sections -fdata-sections \
		-o zig-out/bin/test-quality-api tests/test_quality_api.c -Wl,-dead_strip -lm
	./zig-out/bin/test-quality-api
	python3 tests/test_collect_official.py

test-download-model:
	python3 tests/test_model_download.py

quantize:
	$(MAKE) -C gguf-tools deepseek4-quantize

quants-shared:
	$(MAKE) -C gguf-tools quants-shared

quality-score:
	./tools/zig-build $(ZIG_FLAGS) quality-score
	cp zig-out/bin/score-official gguf-tools/quality-testing/score_official

clean:
	rm -f $(PROGRAMS) ds4_test ds4_agent_test *.o tests/*.o
	rm -rf zig-out
