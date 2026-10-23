CC ?= cc
CXX ?= c++
UNAME_S := $(shell uname -s)
LLAMA_CPP_DIR ?= ../../deepseek-v4-quants/llama.cpp
LLAMA_CPP_BUILD ?= $(LLAMA_CPP_DIR)/build-remote
LLAMA_CPP_BIN ?= $(LLAMA_CPP_BUILD)/bin
LLAMA_CPP_CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 \
	-I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
QUANTS_SHARED := libds4quants.dylib
SHARED_FLAGS := -dynamiclib
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -lggml-blas -lggml-metal \
	-framework Foundation -framework Metal -framework Accelerate
else
NATIVE_CPU_FLAG ?= -march=native
QUANTS_SHARED := libds4quants.so
SHARED_FLAGS := -shared
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -ldl -lpthread
endif

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quants-shared quality-score quality-llama-score

all: deepseek4-quantize $(QUANTS_SHARED)

quants-shared: $(QUANTS_SHARED)

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

$(QUANTS_SHARED): quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SHARED_FLAGS) -fPIC -o $@ quants.c -lm -pthread

# The scorer shares the production Zig engine; quantization remains standalone.
quality-score:
	$(MAKE) -C .. quality-score

quality-llama-score:
	$(CXX) $(LLAMA_CPP_CXXFLAGS) -o quality-testing/score_llama quality-testing/score_llama.cpp $(LLAMA_CPP_LDLIBS)

clean:
	rm -f deepseek4-quantize libds4quants.dylib libds4quants.so \
		quality-testing/score_official quality-testing/score_llama
