# Native XGrammar core

Unmodified native sources from [mlc-ai/xgrammar v0.2.8](https://github.com/mlc-ai/xgrammar/tree/v0.2.8),
commit `97787376faee5ed8466cfad57c99855e4ce2f6aa` (Apache-2.0).
Only `cpp/` excluding `tvm_ffi/`, public headers, picojson and DLPack headers are
included. Python, TVM FFI, CUDA, cpptrace and upstream test dependencies are not
part of this build. The ordinary Zig build compiles these C++17 sources locally.

DLPack headers are pinned to `bbd2f4d32427e548797929af08cfe2a9cbb3cf12`
([dmlc/dlpack](https://github.com/dmlc/dlpack), Apache-2.0); its license is beside
the headers. Picojson is the header shipped by this XGrammar revision; its BSD
license and copyright notices are at the start of `3rdparty/picojson/picojson.h`.

Serving adapters and the C boundary live outside this directory. They own
strict-schema policy, model-specific framing, token identity and sampling;
successful grammar compilation alone does not promise enforcement of every
JSON Schema keyword.
