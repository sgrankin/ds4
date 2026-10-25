# QuickJS-NG

Unmodified embedding sources from QuickJS-NG **v0.17.0**, MIT licensed.
Source archive: https://github.com/quickjs-ng/quickjs/archive/refs/tags/v0.17.0.tar.gz

The four core C translation units and their headers are included, along with
the upstream-generated builtin bytecode headers. The CLI and `quickjs-libc`
OS/filesystem/process modules are not built or exposed.

`build.zig` compiles the core with C11 and ordinary IEEE floating-point behavior;
the model kernels' `-ffast-math` flags must not be applied here. The Zig owner is
`src/server/program_runtime.zig`. Update the source files and generated headers
together from a pinned upstream release.
