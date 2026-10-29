# Distributed qualification reference

Distributed inference must be qualified on the actual machines and transport. Retained source implements native two-rank Metal DSpark and DeepSeek V4.1 CUDA network TP with DSpark. Neither source presence nor a test plan proves a physical run passed. The following conditions and historical commands come from QA snapshot `70537d44` (2026-10-28); confirm they exist at the chosen historical revision. Current Makefile targets do not provide the old CUDA/ROCm build. Mac Metal qualification remains distinct from this historical platform plan.

## Hardware and transport prerequisites

Start workers before the coordinator. Test short and long prompts, complete route setup, clean shutdown and Ctrl+C after the current token or chunk drains. After session or KV changes, save and restore a distributed snapshot. Test actual TB5 links between the two M5 Macs where available and record Wi-Fi/VPN fallback. For physical RDMA identify the device and GID and verify RC transport; a management-network ping does not demonstrate RDMA. Test TCP too, stop a worker during exchange and require bounded failure.

## DeepSeek V4.1 CUDA network TP

Use two Sparks with matching revisions and Q2 files, one GPU per rank, resident expert shards and disk-only Engram. Do not combine network TP with `--ssd-streaming`, `--cuda-tensor-parallel` or `--quality`. Spark uses host-visible exchange buffers with staging fallback, not discrete-GPU GPUDirect. Peers must agree on protocol, expert split, verification and vocabulary head; incompatible peers must fail during handshake before inference.

### Transport and primitive checks

The retained plan built `tests/test_tp_commands`, `tests/test_tp_tcp`, `tests/test_tp_rdma`, `tests/test_tp_link`, `tests/test_cuda_tp`, `tests/test_cuda_tp_link` and `tests/test_cuda_ssd_batch`. Run command, TCP and RDMA tests with ASan/UBSan and CUDA tests with Compute Sanitizer. Tiny socket buffers, stalled or disconnected peers and half-close must fail promptly. Test both directions, mailbox ring wrap, delayed credits, oversized SYNC then short commands, immediate STOP, peer exit and asymmetric mailbox negotiation. Inject head-row and verification-window failures: failed contexts must refuse further commands without touching transport; recovery requires a new connection. Replay readiness and head rows must belong to the current exchange after rollback.

Test owned Q2 and MXFP4 batches with both rank halves, empty contributions, poisoned scratch, guards, 384 experts and small and large row boundaries, including 8191/8192/8193. Small V4.1 rows must match each rank's scalar partials exactly. Check split projections, attention outputs and both vocabulary head halves against full outputs, including ties and nonfinite values; a missing worker half must fail rather than expose stale logits. Check staging growth, registration cleanup, rebind and failure propagation. Test exact row payloads and graph negotiation when diagnostics are enabled on only one peer. RDMA completion tests cover 16/32/64 KiB negotiation and malformed completions; test an older peer, then bulk prefill, decode and bulk again on one connection.

### Model integration and physical qualification

Run the retained `tests/test_deepseek41_prefill --tensor-parallel-cuda MODEL speed-bench/promessi_sposi.txt LISTEN_HOST PORT RDMA_DEVICE GID` with a `--ctx 16384` worker, and its small-append variant. Compare full state and logits, cancellation, snapshots and both-rank prefix replay. Score general100, boundary17, medium continued12 and long continued9, including 32-token appends at 8/16/32K sparse frontiers; use paired scores and 64K/96K vectors. Exercise actual agent and server tool loops, cancellation and slot reuse. Five through eight active sessions exercise native CUDA TP batching; smaller groups use ordered execution. Eight 4K contexts fit the tested pair, while eight 8K workspaces must fail admission. Test isolated full-logit replay and reordered rows with the eight-session oracle and its MoE-disabled control. Decode graphs on/off must agree; diagnostic logging should show all 120 scalar capture regions per rank, with network gates outside capture.

### Historical reference

The October 2, 2026 64K-context reference was 410.92 t/s fresh 32K prefill and 313.70 t/s for the next 8K, with 17.97 and 17.93 t/s decode respectively. These are single measurements, excluding prefix replay from append time. Record both hosts' CPU affinity, clocks and temperature; the tested CPU IDs need not generalize. Historical quality means were 0.364937088, 0.401415795, 0.523971083 and 0.566939577 NLL for the four sets above; measurement revision was not separately identified. Eight-session native throughput was roughly 28 aggregate t/s, not per-client. Expect approximately 80.57 GiB weights per rank in that plan, including 71.19 GiB derived expert artifacts; monitor allocations, host memory and swap rather than retaining duplicate raw or unowned experts.

For physical GLM TP, including its explicit RDMA, expert ownership, long-task and vision gates, see [GLM qualification](glm.md).
