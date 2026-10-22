# Item96 — Worker communication transactions

Native Communications borrows the actual Worker mutex, condition, wake pipe and buffers. It owns publication/status/progress, interrupt/cancel, browser approvals, queued-user drain, save/compact/power flags, busy submission, stop and output ownership transfer. The one process-wide prefill-label counter lives once in the sole facade; original C algorithms remain.

Integration19:43–19:45UTC at3908baa9. Added same-fixture actual idle→PREFILL/busy rejection, progress, interrupt, output ownership and bounded pthread/condition/pipe handoff. C first, native ReleaseFast and Debug all pass Agent/terminal/PTY suites;17 stable captures match. Three string alternatives needed explicit C pointer types for varargs; retained initial compiler failure, no runtime correction. Existing assertions remain.

Actual model/worker-turn behavior remains the upcoming combined Turn gate. No latency or quality claim.
