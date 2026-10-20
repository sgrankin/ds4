# Keep the latest user message fresh during suffix reuse

The opt-in agent suffix selector now requires a recognized latest user role
marker in the edited token stream. It raises the minimum fresh tail to include
that marker and all following tokens; retained frontier endpoints are exclusive.
Absent/unsupported roles or insufficient eligible earlier frontiers take exact
fallback before snapshots, source capture or graph mutation. The configured
128-token default and 64-token minimum remain lower bounds.

The model-free real-agent transaction suite passes in production and private
helper ASan/UBSan builds, covering 16 same-position/even-shift marker boundaries,
missing and unsupported roles, earlier versus latest roles, no-mutation fallback,
existing cancellation/rollback/budget/export/zero-KV persistence and default/override
behavior. Existing MAIN core and auxiliary objects were linked without rebuild;
ASan/UBSan instruments the private helper/agent, not those old objects. Darwin
leak detection is disabled. No model, tokenizer or GPU initialization occurred.

The previously recorded actual-agent reuse endpoint 2582 precedes its latest
user marker 2583, so this new condition permits that recorded geometry. The
old native run is not a execution test of the new source. This change remains
isolated pending review. It enforces query freshness; it does not remove old
conditioning from cached hidden states or establish forgetting.
