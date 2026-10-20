# Item 43: opaque turn IDs

Turn headers now use random 64-bit IDs rendered as 16 lowercase hex characters,
for example `[[DS4_TURN a3c91e704b82d6f0 role=assistant]]`. Conversation order
comes from the file. An edited or moved turn keeps its ID; fresh turns receive
fresh IDs. The metadata remains excluded from conversation tokens/KV.

This is the user's format correction to Item 42's monotonically allocated
decimal IDs. The existing explicit-ID importer already accepts reordered
blocks. Removed its allocation counter, used the standard `getentropy` call,
and updated the standard-parser editing recipe to use string IDs. New IDs are
checked against the current turn map and newly allocated IDs. No allocation
history service was added; identities remain worker-local.

Budget: 2026-10-09 23:12–2026-10-10 00:12 UTC cap; checks complete by 23:15 UTC.
Base revision: `2f8d957953a8e2852cf9b467136c30d99db63f38`.
Tested source/binary hashes are recorded in [checks.json](checks.json).

Validation passed:

- `make -j2 ds4-agent ds4_agent_test` and `./ds4_agent_test`.
- Existing strict-C11 CPU suffix mock: deletion, summary, batch edits, unique
  appended IDs, literal-header escaping, protected prefix and metadata excluded
  from KV. Added one reorder-and-edit round-trip preserving ID/body associations.
- Executed the exact embedded recipe: removed the selected hex ID, retained the
  other block and protected prefix byte-for-byte, and printed zero bytes.
- Source diff whitespace check.

The first build missed the declaration's `sys/random.h` include; `build.log`
retains that error and `build-fixed.log` records the successful build. This
format follow-up used CPU checks; Item 42 retains the earlier real-model archival
screen with decimal labels. No new speed or output-quality result is claimed.

Restart `ds4-agent` with the existing `--live-context-file` flag to receive the
new headers and instructions. Stop after these concrete checks.
