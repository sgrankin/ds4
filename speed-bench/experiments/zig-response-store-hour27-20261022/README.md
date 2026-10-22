# Native durable Responses file store

Baseline `f587cb4a`, completed inside a one-hour ceiling on 2026-10-22.
The seven existing raw-file APIs move into144 handwritten Zig lines. A small
header guard selects prototypes or the original C implementation; production
and the unchanged standalone fixture share the actual selected native object.
JSON parsing, response/conversation semantics and model state remain C.

Scoped descriptor/buffer/temp cleanup preserves the first error. Standard ASCII
checks express the existing ID grammar. Constants,0/1/-1 statuses, output resets,
libc-owned NUL-terminated buffers,0700 directory/0600 file modes,64MiB limit,
EINTR handling, fsync/close/rename and old-inode behavior are unchanged. No
filesystem framework, storage format or schema migration is introduced.

## Qualification

All five C/native programs build without warnings. Existing standalone store
and actual server fixtures pass C/native × Debug/ReleaseFast: eight runs. They
cover raw roundtrip, missing results, permissions, IDs and atomic replacement
while an old descriptor remains open. The normal C reference remains available.

```sh
tools/zig-build --prefix .workspaces/store-native install response-store-test server-test
.workspaces/store-native/bin/test-response-store NEW_STORE_DIRECTORY
.workspaces/store-native/bin/ds4_test --server
```

The real server workflow passes both C-writer→native-reader and native-writer→C-
reader restarts, with distinct process IDs. Each checks exact POST/GET public
objects and bytes, restart GET, a previous_response_id continuation, child GET,
DELETE and404 responses, empty store after deletion, and actual directory/file
modes. Four model calls return exactly OLD|ORCHID42|37 or NEW|ORCHID42|37 with no
caps. Child cached_tokens is0: the new process reconstructs the remembered facts
from the stored response and follows the new instruction marker.

This workflow deliberately uses store:true. The earlier combined SSD driver
uses store:false and would not qualify this particular file-store replacement.
Both control and candidate otherwise use the native stack. Model1017 uses a
48GiB expert cache, context8192, chunk512, route/suffix off, hot cache0. All
pressure samples remain normal without swap growth. These are compatibility
checks, not a latency comparison.

```sh
python3 -B speed-bench/ideas/pressure_run.py NEW_GUARD -- \
  python3 -B speed-bench/experiments/zig-response-store-hour27-20261022/qualify.py \
  --c-server CONTROL_SERVER --native-server NATIVE_SERVER --output NEW_RESULTS
```

Decision: **retain**. [Results, commands and hashes](results.json),
[actual cross-restart caller](qualify.py). Original serialized records, requests,
responses and logs remain under `.workspaces/zig-response-store-20261022/`.
