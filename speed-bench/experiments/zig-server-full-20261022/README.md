# Complete native serving integration

All nine final serving owners are selected together: durable conversations, request materialization, decode tracking, live reuse, persistence, generation, serving, boot and images. The production server suite passed in ReleaseFast and Debug for each selection. The actual model HTTP workflow then passed, restarting from ReleaseFast into Debug with the same durable response and SSD cache directories.

The existing [HTTP driver](../zig-server-protocols-http-20261022/qualify_http.py) exercised OpenAI and Anthropic responses, streaming, simultaneous clients with actual two-session decode batches, disconnect cancellation after a text delta, a surviving request, durable parent responses and conversation/item CRUD. Restarted Debug generation recovered actual SSD checkpoints. Both server processes exited zero. The retained result and per-request records are the observations; ordered process times are not a latency comparison.

Native inference/slot pthreads use an 8 MiB stack after the actual Debug Agent exposed a default-stack guard fault in native prefill. This real Debug HTTP restart validates that serving path. Client connection threads retain their existing stack allocation. The change does not alter model arithmetic or shaders.

The unchanged pressure guard completed successfully. Useful cache payloads and binaries remain in the ignored workspace; only commands, source identities, requests, results and compressed traces are versioned. Actual vision HTTP qualification remains pending here and is covered separately after the public vision transaction is selected.
