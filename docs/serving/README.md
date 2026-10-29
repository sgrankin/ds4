# Serving and clients

The server turns client requests into model work. These manuals explain the
wire contract, conversation lifetime and scheduling; the model manuals own
model-specific prompts, inference state and kernels.

- [Serving models](overview.md): startup, endpoints, slots and scheduling.
- [Stateful Responses](responses.md): HTTP/WebSocket continuation, stored
  ancestry, conversations, replay and recovery.
- [Connecting agent clients](clients.md): client configuration and compatibility.
- [The local agent](agent.md): tools, editable context, archives, compaction and
  local sessions.
- [Sampling and speculative decoding](sampling-and-speculation.md): output
  constraints, sampling semantics and draft verification.

Use [storage](../storage/README.md) for persistence and memory budgets, and
[model manuals](../models/README.md) for each model's eligibility limits.
