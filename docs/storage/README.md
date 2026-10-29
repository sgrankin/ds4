# Storage and memory

Model weights, inference checkpoints and conversation history have different
lifetimes and recovery requirements. A cache miss may cost recomputation;
losing the only durable conversation record loses information.

- [Streaming model weights](expert-streaming.md): expert residency, SSD reads
  and the RAM budget needed to execute a model.
- [Inference checkpoints and conversation storage](session-cache.md): RAM/SSD
  checkpoint storage, logical durability, restore and eviction.

Each [model implementation](../models/README.md) describes the state it can
save and the combinations eligible for reuse. The
[Responses contract](../serving/responses.md) explains client recovery.
