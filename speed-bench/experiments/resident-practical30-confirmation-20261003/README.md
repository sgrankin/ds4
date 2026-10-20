# Resident-expert confirmation on ten unused tasks

The preregistered protected substitution policy did not improve aggregate strict task success: native-before, substitution, and native-after each passed 4/10. Substitution added one capped attempt and 45 sampler decisions. The native brackets matched complete logits, sampling decisions, code artifacts, grade outcomes and native IO controls; grade diagnostic text was excluded by the original contract.

| Arm | Strict successes | Code passes including capped | Caps | Sampler decisions |
| --- | ---: | ---: | ---: | ---: |
| Native before | 4 | 4 | 5 | 16,838 |
| Protected substitution | 4 | 4 | 6 | 16,883 |
| Native after | 4 | 4 | 5 | 16,838 |

The candidate lost `glob_full` to the original 2048-decision cap and gained `lru_trace` under the fixed grader. The native LRU code failed with `NameError: iter is not defined`: its `next(iter(cache))` uses an ordinary Python builtin omitted by the original grader. This is a grader-compatibility difference; the recorded rescue does not establish algorithmic improvement. Original grades are retained without rescore. The current policy therefore remains opt-in.

This confirmation follows the earlier disclosed calibration experiment (7/10 native versus 9/10 protected substitution). The unused-task result does not reproduce that net gain. Ten tasks and one seed cannot establish general quality or speed. All capped, absent-code and wrong-code first attempts are retained; no subsets, retries or post-result policy changes were used.

Policy: gap .30, individual mass .12, rank 24, one replacement, all six original router coefficient bits retained. Each task had three tool rounds and a 2048-decision total cap, with the original first code checked by the original 161-test grader.

The independent CPU audit verified all full FP32 logits and original evidence, without rerunning the model or executing candidate code. Its complete 527-file durable archive is frozen at `98b45265ba49ef3ccb6da5871153c95f2d829ac2` in `.workspaces/idea-practical30-result-archive`; 31 larger original payloads remain protected references. The archive also preserves 1,404 historical evidence references without duplicating them. Native raw files and original first grades remain available there.

Memory pressure stayed normal throughout all 3,647 samples. Peak wired-plus-compressed memory was 79.641 GiB; measured maximum swap growth was 0.001884766 GiB (about 1.93 MiB). These are diagnostic observations, with other small CPU qualification work overlapping; no speed comparison is claimed.
