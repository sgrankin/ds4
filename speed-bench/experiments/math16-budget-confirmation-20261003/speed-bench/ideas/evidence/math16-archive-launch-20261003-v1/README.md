# Released math16 first-attempt archive

The exact ROOT-authorized archive command completed once with rc0. The original
model producer/pressure guard completed with **rc3**, retained unchanged because
some first attempts were INVALID. The archive contains all 48 attempts on 16
fixed problems, full text/token/RNG/piece/prompt traces, the original scoring
summary, ROOT's independently pinned 48-attempt audit, normal-pressure release,
source/build/protocol provenance and first preparation/administrative refusals.

| Arm | Completed correct | Wrong | INVALID | Caps | Forced | Generated tokens |
|---|---:|---:|---:|---:|---:|---:|
| Native before | 9 | 0 | 7 | 7 | 0 | 18,826 |
| Budget 1024 | 10 | 2 | 4 | 3 | 8 | 17,062 |
| Native after | 9 | 0 | 7 | 7 | 0 | 18,826 |

Both native brackets repeat exactly under the original typed/token/RNG/piece
controls. Budget1024 rescues two native failures and loses one native completed
correct answer. Generated work falls by 1,764 tokens (9.3700%); there are also two
wrong completed answers and four INVALID attempts. All cases remain included;
there is no posthoc rescoring, policy tuning or outcome-based subset.

The tool's complete archive is the sibling
`math16-budget-confirmation-archive-20261003-v1`. Its top manifest indexes 337
physical payload files and thirteen directories, excluding only itself. Five
payloads greater than 2 MiB retain original SHA/size/mode/path pointers, including
the qualified caller, two objects, compiler and dSYM. All 48 text/token/prompt
artifact hashes, every copied original, all five external pointers, 147 static
provenance files and frozen MAIN runtime100 were independently rechecked after
copying. The publication manifest here additionally indexes the tool manifest,
launch log/result and this review without modifying the tool's original index.

This is one fixed confirmation set and one seed. Raw files are text bytes, not
full-vocabulary logits. Token reduction is work accounting, with no speed, SSD,
equal-quality, cross-prefix KL/NLL, quantization-causality, broad generalization
or production-default claim. No model/helper/guard was executed by the archive.
