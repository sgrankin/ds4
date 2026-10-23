# Native persistent Metal tensor ownership

Item 72 moves persistent tensor ownership/policy into a native owner while
retaining narrow Objective-C ARC/driver leaves. The retained C-resource control passed its extended boundary fixture. Native
ReleaseFast and Debug builds are clean, and all fifteen selected GPU rows pass.
This qualifies the resource boundary and existing numeric cases; final
application/runtime performance is reported separately, with no speed claim here.

## Actual boundary

The [native resource owner](../../../src/metal/resources.zig) owns admitted
allocation sizes, view offset/range policy, live-handle membership and
live/peak accounting, CPU read/write/fill/copy decisions and the distinction
between an existing command batch and direct CPU access. Its
[public facade](../../../src/metal/resources_entry.zig) preserves the existing
GPU API. A small [ARC bridge](../../../src/metal/resources.inc.m) creates/releases
retained tensor handles, borrows immutable regions/contents and encodes the
existing blit operation. The [shared header](../../../src/metal/resources.h)
describes regions and diagnostic totals rather than copying an Objective-C
class layout. SDK mutex bindings replace guessed platform layouts.

A view independently retains its backing MTLBuffer. Freeing a parent/view
must not retire another alias; views do not double-count the backing allocation.
The free path checks live-handle membership before an Objective-C dereference.
Other operations keep the public caller-retains-handle contract, and borrowed
contents remain valid only while that owner stays live. GPU cleanup discards
tracking metadata; it is not a general capability to keep using handles after
cleanup. Existing command submission and shaders remain outside this owner.

## Control and qualification plan

The retained control uses original Metal source SHA-256
`f48c3b62a07436b8cd6262c7a3f478a63cdaceada4d377372474bb919604f241`
and the previously qualified native engine/server/agent owners. No native
resource flag was selected for it. The [recorded build](evidence/control/control-build.json)
passed in 2.789 s with empty stdout/stderr, producing all five applications and
the existing command-memory/Metal fixtures under
`.workspaces/zig-metal-resources-20261022/control/bin`. This is a cached
integration build, not a matched build-speed result.

The extended real public-API boundary fixture checks nested views, the final
alias's lifetime, zero ranges, a real blit and double-free handling. Root runs
the C control before selecting native resource ownership. Numeric Metal
qualification uses `DS4_METAL_MATH_SAFE=1` for both controls because the
[existing default fast-math FP4 oracle discrepancy](../zig-metal-publication-hour9-20261022/README.md)
was already reproduced on C/native before this item. That diagnostic setting
is separate from runtime defaults and later timing settings; no shader/math
repair is introduced to make this resource port pass.

Root owns serial compiler work and all guarded GPU/model checks. This report
retains bounded small commands, raw results and pressure summaries for the
completed resource campaign. No new fixture framework, model weights,
executable binaries or full cache backing stores are checked in. Final build
and runtime measurement scripts remain frozen in the separate
[performance folder](../zig-engine-final-20261022/README.md) until root releases
the final measurement window.


## Initial boundary/build results

The retained C-resource [command-memory test](evidence/control/control-copy-test.json)
passed in 7.387 s with Metal validation enabled, before native resource
activation. Its raw output and separate
[pressure guard](evidence/control/copy-guard/result.json) are retained. This
establishes that the added public-API boundary assertions hold for the actual
C resource implementation first.

The first native ReleaseFast install/fixture build passed cleanly in 5.247 s;
Debug passed cleanly in 2.003 s. Both had empty stdout/stderr. Exact commands
are retained under [evidence/native/](evidence/native/). These cached build
observations are not full-build performance measurements, and the builds alone
do not establish runtime resource compatibility.

The [published qualifier](qualify.py) serializes the existing real command-memory
fixture in `default`, `row`, `session`, `batch` and `big` modes for C/native,
with bounded `default`/`big` coverage in Debug. Each enables the Metal validation
layer. It then runs the existing complete numeric Metal fixture for all three
selections with safe math. It preserves the first failure and returns without
retry; root wraps the whole loop in the existing pressure guard. The only
publication edit is resolving ROOT from this checked-in path. The original
[workspace version](evidence/qualification/workspace-qualify.py) is retained
because it is the command path in the actual campaign log. Algorithms,
case order and settings are unchanged. The completed GPU results follow below.


## Completed GPU matrix

All [fifteen rows](evidence/gpu/result.json) passed on the first run; the
[outer campaign](evidence/gpu/gpu-campaign.json) completed in 56.948 s with exit
zero. The matrix consists of twelve validation-layer command-memory cases
(five C, five native ReleaseFast and two native Debug modes) plus three complete
existing numeric Metal runs with safe math. Exact binaries, arguments,
environment settings, stdout/stderr and per-row process times are retained.

| Selection | Validation-layer modes | Complete numeric fixture |
| --- | --- | --- |
| C resource control | default, row, session, batch, big: all pass | Safe math: pass |
| Native ReleaseFast | default, row, session, batch, big: all pass | Safe math: pass |
| Native Debug | default, big: both pass | Safe math: pass |

The real public API checks exercise nested views, survival through the last
alias, zero-length ranges, actual blits and duplicate-free handling. Numeric
success means the existing fixture's assertions pass in each selection; it is
not a claim that every C/native test output was independently dumped and
byte-compared. No shader, arithmetic default or fixture oracle was changed in
this resource item. The full numeric diagnostic uses the explicitly recorded
safe-math setting for all three implementations; later runtime measurements
retain their separately recorded defaults.

The [GPU guard](evidence/gpu/guard/result.json) completed with child return code
zero. All [55 samples](evidence/gpu/guard/pressure-summary.json) were normal
with zero swap growth, with peak wired/compressed memory of 4.821/8.677 GiB.
The earlier C-boundary guard is retained separately. No model was opened by
this matrix. Root subsequently owns same-source application/CPU builds,
ordinary inference and the integrated six-turn agent check in the
[final report](../zig-engine-final-20261022/README.md).

The [resource-source author snapshot](evidence/qualification/prepared-resource-source.json)
retains the five actual owner/facade/header/ARC source hashes and selected
native Metal C hash, all six verified against the active files after the matrix;
the tested executable hashes and exact commands identify this campaign.
The final whole-checkout source/build snapshot is deliberately reported with
those final measurements rather than guessed from a source count here.
Neither completed GPU-fixture times nor cached build times establish a
resource-port speed improvement.
