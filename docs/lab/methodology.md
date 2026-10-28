# Experimental methodology

A lab entry explains what we tried, what the measurements show and what we
learned. It gives an engineer who was not present enough context to understand
the decision and its limits. Use a descriptive, date-prefixed filename such as
`2026-10-28-checkpoint-overhead.md`. The format below is preferred, not a rigid
schema: combine or omit sections when that makes the story clearer.

## The experiment loop

Choose one outcome and a question it can answer. Follow the selected work in
[the experiment queue](../../speed-bench/ideas/next-experiments.md); the
[project page](../PROJECT.md) describes the current direction. Give each item
one hour, including preparation, build, runs and analysis. Freeze a meaningful
baseline, change one thing, measure, inspect and decide. Stop when the question
is answered; a side idea or a completed experiment does not start another.
Reuse existing tools and stop improving a harness once it can answer the question.

Compare the same model checkpoint and quantization, workload, sampling and
other relevant settings. State the baseline and candidate revisions and clear
inherited experimental settings that could change a control. Include ordinary
and favorable workloads as well as stress or adverse cases relevant to the
question. For a tunable feature, plot the speed and quality tradeoff across
settings. Keep experimental runtime behavior opt-in.

Comparative performance claims require at least five measured samples per side,
with alternating order. Retain every sample, including first runs, and report
absolute values and variation as well as any relative change. Inspect paired
changes and order effects; a small slowdown within noisy variation is not by
itself a reason to reject a change. One-off timings are useful for smoke checks
or diagnosis, not comparative speed claims. The [index layout follow-up](../../speed-bench/experiments/index-five-samples-20261027/README.md)
shows how an apparent slowdown can disappear in an alternating comparison.

Define the timed scope before running. Startup, bootstrap and application
checkpoint caches, expert residency, OS file cache, inference phases and a
complete request or task can all have different costs. Record whether each is
included, warm or cold, and whether an OS cache flush actually occurred. Logical
reads are not necessarily physical SSD traffic. Profiling can change timing;
use it to locate costs and distinguish profiled observations from final
profile-off latency measurements. Pause unrelated work during those measurements.
[Performance and benchmarking](../PERFORMANCE.md) describes the existing
context sweep and its scope.

Run one heavy model job at a time under the existing
[pressure guard](../../speed-bench/ideas/pressure_run.py), including commands
that open a model for diagnostics. The guard rejects a second cooperating
guarded invocation; it cannot prevent an unguarded job from running. Check for
other heavy model work before starting. It monitors memory pressure and swap;
report relevant conditions and interruptions. Large RAM use
is expected, so judge pressure and swap behavior rather than an arbitrary RAM
cap. Preserve failed runs, caps, fallbacks and unexpected observations alongside
successful results.

## Quality and interpretation

Approximations and output degradation can be worthwhile. Measure practical
correctness on tasks suited to the change: check actual answers, tool behavior
and failures, rather than requiring identical wording. Report quality together
with speed and resources, including the limits of the tested tasks.

For fixed teacher tokens and identical histories, delta-NLL is the candidate's
negative log likelihood minus the baseline's, usually averaged in nats per token.
Positive values mean lower likelihood for those reference tokens;
`exp(delta-NLL)` is the teacher perplexity ratio, not a task failure rate.
`KL(baseline || candidate)` compares their full-vocabulary probability
distributions on the same histories and describes distribution drift. It does
not establish whether a response is useful; partial top-token lists cannot
supply full KL, and paired KL does not describe diverging sampled histories.
Report relevant tails or per-case results as well as means. Neither metric is
an automatic rejection threshold or a substitute for practical task checks.
The [quality probe notes](../../speed-bench/ideas/README.md#paired-quality-probe)
give the project's measurement details.

## Preferred lab entry

Start with the answer when it is known, then use these sections to tell the
story. A short entry can be more useful than a comprehensive checklist.

### Context

Explain the question and why it matters, relevant earlier attempts and how
this experiment differs. Identify the hypothesis as a hypothesis, and link to
prior evidence when it helps a reader understand the choice of baseline.

### Setup

Describe the hardware, model and quantization; baseline and candidate revisions;
workload and settings; what changed; and the timed scope and procedure. Include
a command or link to a reusable driver when useful. Explain cache and startup
conditions and any profiling that affects interpretation. Give enough detail to
understand the comparison without requiring a permanent archived harness.

### Findings

Show absolute results, all measured samples and variation where relevant, plus
practical quality and resource use. Keep failures, unexpected observations and
order effects visible. Distinguish what was measured from a possible explanation;
name any measurement that was not made. A compact table can make samples easy
to compare without replacing the explanation. Larger comparisons can summarize
the results in the entry and link to a complete sample record. The
[zero-edit checkpoint comparison](../../speed-bench/experiments/context-capture-20261028/NO_EDITS.md)
is an example of separating request time, startup and cache conditions.

### Conclusion

Say what was learned and whether the change was adopted, rejected or remains
unresolved. Explain the tradeoffs and evidence limits, and link to the
implementation and any updated technical manual or recipe. Avoid extrapolating
one workload to other models, machines or tasks. The lab entry is the narrative
result report; a linked sample record supplies detail when needed. Update the
owning technical manual's conclusion. A project-journal milestone, when useful,
should be only a sentence or two linking to the entry, rather than a second
account of its results.

## Evidence and tools

Keep the reasoning and the minimum evidence needed to assess the conclusion in
the entry, with links to detailed results where useful. Raw logs, benchmark
output and generated snapshots stay local and ignored under `.workspaces/` or
`speed-bench/`; commit concise reports, necessary fixtures and useful
reproduction scripts. Put reusable tools in `speed-bench/`. Remove disposable
scripts when finished or leave them ignored with local artifacts. Record the
measured revision so historical source can be inspected without requiring every
archived experiment driver to remain runnable forever. Retain failures and
limitations in the report even when raw files are local.
