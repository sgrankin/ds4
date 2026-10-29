# Contributing

Choose checks that exercise the behavior your change can affect. In the change
notes, record the revision, commands, machine and backend, model and
quantization where relevant, results, failures and skipped checks. Describe
what was actually exercised; a model-free fixture does not qualify a full
model or a hardware backend.

Start with [project direction](docs/project.md) and the [source map](src/README.md).
The [build and test guide](docs/development/build-and-test.md) owns current
prerequisites, focused checks and diagnostics. Use the matching model's test
vectors and quality checks when changing inference, templates or quantization.
For release work, select the applicable checks in the [qualification guide](docs/development/qualification/README.md).

For performance changes, choose a fixed baseline and workload, then use the
[benchmarking guide](docs/development/benchmarking.md) and [lab methodology](docs/lab/methodology.md).
Comparative performance claims require at least five measured samples per side,
alternating order, with all samples and variation reported. Measure practical
quality alongside speed; approximations and output degradation can be useful
tradeoffs. Run one model-opening job at a time under the pressure guard. Keep
raw output local and ignored, and retain concise reports and reproduction
instructions in the repository.

Follow the [writing guide](docs/writing-guide.md) when updating documentation.
Update the manual that owns changed behavior and record experimental reasoning
in the lab journal. Keep contributions focused on the selected outcome.
