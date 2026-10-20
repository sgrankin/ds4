Two unused practical confirmation tasks, authored before candidate selection.

Use spec.json task order join, then window_argmax. Keep spec/fixtures/grade outside
advertised project; expose only config.json and the task prompt/system. Each task
requires one read, one write, and clean final stop under the existing helper.
Private fixtures contain exactly161 cases per task: public sample,16 edges,144
fixed-seed inputs. Gold methods are indexed join versus nested pair enumeration,
and per-window max/index versus monotonic queue. The grader reuses original AST,
builtins, JSON type and input-mutation restrictions. CLI: grade.py KEY solution.py.
The bounded stdlib CPU check passed: both independent golds agree on all161 inputs
per task, and both clean reference solutions grade161/161. See check-results.json.
Model execution results are kept separately in ../practical-results/.
Capped or tool-contract-failed attempts stay failed even if their code passes.
