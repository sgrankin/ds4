# Resident expert functional map: actual publication

The measured expert-output map is assembled and independently byte-audited. The immutable map contains all640 captured input probes and245,760 projected expert/input outputs, covering40 layers ×384 experts ×16 probes ×16 features.

| Artifact | Exact extent | Mode |
| --- | ---: | ---: |
| DS4FMAP1 map | 21,140,879 bytes | 0444 |
| DS4FQAL1 qualification | 17,638 bytes | 0444 |

Map SHA: `c9328addaeac914f666c618a5564b87cd8533b75531f8430572fc2cbfabda17e`. Qualification SHA: `da87dd4f73aa80eb2e029e9ab1c05f6005abb644f76eb8516ebb476a0a33a82e`.

ROOT's separate fixed-format reader checked both complete headers, checksums and extents; rebuilt the5,832-byte native tensor binding from accepted capture data; checked all120 tensor roles,16 opaque histories and640 capture records; and verified exact complete probe/signature payload equality. All3,276,800 FP32 probe values and3,932,160 FP16 signatures are finite. Sixteen synthetic qualification fixtures also passed, including mismatched descriptor, kernel, shader, policy and malformed-artifact refusals.

The qualification records the accepted historical model descriptor, complete expert-kernel source identity, construction/build/release/audit provenance and fixed protected1 policy. Native loading must independently reconstruct its current owning FD identity and native tensor ranges before accepting it. Native integration, original coefficient-bit preservation and quality measurements remain pending; publication supplies data rather than evidence that substitutions help.

No model, tokenizer, GPU or current GGUF metadata access occurred during this publication. The first ROOT administrative attempt referenced `root-run.stdout` instead of the actual `stdout`; it was retained before capture/signature reads. The corrected ROOT path did not change candidate source or repeat the native scan. All100 MAIN runtime and shader files remain byte-identical; defaults are unchanged.

The large immutable map remains in the persistent workspace at `.workspaces/idea-resident-functional-map-publication/speed-bench/ideas/artifacts/resident-functional-map-publication-v2/functional.ds4fmap1`. The qualification and compact proofs are copied here for review.
