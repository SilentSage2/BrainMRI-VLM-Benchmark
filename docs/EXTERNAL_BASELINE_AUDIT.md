# External Foundation Baseline Audit

Audit date: 2026-09-14. Decision: **do not execute M3D-LaMed-Phi-3-4B on this CPU host or
send MRI volumes to a hosted API.** Retain it as an important unexecuted limitation.

## Candidate identity

- Official project: BAAI-DCAI/M3D, main branch, accessed 2026-09-14.
  https://github.com/BAAI-DCAI/M3D
- Candidate model: `GoodBaiBai88/M3D-LaMed-Phi-3-4B`, repository revision visible at
  audit time with verified commit `329bed5`.
  https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B
- The official project links this exact Hugging Face repository and describes a 3D medical
  VLM supporting VQA, positioning, and segmentation.

## License and supply-chain findings

- The official code repository is detected as MIT licensed.
- The Hugging Face repository is tagged Apache-2.0, but its model card is only 31 bytes and
  does not provide a complete training-data, intended-use, or limitation statement.
- Loading requires `trust_remote_code=True`; the repository includes custom Python modeling
  code and a pickle-format `model.bin` in addition to safetensors. A future execution must
  pin a revision, review the custom code, prefer safetensors, and record file hashes.
- M3D-Data combines multiple public medical sources. Exact subject-level overlap with MSD
  Task01/BraTS has not been ruled out, so a leakage audit is mandatory before interpreting
  performance on this cohort.

The model tag and the code license do not by themselves resolve the licenses of Phi-3,
SegVol, M3D-CLIP, or training datasets. These transitive terms must be recorded before a
redistributable benchmark bundle is created.

## Compute and protocol compatibility

- The model repository is 32.6 GB and contains approximately 16.2 GB of sharded model
  tensors, plus projector and other weights.
- The official quickstart selects CUDA and bf16; training instructions use eight processes,
  DeepSpeed, and explicitly set `use_cpu: false`.
- The documented image interface consumes a single `1×32×256×256` normalized volume,
  whereas this project evaluates four registered MRI contrasts and every non-empty subset.
  Collapsing four contrasts into one volume would not be a matched comparison.
- CPU inference would require a large download, reviewed custom-code execution, substantial
  RAM, and a new deterministic multi-contrast adapter. It is not a reasonable use of the
  current host and would still be methodologically non-matched.

## Data-transfer boundary

No MSD/BraTS volume, label, derived slice, subject identifier, or question-answer record may
be sent to Hugging Face inference, a hosted demo, or another third-party API. The public
dataset license does not remove the need for explicit user authorization for outbound data
transfer. This audit performed documentation lookup only.

## Reconsideration gate

Execute an external foundation baseline only when all of the following are available:

1. a local GPU host with sufficient memory and storage;
2. pinned, reviewed weights and remote code with complete hashes and license records;
3. a documented four-contrast adapter or a clearly labeled non-matched single-volume task;
4. a training-data overlap audit against MSD Task01/BraTS identifiers and sources; and
5. an evaluation manifest preserving the frozen QA V1, validation cohort, and test seal.

Until then, the correct limitation is: **the modular MR pathway outperforms the small
controlled VLMs evaluated here, but has not been compared with a large pretrained 3D
medical foundation model.**
