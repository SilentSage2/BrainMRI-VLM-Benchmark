# V1 Model Selection

## Decision

Use two distinct model roles rather than forcing one external checkpoint into every
experiment.

### External 3D VLM baseline: M3D-LaMed-Phi-3-4B

The official M3D project supports 3D medical VQA, referring expressions, and segmentation.
Its Phi-3 checkpoint is approximately 4B parameters and the model page declares Apache
2.0. The published quickstart expects a single volume shaped `1 x 32 x 256 x 256` and uses
remote custom model code.

This is suitable as a named single-volume 3D VLM baseline after a security and memory
preflight. It is not treated as a native four-sequence fusion model. Results must identify
which MRI sequence it receives and may not be compared to four-sequence models as though
the inputs were identical.

- Code: https://github.com/BAAI-DCAI/M3D
- Checkpoint: https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B

### Controlled model: MRI-VLM-Small

The primary answer-only versus grounded comparison uses our own compact architecture:

1. one shared 3D visual encoder applied separately to each available MRI sequence;
2. learned sequence-identity embeddings and an explicit availability mask;
3. masked cross-sequence fusion producing bounded visual tokens;
4. a compact language encoder/decoder for the frozen question taxonomy;
5. an optional voxel-evidence head attached to the same fused representation.

M3D-CLIP's Apache-2.0 0.2B 3D encoder is the first pretrained initialization candidate.
A randomly initialized small encoder remains the dependency-free control. The precise text
backbone is unresolved until a memory/license preflight is recorded.

## Rejected as the first baseline

RadFM supports 2D/3D and multi-image inputs, but its official instructions recommend an
80 GB A100 for inference. That exceeds the MVP's accessible single-GPU target. It may be
reconsidered only if hosted compute is explicitly budgeted.

## Preflight gates

Before downloading weights or executing `trust_remote_code`:

- pin the exact model revision and inspect its code/config diff;
- record all transitive checkpoint and base-model licenses;
- estimate weight, activation, and input memory on the target hardware;
- confirm preprocessing semantics for MRI orientation, sequence identity, and intensity;
- run on synthetic non-medical input first;
- cache weights outside Git and record file hashes;
- verify that output parsing can represent abstention and evidence.
