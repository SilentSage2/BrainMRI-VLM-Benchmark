# MRI-VLM-Small Model Card

- **Status:** untrained architecture smoke
- **Intended role:** controlled answer-only versus voxel-grounded MRI-VLM comparison
- **Clinical status:** research only; not a medical device

## Architecture

The current implementation applies a shared 3D convolutional encoder independently to
FLAIR, T1, T1-Gd, and T2 volumes. Learned sequence embeddings preserve modality identity.
An explicit availability mask excludes missing sequences before spatial and pooled fusion.
A token embedding averages non-padding question tokens, and language-conditioned fusion
feeds an answer classifier plus a dense voxel-evidence head.

The implementation is deliberately tiny and dependency-light enough for CPU shape and
gradient tests. It is not yet the frozen V1 training configuration and has no pretrained
weights.

## Verified behavior

- output tensor shapes are stable for batched 3D inputs;
- answer and evidence losses produce finite gradients for every parameter;
- checkpoint serialization round-trips exactly;
- changing a masked sequence cannot change either model output;
- an example with every sequence missing is rejected.

## Unverified behavior

No real MRI has been loaded, no optimization run has been performed, and no accuracy,
grounding, robustness, calibration, or clinical claim is supported. Preprocessing,
question tokenization, pretrained initialization, answer vocabulary, and the final width
remain V1 decisions.
