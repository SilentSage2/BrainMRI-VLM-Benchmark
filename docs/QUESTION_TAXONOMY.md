# V0 Question Taxonomy

All questions are generated from a versioned rule and a subject's label mask or image
geometry. Templates are paraphrased only after a question-only control measures shortcut
risk. Each generated record stores the rule version and target quantity in its manifest.

| Type | Example operation | Answer | Voxel evidence | Required counterfactual |
|---|---|---|---|---|
| Presence | Is enhancing tumor present? | yes/no | enhancing region | remove/add region |
| Laterality | Which hemisphere contains more whole tumor? | left/right/midline | whole tumor | reflect mask |
| Relative volume | Which tumor region is larger? | region name | union of compared regions | swap volumes |
| Enhancing fraction | What fraction of whole tumor is enhancing? | decimal | whole + enhancing | scale enhancing region |
| Cross-region comparison | Is tumor core more than half of whole tumor? | yes/no | both regions | cross threshold |
| Unanswerable | Compare with a prior scan that is not provided. | abstain | none | provide/remove prerequisite |

## Answer rules

Presence uses a frozen minimum-volume threshold to avoid a single noisy voxel changing the
label. Laterality requires an audited left-right orientation and an explicit midline band.
Numeric ratios are computed from physical voxel volume, not raw voxel counts when spacing
differs. Boundary cases within a declared ambiguity margin are excluded or labeled
ambiguous before dataset generation.

## Evidence rules

Evidence is the smallest mask-derived voxel set sufficient for the operation. Comparison
questions include both compared regions. Unanswerable questions contain no answer and no
evidence. Evidence masks are content-addressed and never cross subject splits.

## Counterfactual rules

Each pair changes one target property while preserving question wording and unrelated
content. Both members share a counterfactual group and subject split. Evaluation requires
both answers and both evidence masks to pass; accuracy on only one member is inconsistent.

## Exclusions

V0 excludes diagnosis, tumor grade, survival, treatment response, causal clinical claims,
free-form radiology reports, and questions whose answer is not verifiable from the source
mask and registered MRI geometry.
