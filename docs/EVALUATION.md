# Evaluation Contract

Text answers use normalized exact match and numeric answers use frozen absolute/relative
tolerances. Evidence is scored with voxel Dice. Grounded answer accuracy requires both a
correct answer and evidence Dice at or above the declared threshold.

An unanswerable question is correct only when the VLM abstains. Any non-empty answer counts
as hallucination. Counterfactual consistency requires both members of a subject-grouped
pair to be correct.

Scores are first aggregated per subject, then across subjects. Bootstrap resampling uses
subjects. Every model receives identical questions, volumes, modality masks, preprocessing,
and decoding rules. Before scoring, the evaluator verifies subject isolation, fingerprint
isolation, example coverage, answer/evidence consistency, and fixed label mapping.
