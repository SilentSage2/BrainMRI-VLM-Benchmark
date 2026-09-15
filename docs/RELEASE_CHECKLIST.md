# GitHub Release Checklist

Do not publish or pin an empty scaffold. Create the independent GitHub repository and push
the local history only after all of these are true:

- the research-substance audit is cleared with evidence, not documentation alone;
- at least one named VLM baseline runs end to end;
- a strong segmentation-to-symbolic-QA baseline runs end to end;
- an unconditional spatial-auxiliary control isolates grounding from extra supervision;
- a real-data smoke evaluation produces traceable metrics;
- dataset acquisition and preparation commands are documented without redistributing data;
- tests, lint, typing, and the CPU protocol smoke command pass;
- the README clearly separates completed results from planned experiments;
- no MRI data, derived volumes, model weights, credentials, or run directories are tracked;
- compute, license, research-only use, and clinical non-goals are visible.

After the gate passes: create `SilentSage2/mri-vlm-grounding`, push `main`, enable CI, add
accurate topics (`medical-imaging`, `mri`, `vision-language-model`, `multimodal-learning`,
`grounding`), and create an MVP tag only after the reproduction command is verified.

## Gate status — 14 September 2026

The repository qualifies for an explicitly labeled **research preview**: it has a real-data
residual 3D MR baseline, a reproducible matched 3D MRI-VLM run, the required unconditional
spatial-auxiliary control, traceable negative results, and a clean-install protocol smoke.
The test split was not read by any model run. Ruff, mypy, and all 84 tests pass in both the
development environment and a clean Python 3.12 editable install.

The larger 64/16 development cohort, three seeds, QA V1 question-only control, fixed-slice
VLM diagnostic, modular baseline, and Figures 4/5 are complete. The external M3D-LaMed
baseline was audited but not executed and therefore cannot support a foundation-model
comparison. The **MVP tag and ISMRM evidence gate remain closed** until the explicitly
authorized one-shot held-out run is complete. Public text must not claim that
question-conditioned grounding improves answers or that a foundation model was tested.
