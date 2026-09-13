# PI-GNN Load-Flow — Code and Data Availability

Supporting code for "Explainable Physics-Informed Graph Neural Networks for Load-Flow".

## Structure
- `/original_implementation/` — the primary PI-GNN codebase (torch_geometric GCN, Newton-Raphson
  solver, N-1/N-2 contingency generation). See dissertation Appendix, Items 1-8.
   Needs to be added from the author's original source files (not the PDF-extracted appendix,
  which has corrupted indentation).
- `/verification/` — an independent, second cross-check built separately in NumPy:
  - `gen_dataset.py` — generates the IEEE 9-bus dataset via pandapower/Newton-Raphson
    (800 samples, 70-130% loading).
  - `train_and_ablate.py` — trains baseline GCN vs. PI-GNN across 5 seeds, runs the
    physics-loss-weight (lambda) ablation, and runs a paired t-test.
  - `results.json` — raw numeric output backing Sections 6.1, 11.1, and 11.2 of the paper.
  - ## Verification & Reproducibility
* Independent ablation benchmarks and blind replication experiments can be found under [`verification/tests/`](https://github.com/ayushgautam050501-spec/pi-gnn-loadflow/tree/main/verification/tests).

## Reproducing the verification results
```
pip install pandapower numpy scipy
python gen_dataset.py
python train_and_ablate.py
```

## Citation
[Add citation once published.]
