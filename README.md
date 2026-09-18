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
Note for the Reader :- Dear reader kindly run these codes as shown in this note.
First install the pandapower in your Terminal with code - !pip install panapower.
Second you have to write the code from the file Powerflow9.py
Third you have to apply the sequence from the given code contingency_analysis.pdf
Fourth is to apply code from the file physics_bounded_dataset.py
Fifth is using the code from the file gen_dataset.py
Sixth is by applying the codes given in the file Table10_1_rebuilt.py
Seventh is to apply the code given in the file table13_2_rebuilt.py
Eighth is then to use the code given in the file Train_and_ablate.py
Ninth is to use the gen_dataset.py
Tenth is to use the codes generated in the file real_load_profile.py
Eleventh is to use real_data_rerun.py
Twelvth is applyiing codes given in the file grid_graph_gnn.dataset
If the reader uses the apply the codes in the given sequence,it will be much easier to get the results without getting any unknown errors due to missing packages and the files.The files are sequenced in a way that it will become easy to locate and use the given packages in a manner that can lead to the results without losing the packages installed in the codes.There's chance that packages might miss if we just apply codes without a certain sequence.
