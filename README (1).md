# Original Implementation (reconstructed)

These files reconstruct the code that appeared as text in the dissertation's
Appendix (Items 1-9), which had corrupted indentation from PDF text extraction
and could not run as-is. Logic, values, and comments are preserved from the
original; only whitespace/indentation was fixed, and files were split by
responsibility.

| File | Dissertation Appendix item | Tested? |
|---|---|---|
| `powerflow9.py` | Item 6 | Ran successfully - converged base case |
| `grid_graph_and_gnn.py` | Items 1-3 | Syntax-checked (needs torch + torch_geometric to run) |
| `contingency_analysis.py` | Items 4, 7 | Syntax-checked (needs scikit-learn to run) |
| `physics_bounded_dataset.py` | Item 5 | Syntax-checked (needs pandapower to run) |
| `independent_verification.py` | Item 8 | Ran successfully - `make_dataset()` verified |
| `kcl_check.py` | Item 9 | Syntax-checked (needs torch to run) |

`powerflow9.py` is a shared dependency imported by several of the other files.

**Note:** `grid_graph_and_gnn.py` implements a *classification* GCN (critical
vs. non-critical bus voltage), which is a different model from the
*regression* PI-GNN (predicts Vm, Va directly) described in the paper's main
text and reproduced in `../verification/`. Both appear in the dissertation
and are kept as separate, clearly-labeled files here.

Install requirements: `pip install pandapower torch torch_geometric scikit-learn pandas numpy`
