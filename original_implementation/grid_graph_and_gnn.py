"""
Grid Topology/Feature Extraction, GNN Architecture, and Training Loop.
Reconstructed from dissertation Appendix, Items 1-3.

NOTE: this is the ORIGINAL classification-style model (predicts whether a bus
voltage is critical / out-of-bounds), separate from the regression PI-GNN
(predicts Vm, Va directly) described in the main text and in
independent_verification.py. Both appear in the dissertation appendix and
should be kept distinct when documenting the codebase.

Requires: pandapower, torch, torch_geometric
"""
import pandapower as pp
import pandapower.networks as nw
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader


def initialize_grid_graph(case_type='case9'):
    # 1. Load test system (e.g., IEEE 9-bus) and generate ground truth via Newton-Raphson
    net = nw.case9()
    pp.runpp(net)

    # 2. Node feature extraction: voltage magnitude and phase angle
    x = torch.tensor(net.res_bus[['vm_pu', 'va_degree']].values, dtype=torch.float)

    # 3. Define target (critical if voltage is outside [0.95, 1.05] p.u.)
    v_mag = net.res_bus['vm_pu'].values
    y = torch.tensor(((v_mag < 0.95) | (v_mag > 1.05)).astype(int), dtype=torch.long)

    # 4. Create edge index (undirected graph)
    bus_to_idx = {bus_id: i for i, bus_id in enumerate(net.bus.index.values)}
    from_nodes = [bus_to_idx[b] for b in net.line['from_bus'].values]
    to_nodes = [bus_to_idx[b] for b in net.line['to_bus'].values]
    edge_index = torch.tensor([from_nodes + to_nodes, to_nodes + from_nodes], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


class GCN(torch.nn.Module):
    """Two-layer GCN classifier (torch_geometric)."""

    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)


def train_one_epoch(model, train_loader, optimizer, criterion):
    """
    NOTE (documented for transparency): the dissertation records that a naive
    version of this training loop achieved ~100% accuracy but F1=0.00 for the
    critical class -- the "Accuracy Paradox" -- because the majority class
    (non-critical) dominates the base-case/lightly-loaded dataset. Class
    imbalance handling (e.g. class_weight='balanced', or the N-1/N-2 augmented
    dataset in contingency_analysis.py) is required to get a meaningful F1.
    """
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


if __name__ == "__main__":
    graph = initialize_grid_graph()
    train_dataset = [graph]  # single base-case graph; see contingency_analysis.py for a
                              # properly sized, class-balanced N-1/N-2 dataset
    model = GCN(num_node_features=2, num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  loss={loss:.4f}")
