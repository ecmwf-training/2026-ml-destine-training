import matplotlib.pyplot as plt
import numpy as np
import torch
from anemoi.graphs.edges import KNNEdges
from anemoi.graphs.edges.attributes import GaussianDistanceWeights
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class DummyDataset(Dataset):
    """Dummy dataset for downscaling task of random 2D sine wave fields."""

    def __init__(self, num_samples: int, graph: HeteroData):
        self.num_samples = num_samples
        self.graph = graph
        self.proj_matrix = self.build_interp_matrix(graph)

    @property
    def num_variables(self):
        return 1

    def build_interp_matrix(self, graph: HeteroData):
        edge_builder = KNNEdges(num_nearest_neighbours=3, source_name="target", target_name="input")
        graph = edge_builder.update_graph(graph)
        weights = GaussianDistanceWeights(norm="l1")(
            x=(graph["target"], graph["input"]), edge_index=graph["target", "to", "input"].edge_index
        )

        interp_matrix = torch.sparse_coo_tensor(
            graph["target", "to", "input"].edge_index,
            weights.squeeze(),
            (graph["target"].num_nodes, graph["input"].num_nodes),
            device=graph["target", "to", "input"].edge_index.device,
        )
        return interp_matrix.coalesce().T

    def _create_random_2d_sine_wave_field(self):
        sine_wave = np.sin(10 * np.random.rand() * self.graph["target"].x[:, 0]) * np.cos(
            10 * np.random.rand() * self.graph["target"].x[:, 1]
        )
        return sine_wave.to(torch.float32).unsqueeze(-1)

    def _interpolate_to_coarse(self, fine_field):
        return torch.sparse.mm(self.proj_matrix.to(fine_field.device), fine_field)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_fine = self._create_random_2d_sine_wave_field()
        x_coarse = self._interpolate_to_coarse(x_fine)
        return x_coarse, x_fine


class DownscalingModel(nn.Module):
    """A model wrapper around GNNs"""

    def __init__(self, gnn, graph):
        super().__init__()
        self.gnn = gnn
        self.graph = graph

    @property
    def src_coords(self):
        return self.graph["input"].x

    @property
    def dst_coords(self):
        return self.graph["target"].x

    def forward(self, x_src):
        # We suppose batch size = 1 for simplicity
        # x_src: [num_coarse_nodes, in_channels_src]
        # x_dst: [num_fine_nodes, in_channels_dst]
        assert x_src.shape[0] == 1, "Batch size greater than 1 not supported in this example."
        out = self.gnn(
            x_src=x_src[0, ...].to(torch.float32),
            x_dst=self.graph["target"].x.to(torch.float32),
            edge_index=self.graph["input", "to", "target"].edge_index.to(torch.int64),
            edge_attr=None,
        )
        return out


def train(gnn: nn.Module, dataset: Dataset, epochs: int, steps_per_epoch: int, lr: float = 1e-3) -> list[float]:
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = DownscalingModel(gnn=gnn, graph=dataset.graph)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for i, (x_coarse, x_fine) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(x_coarse)  # shape: (batch, len_fine, num_vars)
            loss = criterion(y_pred, x_fine)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_coarse.size(0)
            if i + 1 >= steps_per_epoch:
                break
        epoch_loss /= steps_per_epoch
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    return model, train_losses


# Plotting utility functions
def plot_loss_curve(train_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()


def plot_sample(model, sample):
    x_coarse, x_fine = sample
    y_pred = model(x_coarse.unsqueeze(0)).detach().cpu()

    plt.figure(figsize=(12, 12))

    # Input
    plt.subplot(2, 2, 1)
    plt.scatter(model.src_coords[:, 1], model.src_coords[:, 0], c=x_coarse.numpy().squeeze(), cmap="viridis")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Input (Coarse)")
    plt.colorbar(label="Value")

    # Target
    plt.subplot(2, 2, 2)
    plt.scatter(model.dst_coords[:, 1], model.dst_coords[:, 0], c=x_fine.numpy().squeeze(), cmap="viridis")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Target (Fine)")
    plt.colorbar(label="Value")

    # Prediction
    plt.subplot(2, 2, 3)
    plt.scatter(model.dst_coords[:, 1], model.dst_coords[:, 0], c=y_pred.numpy().squeeze(), cmap="viridis")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Prediction")
    plt.colorbar(label="Predicted Value")

    # Error
    plt.subplot(2, 2, 4)
    plt.scatter(model.dst_coords[:, 1], model.dst_coords[:, 0], c=(y_pred - x_fine).numpy().squeeze(), cmap="coolwarm")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Error (Prediction - Target)")
    plt.colorbar(label="Error")

    plt.tight_layout()
    plt.show()
