import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.data as pyg_data
from typing import Literal


class BoundaryEvaluator:
    def __init__(self, sampler, discriminator, dataset):
        self.sampler = sampler
        self.discriminator = discriminator
        self.dataset = dataset

        self.embeddings: torch.Tensor = None

    def _extract_boundary_embeddings(self, graph_num=2, level: Literal['linear', 'conv'] = "linear"):
        extraction = []

        def hook_fn(module, input, output):
            extraction.append(output.detach())

        if level == "linear":
            self.discriminator.lin.register_forward_hook(hook_fn)
        else:
            self.discriminator.conv.register_forward_hook(hook_fn)

        graphs = self.sampler(k=graph_num, mode="discrete")
        output = self.discriminator(graphs, edge_weight=graphs.edge_weight)
        self.embeddings = extraction[0]

    def _extract_sample_embeddings(self, batch: pyg_data.Batch, level: Literal['linear', 'conv'] = "linear"):
        extraction = []

        def hook_fn(module, input, output):
            extraction.append(output.detach())


        if level == "linear":
            self.discriminator.lin.register_forward_hook(hook_fn)
        else:
            self.discriminator.conv.register_forward_hook(hook_fn)

        output = self.discriminator(batch)
        return extraction[0]

    def boundary_margin(self, cls_label: int):
        self._extract_boundary_embeddings(graph_num=1, level="linear")
        boundary_embedding = self.embeddings
        graphs_cls = pyg_data.Batch.from_data_list([x for x in self.dataset if x.y == cls_label])
        graphs_cls_embeddings = self._extract_sample_embeddings(graphs_cls, level="linear")

        distances = torch.cdist(boundary_embedding, graphs_cls_embeddings, p=2)
        min_dist = torch.min(distances, dim=1).values

        return min_dist

    def boundary_thickness(self, cls_label: int, gamma: float = 0.75,
                           num_interpolation_points: int = 100):
        """ Computes the boundary thickness for given class labels. """
        self._extract_boundary_embeddings(graph_num=1, level="linear")
        embeddings_boundary = self.embeddings
        graphs_c1 = pyg_data.Batch.from_data_list([x for x in self.dataset if x.y == cls_label])
        embeddings_c1 = self._extract_sample_embeddings(graphs_c1, level="linear")


        distances = torch.norm(embeddings_c1 - embeddings_boundary, dim=1)

        thicknesses = []
        for i in range(len(embeddings_c1)):
            # Generate interpolation points
            t = torch.linspace(0, 1, num_interpolation_points)
            h_t = (1 - t[:, None]) * embeddings_c1[i] + t[:, None] * embeddings_boundary[i]

            # Get logits for interpolated points
            logits = self.discriminator.out(h_t)
            probs = torch.softmax(logits, dim=1)

            # Calculate indicator function: 1_{γ > σ(ηₗ(h(t)))_{c₁} - σ(ηₗ(h(t)))_{c₂}}
            prob_diff = probs[:, 0] - probs[:, 1]  # Assuming binary classification
            indicator = (gamma > prob_diff).float()

            # Calculate denominator (numerical integration)
            denominator = torch.mean(indicator)

            if denominator > 0:
                thicknesses.append(distances[i] / denominator)

        # Calculate expectation
        thickness = torch.mean(torch.tensor(thicknesses)).item()

        return thickness


import torch
import numpy as np
from typing import List, Tuple, Union
from scipy.stats import entropy


def calculate_boundary_margin(embeddings_c1: torch.Tensor, embeddings_boundary: torch.Tensor) -> float:
    """Calculate the boundary margin between two classes.

    Implements the formula:
    Φ(f, c₁, c₂) = min_{G_{c₁}, G_{c₁∥c₂}} ‖φₗ(G_{c₁}) - φₗ(G_{c₁∥c₂})‖

    The boundary margin is defined as the smallest distance to the decision boundary among all training
    points in the same class. A classifier with a large margin can have better generalization properties
    and robustness to input perturbation.

    Args:
        embeddings_c1: Embeddings of samples from class c₁ (N x D tensor)
        embeddings_boundary: Embeddings of boundary samples (M x D tensor)

    Note:
        The embeddings should be from the graph pooling layer, in line with standard practices
        for interpreting deep neural networks.

    Returns:
        float: The boundary margin value
    """
    # Calculate pairwise distances between class samples and boundary samples
    dists = torch.cdist(embeddings_c1, embeddings_boundary)

    # Find the minimum distance
    margin = torch.min(dists).item()

    return margin


def calculate_boundary_thickness(embeddings_c1: torch.Tensor,
                                 embeddings_boundary: torch.Tensor,
                                 logits_func: callable,
                                 gamma: float = 0.75,  # Default value as per paper
                                 num_interpolation_points: int = 100) -> float:
    """Calculate the boundary thickness between two classes.

    Implements the formula:
    Θ(f,γ,c₁,c₂) = E[(‖φₗ(G_{c₁}) - φₗ(G_{c₁∥c₂})‖) / ∫₀¹ 1_{γ > σ(ηₗ(h(t)))_{c₁} - σ(ηₗ(h(t)))_{c₂}} dt]

    Thick decision boundaries enhance model robustness, whereas thin decision boundaries would
    result in overfitting and reduced robustness. The boundary thickness measures the expected
    distance to travel along line segments between different classes across a decision boundary.

    Args:
        embeddings_c1: Embeddings of samples from class c₁ (N x D tensor)
        embeddings_boundary: Embeddings of boundary samples (N x D tensor)
        logits_func: Function that takes embeddings and returns logits
        gamma: Confidence threshold for boundary uncertainty (default: 0.75 as per paper)
        num_interpolation_points: Number of points for numerical integration

    Returns:
        float: The boundary thickness value
    """
    # Calculate numerator: ‖φₗ(G_{c₁}) - φₗ(G_{c₁∥c₂})‖
    distances = torch.norm(embeddings_c1 - embeddings_boundary, dim=1)

    thicknesses = []
    for i in range(len(embeddings_c1)):
        # Generate interpolation points
        t = torch.linspace(0, 1, num_interpolation_points)
        h_t = (1 - t[:, None]) * embeddings_c1[i] + t[:, None] * embeddings_boundary[i]

        # Get logits for interpolated points
        logits = logits_func(h_t)
        probs = torch.softmax(logits, dim=1)

        # Calculate indicator function: 1_{γ > σ(ηₗ(h(t)))_{c₁} - σ(ηₗ(h(t)))_{c₂}}
        prob_diff = probs[:, 0] - probs[:, 1]  # Assuming binary classification
        indicator = (gamma > prob_diff).float()

        # Calculate denominator (numerical integration)
        denominator = torch.mean(indicator)

        if denominator > 0:
            thicknesses.append(distances[i] / denominator)

    # Calculate expectation
    thickness = torch.mean(torch.tensor(thicknesses)).item()

    return thickness


def calculate_boundary_complexity(boundary_embeddings: torch.Tensor) -> float:
    """Calculate the boundary complexity.

    Implements the formula:
    Γ(f,c₁,c₂) = H(λ/‖λ‖₁)/log(D)
    where H(x) = -Σᵢ xᵢlog(xᵢ) is the entropy

    The complexity measures the generalization ability of the decision boundaries. If the boundary
    cases span over fewer dimensions, then the classifier is less likely to overfit. A complexity
    measure of 0 indicates simple decision boundaries, while 1 indicates complex boundaries.

    Args:
        boundary_embeddings: Matrix of boundary embeddings X_{c₁∥c₂} (N x D tensor)

    Note:
        This measurement is only applicable to linearly separable boundaries.
        The embeddings should be from the last hidden layer (ϕL−1) of the network.

    Returns:
        float: The boundary complexity value (ranges from 0 to 1)
    """
    # Calculate covariance matrix
    X = boundary_embeddings - torch.mean(boundary_embeddings, dim=0)
    cov = torch.mm(X.T, X) / (X.shape[0] - 1)

    # Calculate eigenvalues
    eigenvalues = torch.linalg.eigvals(cov).real

    # Normalize eigenvalues
    normalized_eigenvalues = eigenvalues / torch.sum(eigenvalues)

    # Calculate entropy
    H = entropy(normalized_eigenvalues.cpu().numpy())

    # Normalize by log(D)
    D = boundary_embeddings.shape[1]  # embedding dimension
    complexity = H / np.log(D)

    return complexity