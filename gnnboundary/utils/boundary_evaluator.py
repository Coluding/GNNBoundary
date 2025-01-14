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

    def boundary_thickness(self, cls_label_1: int, cls_label_2: int, gamma: float = 0.75, num_steps: int = 100):
        """ Computes the boundary thickness for given class labels. """
        graphs_c1 = pyg_data.Batch.from_data_list([x for x in self.dataset if x.y == cls_label_1])
        graphs_c2 = pyg_data.Batch.from_data_list([x for x in self.dataset if x.y == cls_label_2])

        embeddings_c1 = self._extract_sample_embeddings(graphs_c1, level="linear")
        embeddings_c2 = self._extract_sample_embeddings(graphs_c2, level="linear")

        distances = torch.cdist(embeddings_c1, embeddings_c2, p=2)
        min_idx = torch.argmin(distances)
        G_c1_emb = embeddings_c1[min_idx // embeddings_c2.size(0)]
        G_c1_c2_emb = embeddings_c2[min_idx % embeddings_c2.size(0)]


        t_values = torch.linspace(0, 1, num_steps).view(-1, 1)
        h_t = (1 - t_values) * G_c1_emb + t_values * G_c1_c2_emb

        logits = self.discriminator(h_t)
        softmax_outputs = F.softmax(logits, dim=-1)

        delta_confidence = softmax_outputs[:, cls_label_1] - softmax_outputs[:, cls_label_2]
        indicator_values = (gamma > delta_confidence).float()

        integral_value = indicator_values.mean().item()
        thickness = torch.norm(G_c1_emb - G_c1_c2_emb, p=2).item() * integral_value

        return thickness
