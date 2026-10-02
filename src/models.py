"""
SimCSE model architecture and contrastive loss implementations (U2T02).
Implements:
  1. Base encoder (bert-base-uncased) with configurable pooling:
     - 'cls': [CLS] token representation + optional MLP projection head
     - 'mean': Mean pooling over active token representations
  2. Unsupervised SimCSE contrastive objective (Eq. 1 in Gao et al., 2021)
     - Normal mode: independent dropout masks between view 1 and view 2
     - Ablation mode: identical dropout mask (same view)
  3. Supervised SimCSE contrastive objective (Eq. 5 in Gao et al., 2021)
     - Normal mode: in-batch negatives + hard negatives (contradiction)
     - Ablation mode: hard negatives turned OFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class MLPPooler(nn.Module):
    """
    MLP layer used during SimCSE training (Linear + Tanh).
    As demonstrated in SimCSE Section 3.2, training with an MLP layer
    and discarding it during evaluation/inference yields optimal sentence embeddings.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, first_token_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(first_token_tensor))


class SimCSE(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        pooling: str = "cls",
        temperature: float = 0.05,
        dropout_rate: float = 0.1,
        use_mlp: bool = True
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        # Configure hidden dropout rate
        self.config.attention_probs_dropout_prob = dropout_rate
        self.config.hidden_dropout_prob = dropout_rate

        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.pooling = pooling.lower()
        self.temperature = temperature
        self.use_mlp = use_mlp

        if self.use_mlp:
            self.mlp = MLPPooler(self.config.hidden_size)
        else:
            self.mlp = nn.Identity()

    def get_pooled_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        return_for_eval: bool = False
    ) -> torch.Tensor:
        """
        Extracts pooled sentence embedding.
        If return_for_eval=True, bypasses the MLP pooler (as recommended in SimCSE).
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        last_hidden_state = outputs.last_hidden_state

        if self.pooling == "cls":
            cls_rep = last_hidden_state[:, 0]
            if return_for_eval or not self.use_mlp:
                return cls_rep
            return self.mlp(cls_rep)

        elif self.pooling == "mean":
            # Mean pooling over non-padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_rep = sum_embeddings / sum_mask
            if return_for_eval or not self.use_mlp:
                return mean_rep
            return self.mlp(mean_rep)

        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}. Use 'cls' or 'mean'.")

    def forward_unsupervised(
        self,
        batch_inputs: dict,
        same_dropout_ablation: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unsupervised SimCSE forward pass (Eq. 1).
        batch_inputs: tokenized inputs for a batch of sentences.
        If same_dropout_ablation=True: passes through encoder once and duplicates z1,
        effectively testing zero contrastive variance (identical dropout mask).
        """
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]
        token_type_ids = batch_inputs.get("token_type_ids", None)

        z1 = self.get_pooled_embedding(input_ids, attention_mask, token_type_ids, return_for_eval=False)

        if same_dropout_ablation:
            # Ablation: exact same representation / identical dropout mask
            z2 = z1
        else:
            # Standard: second forward pass with independent dropout mask
            z2 = self.get_pooled_embedding(input_ids, attention_mask, token_type_ids, return_for_eval=False)

        # Normalize embeddings
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)

        # Cosine similarity matrix between view 1 and view 2: (batch_size, batch_size)
        cos_sim = torch.mm(z1_norm, z2_norm.transpose(0, 1)) / self.temperature

        # Labels: diagonal entries are positive pairs
        batch_size = z1_norm.size(0)
        labels = torch.arange(batch_size, device=z1_norm.device)

        loss = F.cross_entropy(cos_sim, labels)
        return loss, cos_sim

    def forward_supervised(
        self,
        premise_inputs: dict,
        entailment_inputs: dict,
        contradiction_inputs: dict = None,
        use_hard_negatives: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Supervised SimCSE forward pass (Eq. 5).
        premise_inputs: anchors (h_i)
        entailment_inputs: positives (h_i^+)
        contradiction_inputs: hard negatives (h_i^-), optional
        """
        h_premise = self.get_pooled_embedding(
            premise_inputs["input_ids"],
            premise_inputs["attention_mask"],
            premise_inputs.get("token_type_ids", None),
            return_for_eval=False
        )
        h_entail = self.get_pooled_embedding(
            entailment_inputs["input_ids"],
            entailment_inputs["attention_mask"],
            entailment_inputs.get("token_type_ids", None),
            return_for_eval=False
        )

        h_premise_norm = F.normalize(h_premise, p=2, dim=1)
        h_entail_norm = F.normalize(h_entail, p=2, dim=1)

        sim_pos = torch.mm(h_premise_norm, h_entail_norm.transpose(0, 1)) / self.temperature
        batch_size = h_premise_norm.size(0)

        if use_hard_negatives and contradiction_inputs is not None:
            h_contra = self.get_pooled_embedding(
                contradiction_inputs["input_ids"],
                contradiction_inputs["attention_mask"],
                contradiction_inputs.get("token_type_ids", None),
                return_for_eval=False
            )
            h_contra_norm = F.normalize(h_contra, p=2, dim=1)
            sim_hard_neg = torch.mm(h_premise_norm, h_contra_norm.transpose(0, 1)) / self.temperature

            # Concatenate positives matrix (N x N) and hard negatives matrix (N x N) -> (N x 2N)
            sim_matrix = torch.cat([sim_pos, sim_hard_neg], dim=1)
        else:
            # Hard negatives OFF ablation: only N in-batch candidates
            sim_matrix = sim_pos

        labels = torch.arange(batch_size, device=h_premise_norm.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss, sim_matrix
