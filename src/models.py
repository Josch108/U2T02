"""
SimCSE model and contrastive objectives for U2T02.

The implementation follows the assignment's two training modes:
- Unsupervised SimCSE: two stochastic views of the same sentence are created
  by independent dropout masks.
- Supervised SimCSE: premise-entailment pairs are positives and only real
  contradiction examples are used as hard negatives.

For reproducible export, both modes use an MLP during training and discard it
for evaluation. This matches the train-only MLP variant reported by SimCSE and
avoids exporting a custom projection head.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class MLPPooler(nn.Module):
    """Linear + tanh projection used only during contrastive training."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(x))


class SimCSE(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        pooling: str = "cls",
        temperature: float = 0.05,
        dropout_rate: float = 0.1,
        use_mlp: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.attention_probs_dropout_prob = dropout_rate
        self.config.hidden_dropout_prob = dropout_rate

        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.pooling = pooling.lower()
        self.temperature = temperature
        self.use_mlp = use_mlp
        self.mlp = MLPPooler(self.config.hidden_size) if use_mlp else nn.Identity()

    def get_pooled_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        apply_mlp: bool = False,
    ) -> torch.Tensor:
        """Return one embedding per sentence.

        During training, apply_mlp=True is used. During STS-B evaluation and
        sentence-transformers export, apply_mlp=False is used.
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        hidden = self.encoder(**kwargs).last_hidden_state

        if self.pooling == "cls":
            rep = hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            rep = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        if apply_mlp and self.use_mlp:
            rep = self.mlp(rep)
        return rep

    def forward_unsupervised(
        self,
        batch_inputs: dict,
        same_dropout_ablation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unsupervised SimCSE objective (Eq. 1)."""
        ids = batch_inputs["input_ids"]
        mask = batch_inputs["attention_mask"]
        token_types = batch_inputs.get("token_type_ids")

        z1 = self.get_pooled_embedding(ids, mask, token_types, apply_mlp=True)

        if same_dropout_ablation:
            # Exact same stochastic view: the required ablation.
            z2 = z1
        else:
            # Independent second forward pass -> independent dropout mask.
            z2 = self.get_pooled_embedding(ids, mask, token_types, apply_mlp=True)

        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        logits = torch.mm(z1, z2.transpose(0, 1)) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    def forward_supervised(
        self,
        premise_inputs: dict,
        entailment_inputs: dict,
        contradiction_inputs: Optional[dict] = None,
        use_hard_negatives: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supervised SimCSE objective (Eq. 5).

        The supplied subset has hard negatives for only about 28% of positive
        pairs. Therefore contradiction_inputs contains only REAL contradiction
        sentences present in the current batch. Missing contradictions are not
        replaced by positives.
        """
        hp = self.get_pooled_embedding(
            premise_inputs["input_ids"],
            premise_inputs["attention_mask"],
            premise_inputs.get("token_type_ids"),
            apply_mlp=True,
        )
        he = self.get_pooled_embedding(
            entailment_inputs["input_ids"],
            entailment_inputs["attention_mask"],
            entailment_inputs.get("token_type_ids"),
            apply_mlp=True,
        )

        hp = F.normalize(hp, p=2, dim=1)
        he = F.normalize(he, p=2, dim=1)
        positive_logits = torch.mm(hp, he.transpose(0, 1)) / self.temperature

        logits = positive_logits
        if use_hard_negatives and contradiction_inputs is not None:
            hc = self.get_pooled_embedding(
                contradiction_inputs["input_ids"],
                contradiction_inputs["attention_mask"],
                contradiction_inputs.get("token_type_ids"),
                apply_mlp=True,
            )
            hc = F.normalize(hc, p=2, dim=1)
            hard_logits = torch.mm(hp, hc.transpose(0, 1)) / self.temperature
            logits = torch.cat([positive_logits, hard_logits], dim=1)

        labels = torch.arange(hp.size(0), device=hp.device)
        loss = F.cross_entropy(logits, labels)
        return loss, logits
