"""CPAS: Conditioned Per-Attribute Steering.

Keeps the composition formula of probe-direction retrieval and learns its three
fixed choices as functions of the query (see docs/method-proposal-cpas.md):

    q = normalize( gamma(x) * v_ref
                   + sum_a  sign_a * alpha_a(x) * normalize(w_a + delta_a(x)) )

One self-attention layer over [reference; attribute directions] lets each step
size and each direction bend depend on the reference *and* on the other
attributes queried together with it - the probe directions are not orthogonal,
so the correct joint step is not the sum of the individual ones.

At initialization the heads emit gamma=0.6, alpha=1, delta=0, i.e. the module
reproduces probes.compose_probe exactly; training departs from that baseline
only if the mined triplets support it.
"""

import math

import torch
from torch import nn

REF, POS, NEG = 0, 1, 2  # role-embedding rows


class CPAS(nn.Module):
    """Reference-conditioned steering of probe directions.

    d: embedding width; heads/ffn: encoder-layer size; delta_max: bound on the
    per-attribute direction bend (0 disables bending, leaving only per-attribute
    rescaling); cross_attributes: when False, an attention mask stops attribute
    tokens from seeing each other, so steps still adapt to the reference but not
    to the attributes queried alongside them - the ablation that isolates the
    non-orthogonality claim; gamma_init/alpha_init: the composition the
    untrained module reproduces.
    """

    def __init__(
        self,
        d: int = 512,
        heads: int = 4,
        ffn: int = 1024,
        delta_max: float = 0.3,
        gamma_init: float = 0.6,
        alpha_init: float = 1.0,
        dropout: float = 0.0,
        cross_attributes: bool = True,
    ) -> None:
        super().__init__()
        self.delta_max = delta_max
        self.cross_attributes = cross_attributes
        self.roles = nn.Embedding(3, d)
        nn.init.normal_(self.roles.weight, std=0.02)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.head_gamma = nn.Linear(d, 1)
        self.head_alpha = nn.Linear(d, 1)
        self.head_delta = nn.Linear(d, d)
        self._init_as_baseline(gamma_init, alpha_init)

    def _init_as_baseline(self, gamma_init: float, alpha_init: float) -> None:
        """Zero the head weights and set biases to the fixed-rule composition."""
        for head in (self.head_gamma, self.head_alpha, self.head_delta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # sigmoid(b) = gamma_init; softplus(b) = alpha_init
        self.head_gamma.bias.data.fill_(math.log(gamma_init / (1 - gamma_init)))
        self.head_alpha.bias.data.fill_(math.log(math.expm1(alpha_init)))

    def _attention_mask(self, length: int, device: torch.device) -> torch.Tensor | None:
        """Block attribute-to-attribute attention when cross_attributes is off.

        Attribute tokens keep access to themselves and to the reference (which
        is never padded, so no row attends to nothing); the reference token
        still sees everything, since gamma summarizes the whole query.
        """
        if self.cross_attributes:
            return None
        blocked = torch.ones(length, length, dtype=torch.bool, device=device)
        blocked.fill_diagonal_(False)
        blocked[0] = False  # reference attends to all attributes
        blocked[:, 0] = False  # every attribute attends to the reference
        return blocked

    def steer(
        self,
        v_ref: torch.Tensor,
        dirs: torch.Tensor,
        signs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict (gamma, alpha, delta) for a batch of queries.

        v_ref: (B, D) L2-normalized references.
        dirs:  (B, K, D) L2-normalized probe directions, padded with anything.
        signs: (B, K) +1 for T+ attributes, -1 for T-.
        mask:  (B, K) bool, True where the slot holds a real attribute.
        Returns gamma (B,), alpha (B, K), delta (B, K, D); alpha and delta are
        zeroed on padded slots.
        """
        roles = torch.where(signs > 0, POS, NEG)
        tokens = torch.cat(
            [
                (v_ref + self.roles.weight[REF]).unsqueeze(1),
                dirs + self.roles(roles),
            ],
            dim=1,
        )  # (B, 1+K, D)
        # The reference slot is always valid, so no row is fully masked.
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], dim=1)
        z = self.encoder(
            tokens,
            src_mask=self._attention_mask(tokens.shape[1], tokens.device),
            src_key_padding_mask=pad,
        )

        gamma = torch.sigmoid(self.head_gamma(z[:, 0])).squeeze(-1)
        alpha = nn.functional.softplus(self.head_alpha(z[:, 1:])).squeeze(-1)
        delta = self.delta_max * torch.tanh(self.head_delta(z[:, 1:]))
        keep = mask.to(alpha.dtype)
        return gamma, alpha * keep, delta * keep.unsqueeze(-1)

    def forward(
        self,
        v_ref: torch.Tensor,
        dirs: torch.Tensor,
        signs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Composite query embeddings, (B, D) L2-normalized."""
        gamma, alpha, delta = self.steer(v_ref, dirs, signs, mask)
        bent = nn.functional.normalize(dirs + delta, dim=-1)
        steps = (signs * alpha).unsqueeze(-1) * bent  # (B, K, D)
        q = gamma.unsqueeze(-1) * v_ref + steps.sum(dim=1)
        return nn.functional.normalize(q, dim=-1)


def pad_queries(
    queries: list[tuple[list[int], list[int]]],
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack per-query attribute rows into padded (dirs, signs, mask) tensors.

    queries: list of (positive rows, negative rows) indexing `directions`;
    directions: (A, D) L2-normalized probe directions.
    """
    d = directions.shape[1]
    k = max((len(p) + len(n) for p, n in queries), default=0)
    dirs = torch.zeros(len(queries), k, d)
    signs = torch.zeros(len(queries), k)
    mask = torch.zeros(len(queries), k, dtype=torch.bool)
    for i, (pos, neg) in enumerate(queries):
        rows = pos + neg
        if not rows:
            continue
        dirs[i, : len(rows)] = directions[rows]
        signs[i, : len(pos)] = 1.0
        signs[i, len(pos) : len(rows)] = -1.0
        mask[i, : len(rows)] = True
    return dirs, signs, mask
