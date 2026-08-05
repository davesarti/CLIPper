"""CPAS-MLP: per-attribute conditioning without a transformer.

Same composition as CPAS (docs/method-proposal-cpas.md), same three learned
quantities, but the encoder layer is replaced by an MLP applied to each
attribute independently plus a pooled context vector:

    h1_a = MLP1([v_ref ; w_a ; sign])       per attribute, no interaction
    c_a  = mean_{b != a} h1_b               what else is queried
    h_a  = MLP2([h1_a ; c_a])               -> alpha_a, delta_a

The report's ablation found the cross-attention layer earned nothing measurable
(0.008 against a 0.019 seed spread) while the full-rank delta head is the only
component that earns the gain - and rotates the probe directions by ~60 degrees,
which is more freedom than "correcting" them needs. This module acts on both:
2.1M of trunk becomes 0.48M of MLP, and the 512x512 delta head becomes a
rank-32 factorization (0.26M -> 0.025M).

Two properties make the cross-attribute ablation exact here in a way the
attention mask could not be. Disabling the context sets c_a = 0 without
removing MLP2, so parameter count, initialization and seed are unchanged; and
because the pool excludes the token itself, a single-attribute query has c_a = 0
either way, so only multi-attribute queries can move.

As in CPAS the heads are initialized so the untrained module reproduces
probes.compose_probe(gamma=0.6) exactly.
"""

import math

import torch
from torch import nn

from src.steering import compose

POS, NEG = 0, 1  # sign-embedding rows


class PerAttributeMLP(nn.Module):
    """Reference-conditioned steering of probe directions, MLP variant.

    d: embedding width; hidden: MLP width; sign_dim: width of the T+/T- tag;
    rank: rank of the delta factorization (the bend's degrees of freedom);
    delta_max: bound on the bend (0 disables bending); cross_attributes: when
    False the pooled context is zeroed, leaving conditioning on the reference
    only; gamma_init/alpha_init: the composition the untrained module
    reproduces.
    """

    def __init__(
        self,
        d: int = 512,
        hidden: int = 256,
        sign_dim: int = 64,
        rank: int = 32,
        delta_max: float = 0.3,
        gamma_init: float = 0.6,
        alpha_init: float = 1.0,
        cross_attributes: bool = True,
    ) -> None:
        super().__init__()
        self.delta_max = delta_max
        self.cross_attributes = cross_attributes
        self.sign_emb = nn.Embedding(2, sign_dim)
        nn.init.normal_(self.sign_emb.weight, std=0.02)
        self.trunk = nn.Sequential(
            nn.Linear(2 * d + sign_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.context = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.head_alpha = nn.Linear(hidden, 1)
        self.head_gamma = nn.Linear(hidden + d, 1)
        # delta = delta_max * tanh(U(V(h))): the bend lives in a rank-r subspace.
        self.delta_down = nn.Linear(hidden, rank)
        self.delta_up = nn.Linear(rank, d, bias=False)
        self._init_as_baseline(gamma_init, alpha_init)

    def _init_as_baseline(self, gamma_init: float, alpha_init: float) -> None:
        """Set the heads so the untrained module is the fixed-rule composition.

        delta_up is zeroed rather than delta_down, so delta = 0 at step 0 while
        delta_up still receives gradient and pulls delta_down into play on the
        first step - zeroing both would leave the bend permanently dead.
        """
        for head in (self.head_gamma, self.head_alpha):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.delta_up.weight)
        # sigmoid(b) = gamma_init; softplus(b) = alpha_init
        self.head_gamma.bias.data.fill_(math.log(gamma_init / (1 - gamma_init)))
        self.head_alpha.bias.data.fill_(math.log(math.expm1(alpha_init)))

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
        k = dirs.shape[1]
        tags = self.sign_emb(torch.where(signs > 0, POS, NEG))
        tokens = torch.cat([v_ref.unsqueeze(1).expand(-1, k, -1), dirs, tags], dim=-1)
        h = self.trunk(tokens)  # (B, K, H)

        # Zero the padded rows before pooling: a padded slot must not leak into
        # any real attribute's context.
        keep = mask.unsqueeze(-1).to(h.dtype)
        h = h * keep
        total = h.sum(dim=1, keepdim=True)
        count = keep.sum(dim=1, keepdim=True)
        # Exclude self, so K=1 queries see an empty context.
        others = (total - h) / (count - keep).clamp(min=1.0)
        if not self.cross_attributes:
            others = torch.zeros_like(others)
        z = self.context(torch.cat([h, others], dim=-1))

        pooled = (total / count.clamp(min=1.0)).squeeze(1)  # (B, H)
        gamma = torch.sigmoid(self.head_gamma(torch.cat([pooled, v_ref], dim=-1)))
        alpha = nn.functional.softplus(self.head_alpha(z)).squeeze(-1)
        delta = self.delta_max * torch.tanh(self.delta_up(self.delta_down(z)))
        return gamma.squeeze(-1), alpha * keep.squeeze(-1), delta * keep

    def forward(
        self,
        v_ref: torch.Tensor,
        dirs: torch.Tensor,
        signs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Composite query embeddings, (B, D) L2-normalized."""
        gamma, alpha, delta = self.steer(v_ref, dirs, signs, mask)
        return compose(v_ref, dirs, signs, gamma, alpha, delta)
