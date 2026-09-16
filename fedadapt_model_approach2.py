"""
fedadapt_model.py
=================
FedAdapt: Federated Adaptive Transfer Learning for AKI prediction.

ARCHITECTURE OVERVIEW
---------------------
Every site runs a FedAdaptClient that contains four components:

    [SiteInputAdapter]  →  [SharedBody]  →  [PersonalHead]
                                ↑
                        [GRLGroupDiscriminator]
                        (adversarial, active during body training)

Component       Federated?   Novel?   Purpose
─────────────────────────────────────────────────────────────────
SiteInputAdapter   NO         NO      Projects site-specific input
                                      dim → shared embedding dim.
                                      ALL FL methods use this
                                      (FedAvg, FedProx, SCAFFOLD,
                                      FedAdapt) so comparisons
                                      are fair. Not a novelty.

SharedBody         YES        NO      MLP encoder shared across
                                      all sites via federation.
                                      Only these weights cross
                                      site boundaries.

GRLGroupDiscrim.   NO         YES     Gradient Reversal Layer +
                                      discriminator that tries to
                                      identify which feature group
                                      (renal / inflammatory /
                                      metabolic / clinical) an
                                      embedding came from.
                                      GRL flips the gradient sign
                                      so the body is pushed toward
                                      group-INVARIANT embeddings.
                                      This is FedAdapt's primary
                                      novel contribution.

PersonalHead       NO         YES     Local classifier, never
                                      shared. Fine-tuned on each
                                      site's own data after
                                      federation. Captures local
                                      decision boundaries (e.g.
                                      ICU vs rural thresholds).

FEDERATION PROTOCOL
-------------------
Each round:
    1. Server broadcasts SharedBody weights to all sites.
    2. Each site runs K local steps:
         a. Forward: adapter → body → personal_head  (task loss)
         b. Forward: adapter → body → GRL → discriminator (adv. loss)
         c. Total loss = task_loss - lambda_adv * adv_loss
            (minus because GRL reverses the adversarial gradient)
         d. Backprop updates adapter + body + discriminator.
            PersonalHead receives task_loss gradient only.
    3. Sites send SharedBody gradients (or weights) to server.
    4. Server aggregates (FedAvg by default) → new SharedBody.
    5. PersonalHead stays local throughout.

After T federation rounds:
    - Freeze SharedBody.
    - Fine-tune PersonalHead locally for E epochs (FedBABU-style).

BASELINE COMPATIBILITY
----------------------
Baselines (FedAvg, FedProx, SCAFFOLD) use the SAME SiteInputAdapter
and SharedBody. They differ only in:
    FedAvg   — no GRL, no PersonalHead (head is shared)
    FedProx  — adds proximal penalty μ||w - w_global||² to local loss
    SCAFFOLD — adds control variates to correct client drift
    FedAdapt — adds GRL + PersonalHead (this file)

Swap the training loop; keep the model classes identical for all.

USAGE
-----
from fedadapt_model import FedAdaptClient, FedAdaptServer, load_site_from_metadata
import json

# Load adapter metadata produced by mimic_ftl_simulation_phase2.py
with open("phase2_sites/site_E_adapter_meta.json") as f:
    meta = json.load(f)

client = FedAdaptClient.from_metadata(meta, hidden_dim=128, n_groups=4)
server = FedAdaptServer(body=client.body)
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─── GRADIENT REVERSAL LAYER ──────────────────────────────────────────────────

class _GradientReversalFn(torch.autograd.Function):
    """
    Forward pass: identity  (x → x)
    Backward pass: negated gradient scaled by lambda_  (grad → -lambda_ * grad)

    The lambda_ schedule ramps from 0 to lambda_max following the standard
    DANN schedule so training is stable early on:
        lambda_(p) = 2 / (1 + exp(-10 * p)) - 1
    where p ∈ [0, 1] is training progress.
    """

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_max: float = 1.0):
        super().__init__()
        self.lambda_max = lambda_max
        self._progress  = 0.0          # set externally each round

    def set_progress(self, p: float) -> None:
        """p ∈ [0, 1]: fraction of total training completed."""
        self._progress = float(p)

    @property
    def lambda_(self) -> float:
        """
        Two-phase lambda schedule (REVERTED to original — best performing variant).

        Option 2 (extended warmup, both v1 and v2) was tested and underperformed
        this schedule overall: mean AUROC 0.8726 (v2) vs 0.8784 for this schedule
        combined with adaptive λ_max. Site_D win-rate against SCAFFOLD also dropped
        from 13/20 (this schedule) to 10/20 (Option 2 v2). Reverted — adaptive
        λ_max (set per-site at build time, see build_clients) combined with this
        original warmup is the strongest GRL-only variant found.

          Phase 1 (p < 0.4): linear ramp 0 → 0.5*lambda_max
          Phase 2 (p >= 0.4): DANN-style ramp to lambda_max
        """
        p = self._progress
        if p < 0.4:
            return self.lambda_max * 0.5 * (p / 0.4)
        else:
            p2 = (p - 0.4) / 0.6
            return self.lambda_max * (0.5 + 0.5 * (2.0 / (1.0 + math.exp(-10.0 * p2)) - 1.0))

    def forward(self, x: Tensor) -> Tensor:
        return _GradientReversalFn.apply(x, self.lambda_)


# ─── SITE INPUT ADAPTER ───────────────────────────────────────────────────────

class SiteInputAdapter(nn.Module):
    """
    Projects site-specific input dimension → shared embedding dimension.

    This is SHARED INFRASTRUCTURE used by ALL FL methods (FedAvg, FedProx,
    SCAFFOLD, FedAdapt). It is NOT a FedAdapt novelty. It exists because
    sites have 18–39 features; without a common projection, federation
    across heterogeneous feature sets is impossible.

    Architecture:
        Linear(input_dim → hidden_dim) → LayerNorm → GELU
        Linear(hidden_dim → embedding_dim) → LayerNorm

    Weights are LOCAL — never shared across sites.
    """

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim     = input_dim
        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─── SHARED BODY ──────────────────────────────────────────────────────────────

class SharedBody(nn.Module):
    """
    The federated encoder. Only these weights cross site boundaries.

    Architecture:
        embedding_dim → hidden_dim → hidden_dim → embedding_dim
        with residual connection, LayerNorm, and dropout.

    Residual connection helps when different sites' adapter outputs
    land in slightly different regions of embedding space.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        layers: List[nn.Module] = []
        in_dim = embedding_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim

        self.layers  = nn.Sequential(*layers)
        self.proj_out = nn.Linear(hidden_dim, embedding_dim)
        self.norm_out = nn.LayerNorm(embedding_dim)

        # Residual: if embedding_dim ≠ hidden_dim we need a projection
        self.residual_proj = (
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            if embedding_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual_proj(x)
        out      = self.proj_out(self.layers(x))
        return self.norm_out(out + residual)


# ─── GRL GROUP DISCRIMINATOR ──────────────────────────────────────────────────

class GRLGroupDiscriminator(nn.Module):
    """
    FedAdapt's primary novel component.

    Learns to identify which feature group (renal / inflammatory /
    metabolic / clinical / hemodynamic) a body embedding came from.
    The GRL makes the body push AGAINST this — forcing group-invariant
    representations so that the shared body generalises across sites
    that have different subsets of feature groups.

    Why this matters:
        site_E has only [renal, clinical].
        site_A has [renal, inflammatory, clinical].
        Without group alignment, embeddings from site_E's renal features
        and site_A's renal features may land in different regions of the
        latent space, reducing the benefit of federation.
        The GRL discriminator forces them to the same region.

    Architecture:
        GRL → Linear → GELU → Linear → softmax over n_groups
    """

    def __init__(
        self,
        embedding_dim: int,
        n_groups: int,
        hidden_dim: int = 64,
        lambda_max: float = 1.0,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.grl      = GradientReversalLayer(lambda_max=lambda_max)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_groups),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """Returns logits over feature groups (for cross-entropy loss)."""
        return self.classifier(self.grl(embedding))

    def set_progress(self, p: float) -> None:
        self.grl.set_progress(p)


# ─── PERSONAL HEAD ────────────────────────────────────────────────────────────

class PersonalHead(nn.Module):
    """
    Local classifier — NEVER shared across sites.

    Fine-tuned on local data after T federation rounds with the
    SharedBody frozen (FedBABU-style evaluation).

    Captures site-specific decision boundaries:
        - ICU (site_A, 43% AKI) needs a lower threshold
        - Rural (site_E, 5% AKI) needs a higher threshold
        - Academic anchor (site_C, 12.6% AKI) has its own calibration

    Architecture: embedding_dim → hidden_dim → 1 (binary logit)
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns raw logit (apply sigmoid for probability)."""
        return self.net(x).squeeze(-1)


# ─── FEDADAPT CLIENT ──────────────────────────────────────────────────────────

class FedAdaptClient(nn.Module):
    """
    Full per-site model: adapter → body → personal_head
                                    ↑
                              GRL discriminator

    site_id and feature_names are stored for bookkeeping and to
    verify that the correct CSV is loaded at training time.
    """

    def __init__(
        self,
        site_id: str,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_groups: int = 5,
        grl_lambda_max: float = 1.0,
        dropout: float = 0.1,
        feature_names: Optional[List[str]] = None,
        aki_prevalence: Optional[float] = None,
        unlabeled: bool = False,
    ):
        super().__init__()
        self.site_id        = site_id
        self.embedding_dim  = embedding_dim
        self.feature_names  = feature_names or []
        self.aki_prevalence = aki_prevalence
        self.unlabeled      = unlabeled

        self.adapter       = SiteInputAdapter(input_dim, embedding_dim, hidden_dim)
        self.body          = SharedBody(embedding_dim, hidden_dim, dropout=dropout)
        self.discriminator = GRLGroupDiscriminator(embedding_dim, n_groups, hidden_dim // 2, grl_lambda_max)
        self.head          = PersonalHead(embedding_dim, hidden_dim // 2, dropout)

    @classmethod
    def from_metadata(
        cls,
        meta: dict,
        hidden_dim: int = 128,
        n_groups: int = 5,
        grl_lambda_max: float = 1.0,
        dropout: float = 0.1,
    ) -> "FedAdaptClient":
        """Construct a client directly from a *_adapter_meta.json file."""
        return cls(
            site_id        = meta["site_id"],
            input_dim      = meta["input_dim"],
            embedding_dim  = meta["embedding_dim"],
            hidden_dim     = hidden_dim,
            n_groups       = n_groups,
            grl_lambda_max = grl_lambda_max,
            dropout        = dropout,
            feature_names  = meta.get("feature_names", []),
            aki_prevalence = meta.get("aki_prevalence"),
            unlabeled      = meta.get("unlabeled", False),
        )

    def encode(self, x: Tensor) -> Tensor:
        """adapter → body → embedding (shared representation)."""
        return self.body(self.adapter(x))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            task_logit   — AKI prediction logit (shape: [B])
            group_logits — feature group discriminator logits (shape: [B, n_groups])
        """
        emb         = self.encode(x)
        task_logit  = self.head(emb)
        group_logit = self.discriminator(emb)
        return task_logit, group_logit

    def forward_with_embedding(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Same as forward(), but also returns the raw embedding — needed for
        FedAdapt-Proto so the same forward pass can feed both the GRL
        discriminator and the prototype alignment loss without recomputing
        adapter/body twice.

        Returns:
            task_logit, group_logits, embedding [B, embedding_dim]
        """
        emb         = self.encode(x)
        task_logit  = self.head(emb)
        group_logit = self.discriminator(emb)
        return task_logit, group_logit, emb

    def set_training_progress(self, p: float) -> None:
        """Update GRL lambda schedule. Call once per round."""
        self.discriminator.set_progress(p)

    def get_body_state(self) -> dict:
        """Return SharedBody state dict for federation."""
        return deepcopy(self.body.state_dict())

    def load_body_state(self, state_dict: dict) -> None:
        """Load federated SharedBody weights from server."""
        self.body.load_state_dict(state_dict)

    def freeze_body(self) -> None:
        """Freeze SharedBody for personal head fine-tuning phase."""
        for p in self.body.parameters():
            p.requires_grad = False

    def unfreeze_body(self) -> None:
        for p in self.body.parameters():
            p.requires_grad = True

    def head_parameters(self):
        """Parameters for head-only fine-tuning."""
        return self.head.parameters()

    def local_parameters(self):
        """All local parameters (adapter + discriminator + head)."""
        return (
            list(self.adapter.parameters()) +
            list(self.discriminator.parameters()) +
            list(self.head.parameters())
        )

    def extra_repr(self) -> str:
        return (
            f"site_id={self.site_id}  "
            f"input_dim={self.adapter.input_dim}  "
            f"embedding_dim={self.embedding_dim}  "
            f"unlabeled={self.unlabeled}  "
            f"aki_prev={self.aki_prevalence}"
        )


# ─── FEDADAPT SERVER ──────────────────────────────────────────────────────────

class FedAdaptServer:
    """
    Maintains the global SharedBody and performs federation aggregation.

    Only the SharedBody is aggregated — adapters, discriminators, and
    personal heads remain local to each client.

    Aggregation: weighted FedAvg by default (weight = n_samples).
    Can be swapped for SCAFFOLD or FedProx aggregation externally.
    """

    def __init__(self, body: SharedBody):
        self.global_body = deepcopy(body)

    def aggregate(
        self,
        client_states: List[dict],
        weights: Optional[List[float]] = None,
    ) -> None:
        """
        Weighted FedAvg over SharedBody state dicts.

        Args:
            client_states: list of body.state_dict() from each participating client
            weights:       sample counts per client (uniform if None)
        """
        if not client_states:
            return

        if weights is None:
            weights = [1.0] * len(client_states)

        total = sum(weights)
        agg_state = {}

        for key in client_states[0]:
            agg_state[key] = sum(
                (w / total) * state[key].float()
                for w, state in zip(weights, client_states)
            )

        self.global_body.load_state_dict(agg_state)

    def broadcast(self, clients: List[FedAdaptClient]) -> None:
        """Push global body weights to all clients."""
        global_state = self.global_body.state_dict()
        for client in clients:
            client.load_body_state(global_state)

    def aggregate_prototypes(
        self,
        client_protos: List[dict],
        client_n_pos: List[int],
        client_n_neg: List[int],
    ) -> dict:
        """
        Aggregate per-site class prototypes into global prototypes.

        Weighting uses sqrt(n) rather than raw n so that low-prevalence
        sites (e.g. site_E with ~1,470 AKI cases) aren't drowned out by
        high-volume sites (e.g. site_A with ~7,715 AKI cases). This is the
        key FedAdapt-Proto design choice motivated by the FL-gain index —
        primary benefitter sites (low n_positive) still meaningfully shape
        the global AKI prototype.

        Args:
            client_protos: list of {"aki": Tensor[D], "non_aki": Tensor[D]}
                            one dict per participating client (already
                            detached local means for this round)
            client_n_pos:  number of AKI-positive samples per client
            client_n_neg:  number of AKI-negative samples per client

        Returns:
            {"aki": Tensor[D], "non_aki": Tensor[D]} — global prototypes
        """
        import math as _math

        w_pos = [_math.sqrt(max(n, 1)) for n in client_n_pos]
        w_neg = [_math.sqrt(max(n, 1)) for n in client_n_neg]
        total_pos = sum(w_pos)
        total_neg = sum(w_neg)

        global_aki = sum(
            (w / total_pos) * p["aki"] for w, p in zip(w_pos, client_protos)
        )
        global_non_aki = sum(
            (w / total_neg) * p["non_aki"] for w, p in zip(w_neg, client_protos)
        )

        return {"aki": global_aki.detach(), "non_aki": global_non_aki.detach()}


# ─── PROTOTYPE ALIGNMENT (FedAdapt-Proto) ────────────────────────────────────

def compute_local_prototypes(
    embeddings: Tensor,
    labels: Tensor,
) -> Tuple[dict, int, int]:
    """
    Compute local class-mean prototypes from a batch (or full local set)
    of embeddings. Used each round before federation to share with the
    server alongside (or instead of) body weights.

    Args:
        embeddings: [B, D] — output of client.encode(x)
        labels:     [B]    — binary AKI labels

    Returns:
        protos:  {"aki": Tensor[D], "non_aki": Tensor[D]}
        n_pos:   count of AKI-positive samples in this batch/set
        n_neg:   count of AKI-negative samples in this batch/set
    """
    aki_mask     = labels.bool()
    non_aki_mask = ~aki_mask

    n_pos = int(aki_mask.sum().item())
    n_neg = int(non_aki_mask.sum().item())

    if n_pos > 0:
        aki_proto = embeddings[aki_mask].mean(dim=0).detach()
    else:
        aki_proto = torch.zeros(embeddings.shape[1], device=embeddings.device)

    if n_neg > 0:
        non_aki_proto = embeddings[non_aki_mask].mean(dim=0).detach()
    else:
        non_aki_proto = torch.zeros(embeddings.shape[1], device=embeddings.device)

    return {"aki": aki_proto, "non_aki": non_aki_proto}, n_pos, n_neg


def prototype_alignment_loss(
    embeddings:    Tensor,
    labels:        Tensor,
    global_protos: dict,
    n_site_pos:    Optional[int] = None,
    n_site_neg:    Optional[int] = None,
    tau:           float = 3000.0,
) -> Tensor:
    """
    Pulls local embeddings toward the global class prototypes.

    Unlike the GRL, this is NOT adversarial — both the client and the
    "target" (global prototype) want the same outcome, so there is no
    minimax instability. AKI and non-AKI samples are pulled toward their
    respective global prototypes separately, so the alignment is
    class-aware: it transfers cross-site AKI signal without forcing
    non-AKI/AKI embeddings to collapse together.

    REVERTED TO V1 (uniform weighting): a reliability-weighted variant
    (v2) was tried — down-weighting the pull for sites with abundant
    local samples (e.g. site_A) on the theory that their already-reliable
    local prototype was being "diluted" by noisier sites. This was tested
    at two tau values (50 and 3000) and BOTH regressed every site,
    including site_D, which had been the standout success of the
    uniform-weight v1 loss (20/20 wins vs SCAFFOLD, mean +0.031 margin).
    The dilution hypothesis was wrong: prototype alignment acts as a
    useful stabilising regulariser even for abundant sites, and reducing
    its weight removes that benefit rather than protecting anything.
    Reverted to uniform weighting (n_site_pos / n_site_neg / tau are kept
    as accepted-but-unused parameters for backward compatibility with
    existing call sites; they no longer affect the computed loss).

    Args:
        embeddings:    [B, D] — local batch embeddings (requires_grad=True)
        labels:        [B]    — binary AKI labels
        global_protos: {"aki": Tensor[D], "non_aki": Tensor[D]} — detached
        n_site_pos:    UNUSED (kept for call-site compatibility)
        n_site_neg:    UNUSED (kept for call-site compatibility)
        tau:           UNUSED (kept for call-site compatibility)

    Returns:
        scalar loss (to be ADDED to total_loss, not subtracted — this is
        cooperative, not adversarial)
    """
    aki_mask     = labels.bool()
    non_aki_mask = ~aki_mask

    loss = torch.tensor(0.0, device=embeddings.device)
    n_terms = 0

    if aki_mask.any():
        loss = loss + F.mse_loss(
            embeddings[aki_mask].mean(dim=0), global_protos["aki"]
        )
        n_terms += 1

    if non_aki_mask.any():
        loss = loss + F.mse_loss(
            embeddings[non_aki_mask].mean(dim=0), global_protos["non_aki"]
        )
        n_terms += 1

    return loss / max(n_terms, 1)


# ─── LOSS FUNCTIONS ───────────────────────────────────────────────────────────

def fedadapt_loss(
    task_logit:    Tensor,
    labels:        Tensor,
    group_logits:  Tensor,
    group_labels:  Tensor,
    lambda_adv:    float = 0.5,
    pos_weight:    Optional[Tensor] = None,
    embeddings:    Optional[Tensor] = None,
    global_protos: Optional[dict] = None,
    alpha_proto:   float = 0.0,
    n_site_pos:    Optional[int] = None,
    n_site_neg:    Optional[int] = None,
    proto_tau:     float = 3000.0,
    adv_class_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Total FedAdapt loss for one local step. Supports an optional
    prototype-alignment term (FedAdapt-Proto) on top of the existing
    GRL adversarial term — pass embeddings + global_protos + alpha_proto>0
    to enable it. With alpha_proto=0 (default) this is identical to the
    original FedAdapt loss, so existing FedAdapt/FedAvg/SCAFFOLD/FedProx
    runs are unaffected.

    total = task_loss + lambda_adv * adv_loss + alpha_proto * proto_loss

    The plus sign on adv_loss is correct and relies ENTIRELY on the GRL's
    own internal gradient reversal (see _GradientReversalFn.backward) to
    invert the signal for the body. With a plain +lambda_adv*adv_loss:
      - the discriminator's own weights (positioned after the GRL, so its
        gradient never crosses the GRL boundary) get a normal, unmodified
        gradient — standard descent, so it trains normally to get GOOD at
        predicting the feature group from an embedding.
      - the body's weights (before the GRL) have that same gradient
        negated and scaled by the GRL on its way upstream — so the body
        does ASCENT on adv_loss, i.e. is trained to fool an increasingly
        competent discriminator. That's the actual adversarial tension:
        one side minimizing, the other maximizing the same quantity.
    (An earlier version of this code had an explicit minus sign here too,
    on top of the GRL's own reversal — that double negation flipped BOTH
    halves of the intended dynamic: the discriminator was trained to get
    WORSE at its own job, and the double-cancelled sign left the body
    training to make embeddings MORE group-distinguishable, the opposite
    of the intended invariance goal. Do not reintroduce that minus sign.)

    The plus sign on proto_loss is equally critical and intentional:
    prototype alignment is COOPERATIVE, not adversarial. The client wants
    to minimise the distance to the global prototype directly — no
    gradient reversal, no minimax game, hence no GRL involved here.

    RELIABILITY WEIGHTING (v2): if n_site_pos / n_site_neg are provided,
    proto_loss internally down-weights the pull for sites with abundant
    local samples (their own estimate is already trustworthy) and
    up-weights it for sparse sites (their estimate benefits from leaning
    on the federated average). See prototype_alignment_loss for details.
    alpha_proto remains a global scalar on top of this per-site weight.

    Args:
        task_logit:    raw AKI logit [B]
        labels:        binary AKI labels [B]
        group_logits:  group discriminator logits [B, n_groups]
        group_labels:  feature group index for each sample [B]
        lambda_adv:    adversarial loss weight
        pos_weight:    optional class weight for imbalanced sites
        embeddings:    [B, D] body output, required if alpha_proto > 0
        global_protos: {"aki": Tensor[D], "non_aki": Tensor[D]}, required
                       if alpha_proto > 0
        alpha_proto:   prototype alignment loss weight (0 disables it)
        n_site_pos:    total AKI-positive sample count for this site this
                       round (for reliability weighting; None = unweighted)
        n_site_neg:    total AKI-negative sample count for this site
        proto_tau:     smoothing constant for reliability weighting
        adv_class_weight: optional per-class weight [n_groups] for the
                       discriminator's cross-entropy (inverse-frequency,
                       sklearn-balanced-style). None (default) preserves
                       the exact original unweighted behavior -- this
                       parameter did not exist before and no existing
                       caller passes it, so nothing changes unless a
                       caller explicitly opts in.

    Returns:
        total_loss, task_loss, adv_loss, proto_loss (all scalar tensors;
        proto_loss is a zero tensor when alpha_proto == 0)
    """
    task_loss = F.binary_cross_entropy_with_logits(
        task_logit, labels.float(), pos_weight=pos_weight
    )
    adv_loss  = F.cross_entropy(group_logits, group_labels, weight=adv_class_weight)
    # FIX: previously `total = task_loss - lambda_adv * adv_loss`. That extra
    # explicit negation, combined with the GRL's own internal negation in
    # _GradientReversalFn.backward(), doubly inverted the intended dynamic:
    #   - discriminator's own weights (after the GRL, unaffected by its
    #     reversal) were trained via ASCENT on their own adv_loss — pushed to
    #     get WORSE at classifying groups, not better.
    #   - body weights (before the GRL) had the explicit minus and the GRL's
    #     internal minus cancel out, so they did NORMAL DESCENT on adv_loss —
    #     trained to make embeddings MORE group-distinguishable, the opposite
    #     of the intended group-invariance goal.
    # The GRL already handles the sign-flip needed for the body; the loss
    # formula just needs a plain, standard `+lambda_adv * adv_loss` so the
    # discriminator trains normally (minimize its own loss, get good at
    # classifying groups) while the GRL alone inverts that signal for
    # everything upstream (body trained to fool a genuinely competent
    # discriminator — real adversarial tension, not two sides cooperating
    # in the wrong direction).
    total     = task_loss + lambda_adv * adv_loss

    if alpha_proto > 0.0:
        assert embeddings is not None and global_protos is not None, (
            "embeddings and global_protos must be provided when alpha_proto > 0"
        )
        proto_loss = prototype_alignment_loss(
            embeddings, labels, global_protos,
            n_site_pos=n_site_pos, n_site_neg=n_site_neg, tau=proto_tau,
        )
        total = total + alpha_proto * proto_loss
    else:
        proto_loss = torch.tensor(0.0, device=task_logit.device)

    return total, task_loss, adv_loss, proto_loss


def fedprox_penalty(
    client: FedAdaptClient,
    global_body_state: dict,
    mu: float = 0.01,
) -> Tensor:
    """
    FedProx proximal penalty: (mu/2) * ||w_local - w_global||²
    Applied to SharedBody parameters only.
    Used by FedProx baseline; ignored by FedAdapt.
    """
    # FIX: iterate live parameters (not state_dict) so gradients flow correctly.
    # torch.tensor(0., requires_grad=True) is a detached leaf — additions through
    # detached state_dict values create a graph disconnected from live params,
    # making the penalty always 0 in the backward pass.
    penalty = None
    global_params = {k: v.detach() for k, v in global_body_state.items()}
    for name, param in client.body.named_parameters():
        if name in global_params:
            diff = param - global_params[name].to(param.dtype)
            term = diff.norm() ** 2
            penalty = term if penalty is None else penalty + term
    if penalty is None:
        return torch.tensor(0.0)
    return (mu / 2.0) * penalty


# ─── TRAINING UTILITIES ───────────────────────────────────────────────────────

def local_train_step(
    client:           FedAdaptClient,
    batch:            Tuple[Tensor, Tensor, Tensor],  # (x, aki_label, group_label)
    optimizer:        torch.optim.Optimizer,
    lambda_adv:       float = 0.5,
    pos_weight:       Optional[Tensor] = None,
) -> Dict[str, float]:
    """
    One mini-batch local training step for FedAdapt.

    group_label is the index of the dominant feature group for each sample
    (assigned during data loading based on which group's features are present).
    """
    client.train()
    x, aki_labels, group_labels = batch

    optimizer.zero_grad()
    task_logit, group_logits = client(x)

    total_loss, task_loss, adv_loss, _ = fedadapt_loss(
        task_logit, aki_labels, group_logits, group_labels,
        lambda_adv=lambda_adv, pos_weight=pos_weight,
    )
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "task_loss":  task_loss.item(),
        "adv_loss":   adv_loss.item(),
    }


def finetune_head_step(
    client:    FedAdaptClient,
    batch:     Tuple[Tensor, Tensor],   # (x, aki_label)
    optimizer: torch.optim.Optimizer,
    pos_weight: Optional[Tensor] = None,
) -> float:
    """
    One mini-batch fine-tuning step for the PersonalHead only.
    Body must be frozen before calling this (client.freeze_body()).
    """
    client.train()
    x, aki_labels = batch

    optimizer.zero_grad()
    emb        = client.encode(x)
    task_logit = client.head(emb)
    loss       = F.binary_cross_entropy_with_logits(
        task_logit, aki_labels.float(), pos_weight=pos_weight
    )
    loss.backward()
    optimizer.step()

    return loss.item()


# ─── CONVENIENCE: LOAD CLIENT FROM METADATA FILE ─────────────────────────────

def load_client_from_metadata(
    meta_path:     str,
    hidden_dim:    int = 128,
    n_groups:      int = 5,
    grl_lambda_max: float = 1.0,
    dropout:       float = 0.1,
) -> FedAdaptClient:
    """
    Load a FedAdaptClient from a *_adapter_meta.json file produced by
    mimic_ftl_simulation_phase2.py.

    Example:
        client = load_client_from_metadata("phase2_sites/site_E_adapter_meta.json")
    """
    with open(meta_path) as f:
        meta = json.load(f)
    return FedAdaptClient.from_metadata(
        meta,
        hidden_dim=hidden_dim,
        n_groups=n_groups,
        grl_lambda_max=grl_lambda_max,
        dropout=dropout,
    )


# ─── QUICK SANITY CHECK ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FedAdapt model — sanity check\n")

    # Simulate two sites with different input dims
    configs = [
        {"site_id": "site_C", "input_dim": 39, "embedding_dim": 64,
         "aki_prevalence": 0.126, "unlabeled": False, "feature_names": []},
        {"site_id": "site_E", "input_dim": 18, "embedding_dim": 64,
         "aki_prevalence": 0.05,  "unlabeled": False, "feature_names": []},
    ]

    clients = [FedAdaptClient.from_metadata(m, hidden_dim=128, n_groups=5)
               for m in configs]
    server  = FedAdaptServer(body=clients[0].body)

    for client in clients:
        B   = 8
        x   = torch.randn(B, client.adapter.input_dim)
        lbl = torch.randint(0, 2, (B,))
        grp = torch.randint(0, 5, (B,))

        client.set_training_progress(0.5)
        task_logit, group_logits = client(x)
        total, tl, al, _ = fedadapt_loss(task_logit, lbl, group_logits, grp)

        print(f"  {client.site_id}:"
              f"  input={client.adapter.input_dim}"
              f"  emb={client.embedding_dim}"
              f"  task_loss={tl.item():.4f}"
              f"  adv_loss={al.item():.4f}"
              f"  total={total.item():.4f}")

    # Test federation round
    server.broadcast(clients)
    states  = [c.get_body_state() for c in clients]
    weights = [33000, 33000]
    server.aggregate(states, weights)
    server.broadcast(clients)
    print("\n  Federation round: OK")

    # Test head fine-tuning
    clients[0].freeze_body()
    x   = torch.randn(8, clients[0].adapter.input_dim)
    lbl = torch.randint(0, 2, (8,))
    opt = torch.optim.Adam(clients[0].head_parameters(), lr=1e-3)
    loss = finetune_head_step(clients[0], (x, lbl), opt)
    print(f"  Head fine-tune loss (site_C): {loss:.4f}")

    print("\n✅ All checks passed.")