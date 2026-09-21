"""
Extended model architectures v3: orchestrator variants, MixStyle, FACT.
Imports and extends the original pathology_models.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import sys
sys.path.insert(0, '/mnt/shared-workspace/shared')
from pathology_models import (
    BaselineClassifier, DANNModel, SSLModel, DisentanglementModel,
    MultiAgentSystem, nt_xent_loss, grl, GradientReversalFn,
    SpecialistAgent, ImageQualityAgent, TissueClassificationAgent,
    DomainAdaptationAgent, TumorDetectionAgent, BiomarkerQuantificationAgent,
    UncertaintySafetyAgent, CPOOrchestrator, QualityReviewer
)

# ============================================================
# Orchestrator Variant B: Uncertainty-Weighted Fusion
# ============================================================

class UncertaintyWeightedOrchestrator(nn.Module):
    """
    Non-learned fusion: weight each agent by inverse MC-Dropout variance.
    Agents with low uncertainty (high confidence) get more weight.
    """
    def __init__(self, input_dim=512, n_agents=6, hidden=64):
        super().__init__()
        self.name = "UncertaintyWeightedOrchestrator"
        self.n_agents = n_agents
        # Keep a simple calibration layer for the weighted average
        self.calibration = nn.Sequential(
            nn.Linear(1, 1),
        )
        self.agent_weights = nn.Parameter(torch.ones(n_agents) / n_agents)

    def forward(self, x, agent_outputs, agent_uncertainties=None):
        """
        Args:
            x: input features (B, dim)
            agent_outputs: list of (B,) tensors
            agent_uncertainties: list of (B,) uncertainty tensors (std)
        Returns:
            fused_logit: (B,) final prediction logit
            weights: (n_agents,) effective weights
        """
        stacked = torch.stack(agent_outputs, dim=1)  # (B, n_agents)

        if agent_uncertainties is not None:
            # Weight by inverse variance: w_i = 1/(sigma_i^2 + eps)
            eps = 1e-6
            inv_var = torch.stack([
                1.0 / (u**2 + eps) for u in agent_uncertainties
            ], dim=1)  # (B, n_agents)
            # Normalize weights per sample
            weights_per_sample = inv_var / inv_var.sum(dim=1, keepdim=True)  # (B, n_agents)
            # Weighted average
            fused_prob = (stacked * weights_per_sample).sum(dim=1)  # (B,)
        else:
            # Fallback to uniform
            fused_prob = stacked.mean(dim=1)

        # Convert probability to logit for consistency
        fused_prob = fused_prob.clamp(1e-6, 1 - 1e-6)
        fused_logit = torch.log(fused_prob / (1 - fused_prob))

        return fused_logit, self.agent_weights.detach()


class MultiAgentSystemV3(nn.Module):
    """
    Multi-agent system with configurable orchestrator variant.
    variant: 'original', 'uncertainty_weighted', 'ood_validated'
    """
    def __init__(self, input_dim=512, n_domains=4, dropout=0.1, variant='uncertainty_weighted',
                 disentanglement_lambda=1.0):
        super().__init__()
        self.variant = variant
        self.disentanglement_lambda = disentanglement_lambda

        # 6 specialist agents (same as original)
        self.quality_agent = ImageQualityAgent(input_dim, dropout)
        self.tissue_agent = TissueClassificationAgent(input_dim, dropout)
        self.domain_agent = DomainAdaptationAgent(input_dim, n_domains=n_domains, dropout=dropout)
        self.tumor_agent = TumorDetectionAgent(input_dim, dropout)
        self.biomarker_agent = BiomarkerQuantificationAgent(input_dim, dropout)
        self.uncertainty_agent = UncertaintySafetyAgent(input_dim, dropout=0.2)

        # Orchestrator variant
        if variant == 'uncertainty_weighted':
            self.orchestrator = UncertaintyWeightedOrchestrator(input_dim, n_agents=6)
        else:
            self.orchestrator = CPOOrchestrator(input_dim, n_agents=6)

        self.reviewer = QualityReviewer(n_agents=6)

    def forward(self, x, lambda_=1.0):
        # Stage 1: Image Quality
        quality_score, q_feat = self.quality_agent(x)

        # Stage 2: Tissue Classification
        tissue_logit, t_feat = self.tissue_agent(x)
        tissue_prob = torch.sigmoid(tissue_logit).squeeze()

        # Stage 3: Domain Adaptation
        domain_logit, z_bio = self.domain_agent(x, lambda_)

        # Stage 4: Tumor Detection
        tumor_logit, tumor_feat = self.tumor_agent(x)
        tumor_prob = torch.sigmoid(tumor_logit).squeeze()

        # Stage 5: Biomarker Quantification
        biomarker_score, b_feat = self.biomarker_agent(x)

        # Stage 6: Uncertainty Estimation
        unc_pred, unc_epistemic, unc_entropy = self.uncertainty_agent.estimate_uncertainty(x)

        # Stage 7: Orchestrator fusion
        agent_outputs = [quality_score, tissue_prob, tumor_prob,
                         biomarker_score, unc_pred, tumor_prob]
        agent_outputs = [ao.squeeze() if ao.dim() > 1 else ao for ao in agent_outputs]

        if self.variant == 'uncertainty_weighted':
            # Compute per-agent uncertainty for weighting
            agent_uncs = [torch.zeros_like(ao) for ao in agent_outputs]
            # Uncertainty agent has its own epistemic uncertainty
            agent_uncs[4] = unc_epistemic
            # For other agents, use MC-Dropout on each
            for i, (agent_module, idx) in enumerate([
                (self.quality_agent, 0), (self.tissue_agent, 1),
                (self.tumor_agent, 2), (self.biomarker_agent, 3),
                (self.tumor_agent, 5)
            ]):
                agent_module.train()
                with torch.no_grad():
                    if idx in [0, 1, 2, 5]:  # agents with logit output
                        outs = torch.stack([torch.sigmoid(agent_module(x)[0]).squeeze()
                                           if idx in [1, 2] else
                                           (agent_module(x)[0].squeeze() if idx == 0 else
                                            torch.sigmoid(agent_module(x)[0]).squeeze())
                                           for _ in range(10)])
                    else:  # biomarker
                        outs = torch.stack([agent_module(x)[0].squeeze() for _ in range(10)])
                agent_module.eval()
                agent_uncs[idx] = outs.std(0)

            fused_logit, agent_weights = self.orchestrator(x, agent_outputs, agent_uncs)
        else:
            fused_logit, agent_weights = self.orchestrator(x, agent_outputs)

        # Quality Reviewer
        consensus, disagreement, flag_review = self.reviewer(agent_outputs)

        return {
            'quality_score': quality_score,
            'tissue_prob': tissue_prob,
            'domain_logit': domain_logit,
            'z_bio': z_bio,
            'tumor_prob': tumor_prob,
            'tumor_logit': tumor_logit.squeeze(),
            'biomarker_score': biomarker_score,
            'uncertainty_pred': unc_pred,
            'uncertainty_epistemic': unc_epistemic,
            'uncertainty_entropy': unc_entropy,
            'fused_logit': fused_logit,
            'agent_weights': agent_weights,
            'consensus': consensus,
            'disagreement': disagreement,
            'flag_for_review': flag_review,
        }

    def predict(self, x, lambda_=1.0):
        out = self.forward(x, lambda_)
        return torch.sigmoid(out['fused_logit'])

    def predict_with_uncertainty(self, x, n_samples=30):
        self.train()
        with torch.no_grad():
            outs = torch.stack([self.predict(x) for _ in range(n_samples)])
        self.eval()
        return outs.mean(0), outs.std(0)

    def get_agent_parameters(self):
        groups = {
            'quality': list(self.quality_agent.parameters()),
            'tissue': list(self.tissue_agent.parameters()),
            'domain': list(self.domain_agent.parameters()),
            'tumor': list(self.tumor_agent.parameters()),
            'biomarker': list(self.biomarker_agent.parameters()),
            'uncertainty': list(self.uncertainty_agent.parameters()),
            'orchestrator': list(self.orchestrator.parameters()),
        }
        return groups


# ============================================================
# MixStyle Baseline
# ============================================================

class MixStyleClassifier(nn.Module):
    """
    Baseline MLP with MixStyle augmentation during training.
    MixStyle mixes instance-level feature statistics (mean/std) across
    samples from different domains in a batch.
    """
    def __init__(self, input_dim=512, hidden=256, dropout=0.1, n_domains=4, mix_prob=0.5, alpha=0.1):
        super().__init__()
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.n_domains = n_domains

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden // 2, 1)

    def mixstyle(self, x, domains):
        """Mix feature statistics across domains."""
        if not self.training or np.random.random() > self.mix_prob:
            return x

        B = x.size(0)
        # Compute per-instance statistics
        feat_mean = x.mean(dim=1, keepdim=True)
        feat_std = x.std(dim=1, keepdim=True)
        feat_norm = (x - feat_mean) / (feat_std + 1e-6)

        # Find samples from different domains
        unique_domains = torch.unique(domains)
        if len(unique_domains) < 2:
            return x

        # Randomly permute within batch to mix statistics
        perm = torch.randperm(B, device=x.device)
        # Ensure we mix across domains
        for _ in range(3):  # Try a few times to get cross-domain pairs
            if (domains[perm] != domains).any():
                break
            perm = torch.randperm(B, device=x.device)

        mixed_mean = (1 - self.alpha) * feat_mean + self.alpha * feat_mean[perm]
        mixed_std = (1 - self.alpha) * feat_std + self.alpha * feat_std[perm]

        return feat_norm * mixed_std + mixed_mean

    def forward(self, x, domains=None, return_features=False):
        if domains is not None:
            x = self.mixstyle(x, domains)
        feat = self.encoder(x)
        logit = self.classifier(feat)
        if return_features:
            return logit, feat
        return logit

    def predict_with_uncertainty(self, x, n_samples=30):
        self.train()
        with torch.no_grad():
            outs = torch.stack([torch.sigmoid(self.forward(x)).squeeze() for _ in range(n_samples)])
        self.eval()
        return outs.mean(0), outs.std(0)


# ============================================================
# FACT Baseline (Feature Augmentation by Channel-wise sTatistics)
# ============================================================

class FACTClassifier(nn.Module):
    """
    Baseline MLP with FACT augmentation during training.
    FACT mixes channel-wise statistics across domains.
    """
    def __init__(self, input_dim=512, hidden=256, dropout=0.1, n_domains=4, mix_prob=0.5, alpha=0.1):
        super().__init__()
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.n_domains = n_domains

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden // 2, 1)

    def fact_augment(self, x, domains):
        """Mix channel-wise statistics across domains."""
        if not self.training or np.random.random() > self.mix_prob:
            return x

        B, D = x.shape
        unique_domains = torch.unique(domains)
        if len(unique_domains) < 2:
            return x

        # Channel-wise statistics
        chan_mean = x.mean(dim=0)  # (D,)
        chan_std = x.std(dim=0)    # (D,)

        # Per-domain channel statistics
        perm = torch.randperm(B, device=x.device)
        for _ in range(3):
            if (domains[perm] != domains).any():
                break
            perm = torch.randperm(B, device=x.device)

        # Mix channel statistics
        x_perm = x[perm]
        chan_mean_perm = x_perm.mean(dim=0)
        chan_std_perm = x_perm.std(dim=0)

        mixed_mean = (1 - self.alpha) * chan_mean + self.alpha * chan_mean_perm
        mixed_std = (1 - self.alpha) * chan_std + self.alpha * chan_std_perm

        # Apply mixed statistics
        x_norm = (x - chan_mean) / (chan_std + 1e-6)
        return x_norm * mixed_std + mixed_mean

    def forward(self, x, domains=None, return_features=False):
        if domains is not None:
            x = self.fact_augment(x, domains)
        feat = self.encoder(x)
        logit = self.classifier(feat)
        if return_features:
            return logit, feat
        return logit

    def predict_with_uncertainty(self, x, n_samples=30):
        self.train()
        with torch.no_grad():
            outs = torch.stack([torch.sigmoid(self.forward(x)).squeeze() for _ in range(n_samples)])
        self.eval()
        return outs.mean(0), outs.std(0)


# ============================================================
# Disentanglement with configurable lambda
# ============================================================

class DisentanglementModelV3(DisentanglementModel):
    """Disentanglement model with configurable loss weight."""
    def __init__(self, input_dim=512, bio_dim=256, dom_dim=128, n_domains=4,
                 dropout=0.1, disentanglement_lambda=1.0):
        super().__init__(input_dim, bio_dim, dom_dim, n_domains, dropout)
        self.disentanglement_lambda = disentanglement_lambda
