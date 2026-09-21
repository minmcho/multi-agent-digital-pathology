"""
Model architectures for multi-agent pathology AI framework.
All models operate on pre-extracted 512-dim ResNet-18 features.

5 models compared:
  1. BaselineClassifier     - Simple MLP tumor classifier
  2. DANNModel              - Domain-Adversarial Neural Network
  3. SSLModel               - Contrastive SSL + tumor classifier
  4. DisentanglementModel   - Monolithic bio/domain disentanglement
  5. MultiAgentSystem       - 6 specialist agents + CPO orchestrator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

# ============================================================
# Shared utilities
# ============================================================

class GradientReversalFn(torch.autograd.Function):
    """Gradient Reversal Layer for domain adversarial training."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        return -ctx.lambda_ * grads, None


def grl(x, lambda_=1.0):
    return GradientReversalFn.apply(x, lambda_)


class MCDropout(nn.Module):
    """Monte Carlo Dropout wrapper for uncertainty estimation."""
    def __init__(self, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, mc_samples=1):
        if self.training or mc_samples == 1:
            return self.dropout(x)
        # During eval with mc_samples > 1, enable dropout manually
        self.dropout.train()
        outs = torch.stack([self.dropout(x) for _ in range(mc_samples)])
        self.dropout.eval()
        return outs.mean(0), outs.std(0)  # mean and epistemic uncertainty


# ============================================================
# Model 1: Baseline Classifier
# ============================================================

class BaselineClassifier(nn.Module):
    """Simple MLP tumor classifier on ResNet-18 features."""
    def __init__(self, input_dim=512, hidden=256, dropout=0.1):
        super().__init__()
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

    def forward(self, x, return_features=False):
        feat = self.encoder(x)
        logit = self.classifier(feat)
        if return_features:
            return logit, feat
        return logit

    def predict_with_uncertainty(self, x, n_samples=30):
        """MC-Dropout uncertainty estimation."""
        self.train()  # Enable dropout
        with torch.no_grad():
            outs = torch.stack([torch.sigmoid(self.forward(x)).squeeze() for _ in range(n_samples)])
        self.eval()
        mean = outs.mean(0)
        std = outs.std(0)
        return mean, std


# ============================================================
# Model 2: DANN (Domain-Adversarial Neural Network)
# ============================================================

class DANNModel(nn.Module):
    """Domain-Adversarial training: shared encoder + tumor + domain heads."""
    def __init__(self, input_dim=512, hidden=256, n_domains=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tumor_classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_domains),
        )

    def forward(self, x, lambda_=1.0, return_features=False):
        feat = self.encoder(x)
        tumor_logit = self.tumor_classifier(feat)
        dom_feat = grl(feat, lambda_)
        domain_logit = self.domain_classifier(dom_feat)
        if return_features:
            return tumor_logit, domain_logit, feat
        return tumor_logit, domain_logit

    def predict_with_uncertainty(self, x, n_samples=30):
        self.train()
        with torch.no_grad():
            outs = torch.stack([torch.sigmoid(self.forward(x)[0]).squeeze() for _ in range(n_samples)])
        self.eval()
        return outs.mean(0), outs.std(0)


# ============================================================
# Model 3: SSL (Contrastive Self-Supervised + Tumor Classifier)
# ============================================================

class SSLModel(nn.Module):
    """Contrastive SSL pre-training + tumor classification fine-tuning."""
    def __init__(self, input_dim=512, hidden=256, proj_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )
        # Tumor classifier (added during fine-tuning)
        self.tumor_classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, return_features=False):
        feat = self.encoder(x)
        logit = self.tumor_classifier(feat)
        if return_features:
            return logit, feat
        return logit

    def project(self, x):
        """Project features for contrastive loss."""
        feat = self.encoder(x)
        return F.normalize(self.projector(feat), dim=1)

    def predict_with_uncertainty(self, x, n_samples=30):
        self.train()
        with torch.no_grad():
            outs = torch.stack([torch.sigmoid(self.forward(x)).squeeze() for _ in range(n_samples)])
        self.eval()
        return outs.mean(0), outs.std(0)


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)."""
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)
    # Mask out self-similarity with large negative value
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
    labels = torch.cat([torch.arange(batch_size, 2 * batch_size),
                        torch.arange(0, batch_size)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ============================================================
# Model 4: Disentanglement Model (Monolithic)
# ============================================================

class DisentanglementModel(nn.Module):
    """
    Monolithic disentanglement: separates Z_bio (biological) and Z_dom (domain).
    - Z_bio: used for tumor classification (domain-invariant)
    - Z_dom: used for domain classification (via GRL on shared encoder)
    - Orthogonality loss between Z_bio and Z_dom
    - Contrastive loss on Z_bio for SSL
    - MC-Dropout for uncertainty
    """
    def __init__(self, input_dim=512, bio_dim=256, dom_dim=128, n_domains=4, dropout=0.1):
        super().__init__()
        # Shared feature encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # Biological latent (domain-invariant)
        self.bio_encoder = nn.Sequential(
            nn.Linear(256, bio_dim),
            nn.BatchNorm1d(bio_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Domain latent
        self.dom_encoder = nn.Sequential(
            nn.Linear(256, dom_dim),
            nn.BatchNorm1d(dom_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Tumor classifier from Z_bio
        self.tumor_classifier = nn.Sequential(
            nn.Linear(bio_dim, bio_dim // 2),
            nn.BatchNorm1d(bio_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bio_dim // 2, 1),
        )
        # Domain classifier from Z_dom
        self.domain_classifier = nn.Sequential(
            nn.Linear(dom_dim, dom_dim // 2),
            nn.BatchNorm1d(dom_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dom_dim // 2, n_domains),
        )
        # Projection head for contrastive learning on Z_bio
        self.projector = nn.Sequential(
            nn.Linear(bio_dim, bio_dim),
            nn.ReLU(),
            nn.Linear(bio_dim, 128),
        )

    def encode(self, x, lambda_=1.0):
        """Encode to Z_bio and Z_dom."""
        shared = self.shared_encoder(x)
        z_bio = self.bio_encoder(shared)
        # GRL on shared features before domain encoding
        dom_input = grl(shared, lambda_)
        z_dom = self.dom_encoder(dom_input)
        return z_bio, z_dom

    def forward(self, x, lambda_=1.0, return_latents=False):
        z_bio, z_dom = self.encode(x, lambda_)
        tumor_logit = self.tumor_classifier(z_bio)
        domain_logit = self.domain_classifier(z_dom)
        if return_latents:
            return tumor_logit, domain_logit, z_bio, z_dom
        return tumor_logit, domain_logit

    def project_bio(self, x):
        """Project Z_bio for contrastive loss."""
        z_bio, _ = self.encode(x)
        return F.normalize(self.projector(z_bio), dim=1)

    def orthogonality_loss(self, z_bio, z_dom):
        """Encourage orthogonality between bio and domain latents.
        Uses cross-covariance Frobenius norm (dimension-agnostic)."""
        # Center
        z_bio_c = z_bio - z_bio.mean(0, keepdim=True)
        z_dom_c = z_dom - z_dom.mean(0, keepdim=True)
        # Cross-covariance matrix (bio_dim x dom_dim)
        cross_cov = z_bio_c.T @ z_dom_c / z_bio.size(0)
        # Minimize off-diagonal cross-covariance
        return (cross_cov ** 2).sum() - (cross_cov.diag() ** 2).sum() if cross_cov.size(0) == cross_cov.size(1) else (cross_cov ** 2).mean()

    def predict_with_uncertainty(self, x, n_samples=30):
        self.train()
        with torch.no_grad():
            outs = torch.stack([torch.sigmoid(self.forward(x)[0]).squeeze() for _ in range(n_samples)])
        self.eval()
        return outs.mean(0), outs.std(0)


# ============================================================
# Model 5: Multi-Agent System
# ============================================================

class SpecialistAgent(nn.Module):
    """Base class for specialist agents."""
    def __init__(self, name, input_dim=512, hidden=128, output_dim=1, dropout=0.1):
        super().__init__()
        self.name = name
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat), feat

    def predict_with_uncertainty(self, x, n_samples=30):
        self.train()
        with torch.no_grad():
            outs = torch.stack([torch.sigmoid(self.forward(x)[0]).squeeze() for _ in range(n_samples)])
        self.eval()
        return outs.mean(0), outs.std(0)


class ImageQualityAgent(SpecialistAgent):
    """Agent 1: Assesses image quality (pass/fail)."""
    def __init__(self, input_dim=512, dropout=0.1):
        super().__init__("ImageQuality", input_dim, hidden=64, output_dim=1, dropout=dropout)

    def forward(self, x):
        feat = self.encoder(x)
        logit = self.head(feat)
        return torch.sigmoid(logit).squeeze(), feat


class TissueClassificationAgent(SpecialistAgent):
    """Agent 2: Classifies tissue type (tumor/normal as proxy)."""
    def __init__(self, input_dim=512, dropout=0.1):
        super().__init__("TissueClassification", input_dim, hidden=128, output_dim=1, dropout=dropout)


class DomainAdaptationAgent(nn.Module):
    """Agent 3: Domain adaptation with GRL - disentangles domain from biology."""
    def __init__(self, input_dim=512, bio_dim=128, dom_dim=64, n_domains=4, dropout=0.1):
        super().__init__()
        self.name = "DomainAdaptation"
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.bio_head = nn.Sequential(
            nn.Linear(128, bio_dim),
            nn.BatchNorm1d(bio_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dom_head = nn.Sequential(
            nn.Linear(128, dom_dim),
            nn.BatchNorm1d(dom_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dom_dim, n_domains),
        )

    def forward(self, x, lambda_=1.0):
        shared = self.shared(x)
        z_bio = self.bio_head(shared)
        dom_input = grl(shared, lambda_)
        domain_logit = self.dom_head(dom_input)
        return domain_logit, z_bio

    def get_bio_features(self, x):
        shared = self.shared(x)
        return self.bio_head(shared)


class TumorDetectionAgent(SpecialistAgent):
    """Agent 4: Main tumor detection (binary classification)."""
    def __init__(self, input_dim=512, dropout=0.1):
        super().__init__("TumorDetection", input_dim, hidden=256, output_dim=1, dropout=dropout)


class BiomarkerQuantificationAgent(nn.Module):
    """Agent 5: Continuous biomarker (tumor burden) quantification."""
    def __init__(self, input_dim=512, dropout=0.1):
        super().__init__()
        self.name = "BiomarkerQuantification"
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feat = self.encoder(x)
        score = torch.sigmoid(self.regressor(feat)).squeeze()
        return score, feat


class UncertaintySafetyAgent(nn.Module):
    """Agent 6: Uncertainty estimation via MC-Dropout + ensemble disagreement."""
    def __init__(self, input_dim=512, dropout=0.2):
        super().__init__()
        self.name = "UncertaintySafety"
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.dropout1(feat)
        feat = F.relu(feat)
        feat = self.dropout2(feat)
        logit = self.fc(feat)
        return torch.sigmoid(logit).squeeze(), feat

    def estimate_uncertainty(self, x, n_samples=50):
        """Estimate epistemic uncertainty via MC-Dropout."""
        self.train()
        with torch.no_grad():
            outs = torch.stack([self.forward(x)[0] for _ in range(n_samples)])
        self.eval()
        mean_pred = outs.mean(0)
        epistemic = outs.std(0)
        # Predictive entropy
        eps = 1e-7
        entropy = -(mean_pred * torch.log(mean_pred + eps) +
                     (1 - mean_pred) * torch.log(1 - mean_pred + eps))
        return mean_pred, epistemic, entropy


class CPOOrchestrator(nn.Module):
    """
    Chief Pathology Officer: attention-based orchestrator that routes
    and weights specialist agent outputs.
    """
    def __init__(self, input_dim=512, n_agents=6, hidden=64):
        super().__init__()
        self.name = "CPOOrchestrator"
        # Attention-based routing
        self.query = nn.Linear(input_dim, hidden)
        self.keys = nn.Linear(input_dim, hidden)
        self.agent_weights = nn.Parameter(torch.ones(n_agents) / n_agents)
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(n_agents, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, agent_outputs):
        """
        Args:
            x: input features (B, 512)
            agent_outputs: list of (B,) tensors from each agent
        Returns:
            fused_logit: (B,) final prediction logit
            weights: (B, n_agents) attention weights per sample
        """
        # Stack agent outputs
        stacked = torch.stack(agent_outputs, dim=1)  # (B, n_agents)

        # Compute attention weights based on input features
        q = self.query(x)  # (B, hidden)
        # Use learned agent weights modulated by input
        attn_scores = stacked * self.agent_weights.unsqueeze(0)
        fused_logit = self.fusion(attn_scores).squeeze(-1)

        return fused_logit, self.agent_weights.detach()


class QualityReviewer(nn.Module):
    """
    Quality Reviewer: checks consistency across agents and flags
    disagreements for human review.
    """
    def __init__(self, n_agents=6):
        super().__init__()
        self.name = "QualityReviewer"
        self.consistency_threshold = 0.15

    def forward(self, agent_probs):
        """
        Args:
            agent_probs: list of probability tensors from agents
        Returns:
            consensus: (B,) consensus probability
            disagreement: (B,) disagreement score
            flag_for_review: (B,) boolean mask
        """
        stacked = torch.stack(agent_probs, dim=1)  # (B, n_agents)
        consensus = stacked.mean(dim=1)
        disagreement = stacked.std(dim=1)
        flag_for_review = disagreement > self.consistency_threshold
        return consensus, disagreement, flag_for_review


class MultiAgentSystem(nn.Module):
    """
    Full multi-agent pathology system.
    7-stage pipeline: quality → tissue → domain → tumor → biomarker → uncertainty → reporting
    """
    def __init__(self, input_dim=512, n_domains=4, dropout=0.1):
        super().__init__()
        # 6 specialist agents
        self.quality_agent = ImageQualityAgent(input_dim, dropout)
        self.tissue_agent = TissueClassificationAgent(input_dim, dropout)
        self.domain_agent = DomainAdaptationAgent(input_dim, n_domains=n_domains, dropout=dropout)
        self.tumor_agent = TumorDetectionAgent(input_dim, dropout)
        self.biomarker_agent = BiomarkerQuantificationAgent(input_dim, dropout)
        self.uncertainty_agent = UncertaintySafetyAgent(input_dim, dropout=0.2)

        # Orchestrator and reviewer
        self.orchestrator = CPOOrchestrator(input_dim, n_agents=6)
        self.reviewer = QualityReviewer(n_agents=6)

    def forward(self, x, lambda_=1.0):
        """
        Full pipeline forward pass.
        Returns dict of all agent outputs + final fused prediction.
        """
        # Stage 1: Image Quality
        quality_score, q_feat = self.quality_agent(x)

        # Stage 2: Tissue Classification
        tissue_logit, t_feat = self.tissue_agent(x)
        tissue_prob = torch.sigmoid(tissue_logit).squeeze()

        # Stage 3: Domain Adaptation
        domain_logit, z_bio = self.domain_agent(x, lambda_)

        # Stage 4: Tumor Detection (uses bio features from domain agent)
        tumor_logit, tumor_feat = self.tumor_agent(x)
        tumor_prob = torch.sigmoid(tumor_logit).squeeze()

        # Stage 5: Biomarker Quantification
        biomarker_score, b_feat = self.biomarker_agent(x)

        # Stage 6: Uncertainty Estimation
        unc_pred, unc_epistemic, unc_entropy = self.uncertainty_agent.estimate_uncertainty(x)

        # Stage 7: Orchestrator fusion
        agent_outputs = [quality_score, tissue_prob, tumor_prob,
                         biomarker_score, unc_pred, tumor_prob]  # 6 outputs
        # Ensure all are same shape
        agent_outputs = [ao.squeeze() if ao.dim() > 1 else ao for ao in agent_outputs]
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
        """Get final prediction probability."""
        out = self.forward(x, lambda_)
        return torch.sigmoid(out['fused_logit'])

    def predict_with_uncertainty(self, x, n_samples=30):
        """MC-Dropout uncertainty for the full system."""
        self.train()
        with torch.no_grad():
            outs = torch.stack([self.predict(x) for _ in range(n_samples)])
        self.eval()
        return outs.mean(0), outs.std(0)

    def get_agent_parameters(self):
        """Return parameter groups for each agent (for separate optimization)."""
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
