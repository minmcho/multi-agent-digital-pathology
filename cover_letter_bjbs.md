# Cover Letter — British Journal of Biomedical Science

**To:** The Editor, British Journal of Biomedical Science
**Re:** Submission for Special Issue "Digital Pathology and Artificial Intelligence (AI)"
**Date:** [Submission date]

---

Dear Editor,

We are pleased to submit our manuscript, "Foundation Model Quality Sets the Generalisation Ceiling in Digital Pathology: A Systematic Multi-Backbone Benchmark of Architectural Interventions," for consideration in the BJBS Special Issue on Digital Pathology and Artificial Intelligence.

This study addresses a question of direct practical importance for clinical AI deployment: given the rapid proliferation of pathology-specific foundation models, how much benefit do architectural interventions—domain adaptation, disentangled representations, uncertainty quantification, multi-agent fusion—provide above the feature extractor? We systematically benchmarked a modular specialist framework comprising six sub-networks coordinated by an attention-based orchestrator across six foundation model backbones (ResNet-18, Phikon ViT-Tiny, Phikon ViT-B, UNI ViT-L, Virchow ViT-L, and CONCH ViT-B), five architectures, two domain generalisation baselines (MixStyle, FACT), and two orchestrator variants, trained with 10 random seeds each.

Our key findings include:

1. **Foundation model quality sets the generalisation ceiling.** With ViT-B features, all architectures achieved OOD AUC above 0.985 (generalisation gap 0.007–0.014), and no architectural intervention reached statistical significance. With ViT-Tiny features, the baseline collapsed to near-random OOD AUC (0.537), while DANN improved to 0.663 (DeLong p < 0.001). The return on investment for architectural complexity diminishes rapidly as foundation model capacity increases.

2. **No fusion strategy compensates for poor agent-level robustness.** The uncertainty-only agent achieved OOD AUC of 0.829, but no orchestrator fusion strategy could recover this performance (best variant: 0.637), redirecting the research agenda from fusion architecture to agent-level robustness.

3. **Calibration does not transfer across domains.** Temperature scaling learned on in-distribution data had minimal effect on OOD calibration (ΔECE < 0.02), a critical finding for clinical deployment.

4. **Clinical utility exposes what AUC conceals.** At 95% specificity thresholds required for screening, ViT-Tiny models achieved sensitivity of only 0.03 (clinically useless), while ViT-B models maintained sensitivity of 0.94–0.97.

The study includes multiple-method evaluation (diagnostic validity, external generalisation, cross-cancer transfer, technical robustness, uncertainty calibration, disentanglement probing, confound control, and clinical utility with decision curve analysis), use of publicly available datasets (CAMELYON17, PCam, Kather2016, LC25000), and rigorous statistics (DeLong tests with Bonferroni correction, bootstrap CIs, power analysis). All code is publicly available at [GitHub URL].

We believe this work is particularly suited for the Digital Pathology and AI special issue because it provides actionable guidance for institutions deploying pathology AI: invest in the best available foundation model first, then consider architectural interventions only when feature quality is the binding constraint.

This manuscript has not been published previously and is not under consideration elsewhere. All authors have approved the submission. We have no conflicts of interest to declare.

We suggest the following potential reviewers with relevant expertise:
- [Reviewer suggestions to be added]

Thank you for your consideration. We look forward to your response.

Sincerely,

[Author 1], [Affiliation]
[Author 2], [Affiliation]
