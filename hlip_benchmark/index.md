---
layout: null
---

<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    max-width: 960px;
    margin: 40px auto;
    padding: 0 16px;
    line-height: 1.6;
    font-size: 16px;
  }

  h1, h2, h3 {
    font-weight: 600;
    margin-top: 2em;
    margin-bottom: 0.6em;
  }

  h1 {
    font-size: 2rem;
  }

  h2 {
    font-size: 1.4rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem auto 1.25rem;
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
  }

  th, td {
    padding: 6px 4px;
    text-align: left;
    vertical-align: middle;
  }

  thead th {
    border-block: 1.5px solid #222;
    font-weight: 600;
  }

  tbody td {
    border-bottom: 1px solid #ddd;
  }

  tbody tr:last-child td {
    border-bottom: 1.5px solid #222;
  }

  @media (max-width: 800px) {
    table {
      display: block;
      overflow-x: auto;
    }
  }
</style>

# NeuroBench zero-shot classification

2026-11-20 · Chenhui Zhao · MLiNS @ University of Michigan

In [Neuro-VL](https://arxiv.org/abs/2611.15094), we assembled NeuroBench from 24 public datasets and used it primarily for visual question answering, report generation, and red-team evaluation. NeuroBench can also be used to evaluate zero-shot classification performance on neuroimaging studies. With Pub-Brain-5, an earlier benchmark for zero-shot classification of brain MRI studies, researchers must assemble the evaluation set from its original sources. With NeuroBench, researchers can download the images and classification labels directly from [Hugging Face](https://huggingface.co/datasets/mlinslab/neurobench) under the CC BY-NC-SA 4.0 license for noncommercial use. In this post, we report the full zero-shot classification results for [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [ConceptCLIP](https://huggingface.co/JerrryNie/ConceptCLIP), [ItemizedCLIP](https://github.com/MLNeurosurg/ItemizedCLIP), and [HLIP](https://huggingface.co/zch0414/clip-vit_large-slice_scan_study-dualdinotxt1568).

This evaluation covers **40,119 studies across 5 groups**.

- **MR-RATE** includes 5,550 MRI studies with 74 finding labels from NeuroVFM. Of these findings, 70 have at least 1 positive case and 38 have at least 32 positive cases. We use AUC.
- **CTRG-Brain-263K** includes 6,001 head CT studies with 82 finding labels from NeuroVFM. Of these findings, 45 have at least 1 positive case and 24 have at least 32 positive cases. We use AUC.
- **Open-Brain** includes 6,336 brain MRI studies from 17 sources. Each study has 1 of 6 pathology labels (stroke, glioma, metastasis, meningioma, schwannoma, or multiple sclerosis). **Open-Brain-Large** is a subset of 2,954 studies selected by lesion size. We use balanced accuracy for both cohorts.
- **RSNA** includes 21,744 head CT studies with 6 hemorrhage labels. We use AUC.
- **CQ500** includes 488 head CT studies with 11 finding labels. We use AUC.

The [HLIP ablation blog](/hlip_ablation/) raised concerns about the reliability of benchmarks assembled from multiple datasets for zero-shot classification. Scores can vary substantially even when only the training seed changes. In this post, we reproduce this variation on Pub-Brain-5 by training the same HLIP model (`clip_vit_base_scan_study_dualdinotxt1568`) on the same internal dataset with 3 different random seeds. We also evaluate the resulting 3 checkpoints on NeuroBench. We find that seed-to-seed variation is much smaller in all 5 NeuroBench groups than on Pub-Brain-5.

## Implementation details

All four models start from the same input. Each scan is stored as a `uint8` volume of shape `(48, 224, 224)` and processed with each model’s official data processor. BiomedCLIP and ConceptCLIP encode each slice independently, normalize its image embedding, and average these embeddings across all slices in the study to obtain a single study embedding. All four models use 18 prompt templates per class. For BiomedCLIP and ConceptCLIP, we explicitly specify “brain MRI” or “head CT” in each prompt to match the study modality.

## Main results

| Dataset | Metric | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | --- | ---: | ---: | ---: | ---: |
| MR-RATE | Macro AUC across 70 findings (≥1 positive study) | 51.71% | 56.65% | 81.45% | **85.90%** |
| MR-RATE | Macro AUC across 38 findings (≥32 positive studies) | 51.40% | 58.02% | 83.79% | **86.99%** |
| CTRG-Brain-263K | Macro AUC across 45 findings (≥1 positive study) | 58.25% | 67.87% | 79.63% | **85.03%** |
| CTRG-Brain-263K | Macro AUC across 24 findings (≥32 positive studies) | 61.26% | 70.96% | 83.89% | **90.00%** |
| Open-Brain | Balanced accuracy by source | 26.89% | 44.56% | **76.89%** | 75.21% |
| Open-Brain | Balanced accuracy by pathology | 38.02% | 46.46% | **66.05%** | 63.83% |
| Open-Brain-Large | Balanced accuracy by source | 31.25% | 54.99% | 79.60% | **80.28%** |
| Open-Brain-Large | Balanced accuracy by pathology | 44.02% | 55.65% | 70.35% | **72.07%** |
| CQ500 | Macro AUC across 11 findings | 64.41% | 73.93% | 89.02% | **93.99%** |
| RSNA | Macro AUC across 6 findings | 68.07% | 69.78% | 89.33% | **91.25%** |

The [Appendix](#appendix) reports results for each finding and source dataset.

## Training-seed robustness

| Dataset | Metric | Seed 0 | Seed 42 | Seed 40 | Δ (max − min) | Sample SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Pub-Brain-5** | **Balanced accuracy by pathology** | **63.88%** | **64.11%** | **58.26%** | **5.85%** | **3.31%** |
| MR-RATE | Macro AUC across 70 findings (≥1 positive study) | 82.52% | 82.26% | 82.24% | 0.28% | 0.16% |
| MR-RATE | Macro AUC across 38 findings (≥32 positive studies) | 83.95% | 84.36% | 84.19% | 0.41% | 0.21% |
| CTRG-Brain-263K | Macro AUC across 45 findings (≥1 positive study) | 81.99% | 84.41% | 84.57% | 2.58% | 1.45% |
| CTRG-Brain-263K | Macro AUC across 24 findings (≥32 positive studies) | 89.18% | 89.05% | 88.82% | 0.36% | 0.18% |
| Open-Brain | Balanced accuracy by source | 72.67% | 73.15% | 73.12% | 0.48% | 0.27% |
| Open-Brain | Balanced accuracy by pathology | 56.46% | 58.81% | 58.98% | 2.52% | 1.41% |
| Open-Brain-Large | Balanced accuracy by source | 75.83% | 77.77% | 78.23% | 2.40% | 1.27% |
| Open-Brain-Large | Balanced accuracy by pathology | 61.87% | 64.73% | 65.35% | 3.48% | 1.86% |
| CQ500 | Macro AUC across 11 findings | 92.70% | 91.95% | 93.00% | 1.05% | 0.54% |
| RSNA | Macro AUC across 6 findings | 89.56% | 88.93% | 88.64% | 0.92% | 0.47% |

Across 3 training seeds, balanced accuracy by pathology varied by 5.85 percentage points on Pub-Brain-5. On Open-Brain, its closest counterpart in NeuroBench, the spread was 2.52 percentage points on the full cohort and 3.48 on Open-Brain-Large. Across all evaluable findings, the spread in macro AUC was 0.28 percentage points on MR-RATE, 2.58 on CTRG-Brain-263K, 1.05 on CQ500, and 0.92 on RSNA. We hope this lower seed-to-seed variation across all 5 groups helps address researchers’ concerns about how reliably NeuroBench reflects a model’s zero-shot classification ability.

## Appendix

| MR-RATE | Positive studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Subdural hematoma | 39 | 42.01% | 39.53% | 93.40% | 95.00% |
| Epidural hematoma | 3 | 48.60% | 52.65% | 85.06% | 89.87% |
| Brain contusion | 9 | 47.34% | 42.77% | 52.30% | 71.67% |
| Diffuse axonal injury | 8 | 30.94% | 40.43% | 57.93% | 89.80% |
| Cerebral aneurysm | 26 | 61.52% | 46.97% | 64.68% | 71.69% |
| Arteriovenous malformation | 19 | 52.63% | 53.03% | 84.06% | 85.05% |
| Cavernous malformation cavernoma | 55 | 42.47% | 54.80% | 65.51% | 76.90% |
| Venous sinus thrombosis | 18 | 40.84% | 46.18% | 68.15% | 69.81% |
| Acute ischemic stroke | 106 | 44.75% | 44.88% | 80.23% | 86.14% |
| Chronic ischemic stroke | 756 | 48.87% | 55.42% | 82.17% | 80.36% |
| Moya moya disease | 0 | N/A | N/A | N/A | N/A |
| Small vessel ischemic disease | 1,593 | 48.42% | 56.78% | 81.61% | 79.90% |
| Lacunar stroke | 241 | 46.56% | 61.08% | 85.43% | 87.62% |
| Intracranial hemorrhage | 247 | 44.84% | 47.07% | 78.80% | 85.46% |
| Intraparenchymal hemorrhage | 199 | 64.55% | 63.13% | 87.85% | 86.73% |
| Intraventricular hemorrhage | 18 | 66.30% | 79.74% | 98.75% | 98.18% |
| Aneurysmal subarachnoid hemorrhage | 4 | 57.85% | 51.19% | 72.94% | 86.81% |
| Traumatic subarachnoid hemorrhage | 3 | 58.12% | 49.96% | 78.61% | 94.97% |
| Multiple sclerosis | 122 | 49.30% | 57.73% | 81.70% | 88.17% |
| Neuromyelitis optica | 15 | 57.63% | 61.94% | 54.33% | 83.00% |
| Neurosarcoidosis | 0 | N/A | N/A | N/A | N/A |
| Brain abscess | 3 | 73.35% | 68.54% | 85.95% | 87.77% |
| Viral encephalitis | 11 | 23.58% | 44.31% | 82.60% | 82.81% |
| Arachnoid cyst | 231 | 59.47% | 62.04% | 76.82% | 81.95% |
| Pineal cyst | 77 | 57.95% | 62.12% | 72.31% | 69.18% |
| Epidermoid cyst | 12 | 43.03% | 61.52% | 64.11% | 79.65% |
| Colloid cyst | 4 | 74.14% | 91.25% | 98.59% | 96.56% |
| Brain tumor | 595 | 53.30% | 53.96% | 80.38% | 87.92% |
| Orbital tumor | 25 | 44.19% | 39.93% | 72.10% | 84.24% |
| Head neck tumor | 135 | 43.19% | 51.39% | 65.97% | 78.85% |
| Spine tumor | 57 | 55.09% | 54.55% | 60.84% | 82.67% |
| Nerve tumor | 47 | 40.37% | 38.32% | 72.48% | 81.53% |
| Spinal degenerative changes | 144 | 43.88% | 35.41% | 56.35% | 58.86% |
| Glioma | 111 | 72.79% | 59.75% | 93.02% | 94.36% |
| High grade glioma | 72 | 79.06% | 78.94% | 97.94% | 97.28% |
| Low grade glioma | 30 | 42.35% | 44.85% | 80.28% | 83.46% |
| Brain metastasis | 207 | 47.08% | 65.76% | 90.05% | 96.22% |
| Meningioma | 115 | 64.52% | 49.11% | 80.42% | 84.70% |
| Schwannoma | 26 | 59.15% | 40.34% | 75.15% | 82.60% |
| Lymphoma | 9 | 45.84% | 49.62% | 87.65% | 89.29% |
| Pineal tumor | 12 | 49.98% | 66.41% | 82.20% | 73.68% |
| Intraventricular tumor | 59 | 74.67% | 77.35% | 80.78% | 82.81% |
| Pediatric posterior fossa tumor | 5 | 64.15% | 61.65% | 80.59% | 85.51% |
| Brainstem glioma | 14 | 55.82% | 54.77% | 90.79% | 92.33% |
| Pituitary tumor | 43 | 48.23% | 49.34% | 66.28% | 64.21% |
| Rathkes cleft cyst | 17 | 58.16% | 53.02% | 54.07% | 61.73% |
| Craniopharyngioma | 4 | 48.43% | 85.54% | 98.70% | 99.91% |
| Chordoma chondrosarcoma | 4 | 37.91% | 40.48% | 71.29% | 74.71% |
| Chiari malformation | 36 | 47.96% | 57.75% | 93.18% | 92.10% |
| Cerebral atrophy | 879 | 51.42% | 65.72% | 91.72% | 91.85% |
| Brain herniation | 34 | 35.64% | 42.53% | 90.18% | 95.24% |
| Brain mass effect | 263 | 43.96% | 56.06% | 89.50% | 92.28% |
| Brain midline shift | 50 | 67.25% | 85.60% | 99.01% | 99.07% |
| Cephaloceles | 3 | 69.77% | 58.39% | 86.23% | 95.88% |
| Encephalomalacia gliosis | 2,093 | 40.04% | 49.49% | 74.57% | 68.76% |
| Cerebral edema | 317 | 17.33% | 32.25% | 95.05% | 94.87% |
| Cavum septum pellucidum | 88 | 57.41% | 58.68% | 75.08% | 82.19% |
| Dandy walker malformation | 1 | 99.54% | 94.49% | 98.31% | 99.39% |
| Mega cisterna magna | 90 | 52.58% | 60.17% | 96.66% | 92.67% |
| Heterotopia | 4 | 41.17% | 56.80% | 52.99% | 59.57% |
| Pachygyria lissencephaly | 0 | N/A | N/A | N/A | N/A |
| Schizencephaly | 0 | N/A | N/A | N/A | N/A |
| Dysgenesis corpus callosum | 5 | 62.94% | 56.05% | 82.27% | 80.94% |
| Ventriculomegaly | 757 | 50.06% | 63.45% | 92.32% | 91.97% |
| Hydrocephalus ex vacuo | 599 | 55.04% | 77.80% | 93.57% | 91.23% |
| Obstructive hydrocephalus | 49 | 64.89% | 80.33% | 99.11% | 99.24% |
| Subdural hygroma effusions | 46 | 30.87% | 37.14% | 87.21% | 93.91% |
| Intracranial hypotension | 16 | 38.55% | 39.80% | 74.18% | 83.00% |
| Craniotomy craniectomy | 243 | 52.93% | 78.30% | 90.09% | 98.36% |
| Tumor resection cavity | 196 | 77.14% | 88.99% | 97.66% | 99.24% |
| Postsurgical changes | 297 | 37.22% | 52.09% | 88.86% | 95.90% |
| Pneumocephalus | 27 | 25.23% | 22.16% | 99.48% | 99.80% |
| Catheter | 17 | 46.86% | 51.99% | 94.03% | 92.89% |
| Spine syrinx | 2 | 40.81% | 54.18% | 89.04% | 80.60% |

| CTRG-Brain-263K | Positive studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acute ischemic stroke | 149 | 44.46% | 59.17% | 51.98% | 69.91% |
| Acute subdural hematoma | 139 | 70.83% | 87.98% | 93.22% | 98.03% |
| Airway obstruction | 0 | N/A | N/A | N/A | N/A |
| Aneurysm clip | 0 | N/A | N/A | N/A | N/A |
| Aneurysm coil | 0 | N/A | N/A | N/A | N/A |
| Aneurysmal subarachnoid hemorrhage | 0 | N/A | N/A | N/A | N/A |
| Arachnoid cyst | 19 | 64.74% | 76.60% | 82.55% | 94.89% |
| Arterial dissection | 0 | N/A | N/A | N/A | N/A |
| Basal ganglia calcification | 20 | 64.82% | 71.73% | 76.47% | 93.01% |
| Brain contusion | 178 | 70.77% | 86.50% | 93.63% | 97.26% |
| Brain herniation | 31 | 37.36% | 85.26% | 95.89% | 95.29% |
| Brain lead | 0 | N/A | N/A | N/A | N/A |
| Brain mass effect | 280 | 57.33% | 52.41% | 95.64% | 96.25% |
| Burr hole | 8 | 22.51% | 36.08% | 62.60% | 84.19% |
| Carotid cavernous fistula | 0 | N/A | N/A | N/A | N/A |
| Catheter | 11 | 70.26% | 98.12% | 84.78% | 98.76% |
| Cavernous malformation cavernoma | 0 | N/A | N/A | N/A | N/A |
| Cerebral abscess | 0 | N/A | N/A | N/A | N/A |
| Cerebral arteriovenous malformation | 0 | N/A | N/A | N/A | N/A |
| Cerebral atrophy | 3,313 | 62.43% | 74.88% | 79.97% | 83.24% |
| Cerebral edema | 269 | 37.13% | 44.01% | 86.78% | 90.34% |
| Cerebral foreign body | 0 | N/A | N/A | N/A | N/A |
| Cerebral subdural empyema | 0 | N/A | N/A | N/A | N/A |
| Cerebral venous sinus thrombosis | 0 | N/A | N/A | N/A | N/A |
| Chiari malformation | 0 | N/A | N/A | N/A | N/A |
| Cholesteatoma | 0 | N/A | N/A | N/A | N/A |
| Colloid cyst | 0 | N/A | N/A | N/A | N/A |
| Craniofacial injury | 109 | 57.35% | 75.09% | 77.90% | 90.80% |
| Cranioplasty implant | 0 | N/A | N/A | N/A | N/A |
| Craniotomy | 4 | 71.00% | 87.25% | 61.11% | 97.68% |
| Dandy walker malformation | 0 | N/A | N/A | N/A | N/A |
| Diffuse axonal injury | 0 | N/A | N/A | N/A | N/A |
| Displaced skull fracture | 11 | 67.56% | 90.73% | 95.87% | 88.69% |
| Dysgenesis corpus callosum | 0 | N/A | N/A | N/A | N/A |
| Embolization material | 0 | N/A | N/A | N/A | N/A |
| Encephalomalacia gliosis | 869 | 54.80% | 60.69% | 76.97% | 83.80% |
| Epidural hematoma | 70 | 75.10% | 90.11% | 89.28% | 96.92% |
| Extra axial brain tumor | 10 | 57.17% | 48.67% | 67.40% | 85.48% |
| Head neck enlarged lymph node | 1 | 41.36% | 25.35% | 87.03% | 3.50% |
| Head neck tumor | 2 | 35.10% | 34.24% | 50.38% | 37.49% |
| Intra axial brain tumor | 0 | N/A | N/A | N/A | N/A |
| Intracranial aneurysm | 1 | 27.36% | 1.28% | 35.05% | 46.14% |
| Intracranial atherosclerosis | 4 | 75.70% | 67.73% | 33.66% | 80.61% |
| Intracranial epidermoid dermoid cyst | 1 | 38.34% | 48.78% | 73.78% | 65.02% |
| Intracranial hemorrhage | 753 | 75.28% | 82.19% | 92.42% | 97.48% |
| Intracranial hypotension | 0 | N/A | N/A | N/A | N/A |
| Intracranial pressure monitor | 0 | N/A | N/A | N/A | N/A |
| Intraparenchymal hemorrhage | 468 | 82.21% | 87.12% | 94.03% | 97.02% |
| Intraventricular hemorrhage | 155 | 85.28% | 94.05% | 96.99% | 97.75% |
| Large vessel occlusion | 2 | 85.64% | 96.80% | 90.17% | 88.41% |
| Mastoid effusion | 52 | 41.27% | 37.76% | 69.73% | 87.45% |
| Mega cisterna magna | 24 | 49.18% | 57.60% | 96.98% | 98.31% |
| Midline shift | 199 | 85.88% | 88.83% | 97.12% | 97.15% |
| Nondisplaced skull fracture | 78 | 39.98% | 67.45% | 91.35% | 96.81% |
| Obstructive hydrocephalus | 0 | N/A | N/A | N/A | N/A |
| Orbital emphysema | 3 | 43.53% | 63.23% | 99.17% | 65.13% |
| Orbital trauma | 58 | 62.64% | 72.27% | 68.36% | 86.87% |
| Orbital tumor | 0 | N/A | N/A | N/A | N/A |
| Otitis media | 22 | 41.09% | 54.82% | 74.79% | 91.27% |
| Parathyroid nodule | 0 | N/A | N/A | N/A | N/A |
| Peritonsillar abscess | 0 | N/A | N/A | N/A | N/A |
| Pharyngeal abscess | 0 | N/A | N/A | N/A | N/A |
| Pineal tumor | 0 | N/A | N/A | N/A | N/A |
| Pituitary tumor | 1 | 91.78% | 100.00% | 97.98% | 99.95% |
| Pneumocephalus | 47 | 37.73% | 45.15% | 95.70% | 95.39% |
| Posterior fossa tumors | 0 | N/A | N/A | N/A | N/A |
| Postsurgical changes | 25 | 53.50% | 68.35% | 58.13% | 84.43% |
| Resection cavity | 0 | N/A | N/A | N/A | N/A |
| Scalp hemorrhage hematoma | 129 | 78.75% | 86.54% | 78.56% | 91.35% |
| Sinusitis | 748 | 48.73% | 50.19% | 68.64% | 77.60% |
| Skull base fracture | 55 | 48.45% | 79.39% | 95.79% | 97.35% |
| Skull bone tumor | 12 | 57.63% | 54.78% | 51.78% | 69.60% |
| Slit ventricle | 0 | N/A | N/A | N/A | N/A |
| Small vessel ischemic disease | 4,268 | 61.37% | 66.95% | 71.85% | 75.28% |
| Spinal degenerative changes | 0 | N/A | N/A | N/A | N/A |
| Spine tumor | 0 | N/A | N/A | N/A | N/A |
| Subacute chronic subdural hematoma | 24 | 55.64% | 83.62% | 94.15% | 98.58% |
| Subdural hygroma effusions | 44 | 61.23% | 68.57% | 89.55% | 94.20% |
| Thyroid nodule | 0 | N/A | N/A | N/A | N/A |
| Transependymal flow | 0 | N/A | N/A | N/A | N/A |
| Traumatic subarachnoid hemorrhage | 193 | 73.46% | 88.29% | 93.73% | 97.68% |
| Ventriculomegaly | 1,007 | 57.65% | 57.47% | 64.27% | 64.13% |

| Open-Brain (balanced accuracy by pathology) | Studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stroke | 247 | 40.49% | 100.00% | 51.42% | 80.16% |
| Glioma | 3,119 | 3.01% | 46.59% | 89.45% | 89.07% |
| Metastasis | 768 | 11.46% | 48.96% | 60.03% | 49.48% |
| Meningioma | 1,000 | 4.30% | 0.00% | 31.80% | 15.60% |
| Schwannoma | 1,057 | 68.87% | 21.85% | 84.96% | 85.24% |
| Multiple sclerosis | 145 | 100.00% | 61.38% | 78.62% | 63.45% |

| Open-Brain (balanced accuracy by source) | Studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| isles | 247 | 40.49% | 100.00% | 51.42% | 80.16% |
| brats_gli | 1,251 | 4.32% | 57.79% | 92.81% | 89.05% |
| brats_ped | 99 | 0.00% | 60.61% | 98.99% | 100.00% |
| brats_ssa | 60 | 33.33% | 96.67% | 96.67% | 95.00% |
| cfb_gbm | 191 | 0.52% | 30.37% | 88.48% | 83.25% |
| mu_glioma_post | 546 | 1.65% | 41.58% | 85.53% | 89.93% |
| remind | 171 | 0.00% | 22.22% | 70.76% | 83.63% |
| ucsd_ptgbm | 179 | 0.00% | 1.68% | 84.92% | 93.30% |
| utsw_glioma | 622 | 1.61% | 45.98% | 90.68% | 88.10% |
| brats_met | 238 | 0.84% | 0.84% | 52.52% | 39.08% |
| molab_mri_long | 373 | 16.09% | 97.05% | 63.27% | 55.23% |
| ucluk_mri_long | 157 | 16.56% | 7.64% | 63.69% | 51.59% |
| brats_men | 1,000 | 4.30% | 0.00% | 31.80% | 15.60% |
| ucsd_vs_longitudinal | 524 | 53.05% | 22.33% | 87.98% | 87.79% |
| vestibular_schwannoma_mc_rc2 | 533 | 84.43% | 21.39% | 81.99% | 82.74% |
| ms_les_seg | 115 | 100.00% | 51.30% | 75.65% | 57.39% |
| open_ms | 30 | 100.00% | 100.00% | 90.00% | 86.67% |

| Open-Brain-Large (balanced accuracy by pathology) | Studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stroke | 63 | 61.90% | 100.00% | 68.25% | 95.24% |
| Glioma | 1,585 | 5.17% | 68.77% | 95.46% | 93.19% |
| Metastasis | 348 | 14.94% | 54.31% | 48.56% | 45.69% |
| Meningioma | 500 | 8.40% | 0.00% | 27.40% | 12.20% |
| Schwannoma | 414 | 73.67% | 28.99% | 91.55% | 95.17% |
| Multiple sclerosis | 44 | 100.00% | 81.82% | 90.91% | 90.91% |

| Open-Brain-Large (balanced accuracy by source) | Studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| isles | 63 | 61.90% | 100.00% | 68.25% | 95.24% |
| brats_gli | 625 | 7.52% | 85.44% | 97.76% | 93.44% |
| brats_ped | 49 | 0.00% | 87.76% | 100.00% | 100.00% |
| brats_ssa | 30 | 50.00% | 100.00% | 100.00% | 93.33% |
| cfb_gbm | 95 | 1.05% | 52.63% | 93.68% | 87.37% |
| mu_glioma_post | 297 | 3.03% | 57.24% | 92.59% | 93.60% |
| remind | 85 | 0.00% | 30.59% | 77.65% | 94.12% |
| ucsd_ptgbm | 92 | 0.00% | 3.26% | 96.74% | 94.57% |
| utsw_glioma | 312 | 3.21% | 75.00% | 97.44% | 92.31% |
| brats_met | 119 | 1.68% | 1.68% | 40.34% | 31.09% |
| molab_mri_long | 186 | 19.89% | 98.39% | 56.99% | 54.30% |
| ucluk_mri_long | 43 | 30.23% | 9.30% | 34.88% | 48.84% |
| brats_men | 500 | 8.40% | 0.00% | 27.40% | 12.20% |
| ucsd_vs_longitudinal | 143 | 67.13% | 35.66% | 98.60% | 97.20% |
| vestibular_schwannoma_mc_rc2 | 271 | 77.12% | 25.46% | 87.82% | 94.10% |
| ms_les_seg | 29 | 100.00% | 72.41% | 89.66% | 89.66% |
| open_ms | 15 | 100.00% | 100.00% | 93.33% | 93.33% |

| CQ500 | Positive studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intracranial hemorrhage (ICH) | 205 | 65.35% | 72.98% | 89.60% | 94.19% |
| Intraparenchymal hemorrhage (IPH) | 134 | 71.56% | 75.39% | 88.56% | 93.99% |
| Intraventricular hemorrhage (IVH) | 28 | 77.88% | 92.99% | 95.77% | 99.10% |
| Subdural hemorrhage (SDH) | 53 | 57.17% | 73.95% | 85.27% | 89.26% |
| Extradural hemorrhage (EDH) | 13 | 58.81% | 79.21% | 91.09% | 95.96% |
| Subarachnoid hemorrhage (SAH) | 60 | 70.48% | 84.20% | 92.63% | 95.65% |
| Left-sided intracranial hemorrhage | 131 | 61.33% | 63.56% | 86.16% | 88.93% |
| Right-sided intracranial hemorrhage | 129 | 57.22% | 57.38% | 89.24% | 89.43% |
| Skull fracture | 39 | 52.17% | 66.42% | 67.85% | 92.89% |
| Mass effect | 127 | 56.60% | 65.12% | 95.04% | 96.58% |
| Midline shift | 65 | 79.97% | 81.99% | 98.02% | 97.90% |

| RSNA | Positive studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Any intracranial hemorrhage | 8,882 | 69.94% | 70.16% | 92.58% | 94.26% |
| Epidural hemorrhage | 354 | 62.50% | 64.71% | 81.44% | 83.30% |
| Intraparenchymal hemorrhage | 5,321 | 80.70% | 79.47% | 91.53% | 94.28% |
| Intraventricular hemorrhage | 3,692 | 76.75% | 82.13% | 94.65% | 95.67% |
| Subarachnoid hemorrhage | 3,932 | 62.74% | 62.98% | 88.56% | 90.26% |
| Subdural hemorrhage | 3,812 | 55.82% | 59.19% | 87.22% | 89.74% |
