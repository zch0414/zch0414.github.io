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

## Main results

| Dataset | Metric | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | --- | ---: | ---: | ---: | ---: |
| MR-RATE | Macro AUC across 70 findings (≥1 positive study) | 53.82% | 57.27% | 81.45% | **85.90%** |
| MR-RATE | Macro AUC across 38 findings (≥32 positive studies) | 53.14% | 58.56% | 83.79% | **86.99%** |
| CTRG-Brain-263K | Macro AUC across 45 findings (≥1 positive study) | 58.45% | 66.37% | 79.63% | **85.03%** |
| CTRG-Brain-263K | Macro AUC across 24 findings (≥32 positive studies) | 61.67% | 69.65% | 83.89% | **90.00%** |
| Open-Brain | Balanced accuracy by source | 21.47% | 50.27% | **76.89%** | 75.21% |
| Open-Brain | Balanced accuracy by pathology | 21.02% | 33.10% | **66.05%** | 63.83% |
| Open-Brain-Large | Balanced accuracy by source | 28.01% | 55.07% | 79.60% | **80.28%** |
| Open-Brain-Large | Balanced accuracy by pathology | 24.64% | 39.91% | 70.35% | **72.07%** |
| CQ500 | Macro AUC across 11 findings | 66.07% | 72.07% | 89.02% | **93.99%** |
| RSNA | Macro AUC across 6 findings | 69.02% | 71.14% | 89.33% | **91.25%** |

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
| Subdural hematoma | 39 | 52.38% | 41.83% | 93.40% | 95.00% |
| Epidural hematoma | 3 | 73.47% | 15.68% | 85.06% | 89.87% |
| Brain contusion | 9 | 73.89% | 64.36% | 52.30% | 71.67% |
| Diffuse axonal injury | 8 | 36.87% | 65.20% | 57.93% | 89.80% |
| Cerebral aneurysm | 26 | 65.38% | 47.76% | 64.68% | 71.69% |
| Arteriovenous malformation | 19 | 46.38% | 42.98% | 84.06% | 85.05% |
| Cavernous malformation cavernoma | 55 | 42.48% | 51.87% | 65.51% | 76.90% |
| Venous sinus thrombosis | 18 | 45.87% | 48.46% | 68.15% | 69.81% |
| Acute ischemic stroke | 106 | 41.35% | 43.01% | 80.23% | 86.14% |
| Chronic ischemic stroke | 756 | 44.15% | 50.42% | 82.17% | 80.36% |
| Moya moya disease | 0 | N/A | N/A | N/A | N/A |
| Small vessel ischemic disease | 1,593 | 44.51% | 53.16% | 81.61% | 79.90% |
| Lacunar stroke | 241 | 48.70% | 64.60% | 85.43% | 87.62% |
| Intracranial hemorrhage | 247 | 54.75% | 50.21% | 78.80% | 85.46% |
| Intraparenchymal hemorrhage | 199 | 78.85% | 69.95% | 87.85% | 86.73% |
| Intraventricular hemorrhage | 18 | 77.79% | 82.81% | 98.75% | 98.18% |
| Aneurysmal subarachnoid hemorrhage | 4 | 65.23% | 61.06% | 72.94% | 86.81% |
| Traumatic subarachnoid hemorrhage | 3 | 65.48% | 73.98% | 78.61% | 94.97% |
| Multiple sclerosis | 122 | 49.73% | 60.13% | 81.70% | 88.17% |
| Neuromyelitis optica | 15 | 55.97% | 63.26% | 54.33% | 83.00% |
| Neurosarcoidosis | 0 | N/A | N/A | N/A | N/A |
| Brain abscess | 3 | 57.29% | 70.95% | 85.95% | 87.77% |
| Viral encephalitis | 11 | 24.27% | 46.52% | 82.60% | 82.81% |
| Arachnoid cyst | 231 | 58.59% | 60.56% | 76.82% | 81.95% |
| Pineal cyst | 77 | 57.48% | 64.05% | 72.31% | 69.18% |
| Epidermoid cyst | 12 | 47.68% | 59.59% | 64.11% | 79.65% |
| Colloid cyst | 4 | 70.84% | 60.00% | 98.59% | 96.56% |
| Brain tumor | 595 | 56.02% | 60.36% | 80.38% | 87.92% |
| Orbital tumor | 25 | 47.86% | 47.22% | 72.10% | 84.24% |
| Head neck tumor | 135 | 51.31% | 53.11% | 65.97% | 78.85% |
| Spine tumor | 57 | 68.25% | 71.98% | 60.84% | 82.67% |
| Nerve tumor | 47 | 42.62% | 50.31% | 72.48% | 81.53% |
| Spinal degenerative changes | 144 | 62.53% | 45.06% | 56.35% | 58.86% |
| Glioma | 111 | 70.30% | 48.95% | 93.02% | 94.36% |
| High grade glioma | 72 | 70.00% | 75.00% | 97.94% | 97.28% |
| Low grade glioma | 30 | 47.17% | 50.04% | 80.28% | 83.46% |
| Brain metastasis | 207 | 63.44% | 74.05% | 90.05% | 96.22% |
| Meningioma | 115 | 66.55% | 51.53% | 80.42% | 84.70% |
| Schwannoma | 26 | 67.55% | 37.16% | 75.15% | 82.60% |
| Lymphoma | 9 | 53.75% | 67.65% | 87.65% | 89.29% |
| Pineal tumor | 12 | 58.32% | 73.46% | 82.20% | 73.68% |
| Intraventricular tumor | 59 | 75.42% | 80.95% | 80.78% | 82.81% |
| Pediatric posterior fossa tumor | 5 | 64.72% | 68.32% | 80.59% | 85.51% |
| Brainstem glioma | 14 | 63.62% | 54.90% | 90.79% | 92.33% |
| Pituitary tumor | 43 | 46.87% | 48.05% | 66.28% | 64.21% |
| Rathkes cleft cyst | 17 | 56.83% | 52.24% | 54.07% | 61.73% |
| Craniopharyngioma | 4 | 49.17% | 65.13% | 98.70% | 99.91% |
| Chordoma chondrosarcoma | 4 | 35.32% | 42.91% | 71.29% | 74.71% |
| Chiari malformation | 36 | 48.42% | 55.39% | 93.18% | 92.10% |
| Cerebral atrophy | 879 | 50.12% | 59.81% | 91.72% | 91.85% |
| Brain herniation | 34 | 40.67% | 47.10% | 90.18% | 95.24% |
| Brain mass effect | 263 | 53.89% | 52.43% | 89.50% | 92.28% |
| Brain midline shift | 50 | 60.17% | 80.45% | 99.01% | 99.07% |
| Cephaloceles | 3 | 55.84% | 41.39% | 86.23% | 95.88% |
| Encephalomalacia gliosis | 2,093 | 38.78% | 50.01% | 74.57% | 68.76% |
| Cerebral edema | 317 | 22.58% | 29.09% | 95.05% | 94.87% |
| Cavum septum pellucidum | 88 | 54.93% | 52.05% | 75.08% | 82.19% |
| Dandy walker malformation | 1 | 99.20% | 98.58% | 98.31% | 99.39% |
| Mega cisterna magna | 90 | 46.99% | 52.21% | 96.66% | 92.67% |
| Heterotopia | 4 | 44.44% | 70.65% | 52.99% | 59.57% |
| Pachygyria lissencephaly | 0 | N/A | N/A | N/A | N/A |
| Schizencephaly | 0 | N/A | N/A | N/A | N/A |
| Dysgenesis corpus callosum | 5 | 44.50% | 53.08% | 82.27% | 80.94% |
| Ventriculomegaly | 757 | 50.03% | 58.34% | 92.32% | 91.97% |
| Hydrocephalus ex vacuo | 599 | 51.22% | 73.61% | 93.57% | 91.23% |
| Obstructive hydrocephalus | 49 | 59.62% | 81.41% | 99.11% | 99.24% |
| Subdural hygroma effusions | 46 | 29.16% | 37.22% | 87.21% | 93.91% |
| Intracranial hypotension | 16 | 46.93% | 50.93% | 74.18% | 83.00% |
| Craniotomy craniectomy | 243 | 56.57% | 75.06% | 90.09% | 98.36% |
| Tumor resection cavity | 196 | 70.43% | 90.32% | 97.66% | 99.24% |
| Postsurgical changes | 297 | 39.62% | 61.51% | 88.86% | 95.90% |
| Pneumocephalus | 27 | 38.93% | 23.14% | 99.48% | 99.80% |
| Catheter | 17 | 43.71% | 36.09% | 94.03% | 92.89% |
| Spine syrinx | 2 | 23.35% | 48.51% | 89.04% | 80.60% |

| CTRG-Brain-263K | Positive studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acute ischemic stroke | 149 | 42.13% | 57.28% | 51.98% | 69.91% |
| Acute subdural hematoma | 139 | 75.93% | 89.90% | 93.22% | 98.03% |
| Airway obstruction | 0 | N/A | N/A | N/A | N/A |
| Aneurysm clip | 0 | N/A | N/A | N/A | N/A |
| Aneurysm coil | 0 | N/A | N/A | N/A | N/A |
| Aneurysmal subarachnoid hemorrhage | 0 | N/A | N/A | N/A | N/A |
| Arachnoid cyst | 19 | 70.95% | 79.58% | 82.55% | 94.89% |
| Arterial dissection | 0 | N/A | N/A | N/A | N/A |
| Basal ganglia calcification | 20 | 65.94% | 69.49% | 76.47% | 93.01% |
| Brain contusion | 178 | 79.01% | 86.34% | 93.63% | 97.26% |
| Brain herniation | 31 | 53.64% | 82.08% | 95.89% | 95.29% |
| Brain lead | 0 | N/A | N/A | N/A | N/A |
| Brain mass effect | 280 | 66.18% | 14.16% | 95.64% | 96.25% |
| Burr hole | 8 | 23.99% | 26.94% | 62.60% | 84.19% |
| Carotid cavernous fistula | 0 | N/A | N/A | N/A | N/A |
| Catheter | 11 | 73.49% | 96.31% | 84.78% | 98.76% |
| Cavernous malformation cavernoma | 0 | N/A | N/A | N/A | N/A |
| Cerebral abscess | 0 | N/A | N/A | N/A | N/A |
| Cerebral arteriovenous malformation | 0 | N/A | N/A | N/A | N/A |
| Cerebral atrophy | 3,313 | 64.89% | 72.63% | 79.97% | 83.24% |
| Cerebral edema | 269 | 42.63% | 40.56% | 86.78% | 90.34% |
| Cerebral foreign body | 0 | N/A | N/A | N/A | N/A |
| Cerebral subdural empyema | 0 | N/A | N/A | N/A | N/A |
| Cerebral venous sinus thrombosis | 0 | N/A | N/A | N/A | N/A |
| Chiari malformation | 0 | N/A | N/A | N/A | N/A |
| Cholesteatoma | 0 | N/A | N/A | N/A | N/A |
| Colloid cyst | 0 | N/A | N/A | N/A | N/A |
| Craniofacial injury | 109 | 55.33% | 75.85% | 77.90% | 90.80% |
| Cranioplasty implant | 0 | N/A | N/A | N/A | N/A |
| Craniotomy | 4 | 72.72% | 85.16% | 61.11% | 97.68% |
| Dandy walker malformation | 0 | N/A | N/A | N/A | N/A |
| Diffuse axonal injury | 0 | N/A | N/A | N/A | N/A |
| Displaced skull fracture | 11 | 64.22% | 92.41% | 95.87% | 88.69% |
| Dysgenesis corpus callosum | 0 | N/A | N/A | N/A | N/A |
| Embolization material | 0 | N/A | N/A | N/A | N/A |
| Encephalomalacia gliosis | 869 | 52.89% | 62.67% | 76.97% | 83.80% |
| Epidural hematoma | 70 | 77.67% | 92.67% | 89.28% | 96.92% |
| Extra axial brain tumor | 10 | 53.36% | 45.86% | 67.40% | 85.48% |
| Head neck enlarged lymph node | 1 | 28.32% | 19.35% | 87.03% | 3.50% |
| Head neck tumor | 2 | 54.23% | 59.06% | 50.38% | 37.49% |
| Intra axial brain tumor | 0 | N/A | N/A | N/A | N/A |
| Intracranial aneurysm | 1 | 26.96% | 1.15% | 35.05% | 46.14% |
| Intracranial atherosclerosis | 4 | 62.20% | 52.36% | 33.66% | 80.61% |
| Intracranial epidermoid dermoid cyst | 1 | 35.82% | 44.01% | 73.78% | 65.02% |
| Intracranial hemorrhage | 753 | 75.81% | 74.40% | 92.42% | 97.48% |
| Intracranial hypotension | 0 | N/A | N/A | N/A | N/A |
| Intracranial pressure monitor | 0 | N/A | N/A | N/A | N/A |
| Intraparenchymal hemorrhage | 468 | 84.02% | 85.84% | 94.03% | 97.02% |
| Intraventricular hemorrhage | 155 | 81.62% | 92.66% | 96.99% | 97.75% |
| Large vessel occlusion | 2 | 75.30% | 96.85% | 90.17% | 88.41% |
| Mastoid effusion | 52 | 35.54% | 40.35% | 69.73% | 87.45% |
| Mega cisterna magna | 24 | 54.14% | 57.69% | 96.98% | 98.31% |
| Midline shift | 199 | 80.14% | 86.84% | 97.12% | 97.15% |
| Nondisplaced skull fracture | 78 | 38.44% | 71.02% | 91.35% | 96.81% |
| Obstructive hydrocephalus | 0 | N/A | N/A | N/A | N/A |
| Orbital emphysema | 3 | 39.42% | 62.06% | 99.17% | 65.13% |
| Orbital trauma | 58 | 59.93% | 75.31% | 68.36% | 86.87% |
| Orbital tumor | 0 | N/A | N/A | N/A | N/A |
| Otitis media | 22 | 38.98% | 42.67% | 74.79% | 91.27% |
| Parathyroid nodule | 0 | N/A | N/A | N/A | N/A |
| Peritonsillar abscess | 0 | N/A | N/A | N/A | N/A |
| Pharyngeal abscess | 0 | N/A | N/A | N/A | N/A |
| Pineal tumor | 0 | N/A | N/A | N/A | N/A |
| Pituitary tumor | 1 | 93.02% | 99.97% | 97.98% | 99.95% |
| Pneumocephalus | 47 | 42.65% | 50.49% | 95.70% | 95.39% |
| Posterior fossa tumors | 0 | N/A | N/A | N/A | N/A |
| Postsurgical changes | 25 | 47.38% | 59.84% | 58.13% | 84.43% |
| Resection cavity | 0 | N/A | N/A | N/A | N/A |
| Scalp hemorrhage hematoma | 129 | 80.69% | 85.93% | 78.56% | 91.35% |
| Sinusitis | 748 | 48.27% | 50.31% | 68.64% | 77.60% |
| Skull base fracture | 55 | 40.68% | 82.23% | 95.79% | 97.35% |
| Skull bone tumor | 12 | 60.48% | 55.20% | 51.78% | 69.60% |
| Slit ventricle | 0 | N/A | N/A | N/A | N/A |
| Small vessel ischemic disease | 4,268 | 58.22% | 68.41% | 71.85% | 75.28% |
| Spinal degenerative changes | 0 | N/A | N/A | N/A | N/A |
| Spine tumor | 0 | N/A | N/A | N/A | N/A |
| Subacute chronic subdural hematoma | 24 | 55.50% | 87.05% | 94.15% | 98.58% |
| Subdural hygroma effusions | 44 | 62.47% | 69.91% | 89.55% | 94.20% |
| Thyroid nodule | 0 | N/A | N/A | N/A | N/A |
| Transependymal flow | 0 | N/A | N/A | N/A | N/A |
| Traumatic subarachnoid hemorrhage | 193 | 76.79% | 88.47% | 93.73% | 97.68% |
| Ventriculomegaly | 1,007 | 58.11% | 57.46% | 64.27% | 64.13% |

| Open-Brain (balanced accuracy by pathology) | Studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stroke | 247 | 0.00% | 61.54% | 51.42% | 80.16% |
| Glioma | 3,119 | 19.59% | 93.59% | 89.45% | 89.07% |
| Metastasis | 768 | 0.00% | 0.00% | 60.03% | 49.48% |
| Meningioma | 1,000 | 7.20% | 0.00% | 31.80% | 15.60% |
| Schwannoma | 1,057 | 0.00% | 0.00% | 84.96% | 85.24% |
| Multiple sclerosis | 145 | 99.31% | 43.45% | 78.62% | 63.45% |

| Open-Brain (balanced accuracy by source) | Studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| isles | 247 | 0.00% | 61.54% | 51.42% | 80.16% |
| brats_gli | 1,251 | 26.86% | 98.56% | 92.81% | 89.05% |
| brats_ped | 99 | 6.06% | 96.97% | 98.99% | 100.00% |
| brats_ssa | 60 | 50.00% | 100.00% | 96.67% | 95.00% |
| cfb_gbm | 191 | 40.84% | 98.43% | 88.48% | 83.25% |
| mu_glioma_post | 546 | 14.84% | 94.14% | 85.53% | 89.93% |
| remind | 171 | 0.00% | 25.73% | 70.76% | 83.63% |
| ucsd_ptgbm | 179 | 10.06% | 96.65% | 84.92% | 93.30% |
| utsw_glioma | 622 | 9.97% | 98.23% | 90.68% | 88.10% |
| brats_met | 238 | 0.00% | 0.00% | 52.52% | 39.08% |
| molab_mri_long | 373 | 0.00% | 0.00% | 63.27% | 55.23% |
| ucluk_mri_long | 157 | 0.00% | 0.00% | 63.69% | 51.59% |
| brats_men | 1,000 | 7.20% | 0.00% | 31.80% | 15.60% |
| ucsd_vs_longitudinal | 524 | 0.00% | 0.00% | 87.98% | 87.79% |
| vestibular_schwannoma_mc_rc2 | 533 | 0.00% | 0.00% | 81.99% | 82.74% |
| ms_les_seg | 115 | 99.13% | 44.35% | 75.65% | 57.39% |
| open_ms | 30 | 100.00% | 40.00% | 90.00% | 86.67% |

| Open-Brain-Large (balanced accuracy by pathology) | Studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stroke | 63 | 0.00% | 77.78% | 68.25% | 95.24% |
| Glioma | 1,585 | 33.63% | 95.77% | 95.46% | 93.19% |
| Metastasis | 348 | 0.00% | 0.00% | 48.56% | 45.69% |
| Meningioma | 500 | 14.20% | 0.00% | 27.40% | 12.20% |
| Schwannoma | 414 | 0.00% | 0.00% | 91.55% | 95.17% |
| Multiple sclerosis | 44 | 100.00% | 65.91% | 90.91% | 90.91% |

| Open-Brain-Large (balanced accuracy by source) | Studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| isles | 63 | 0.00% | 77.78% | 68.25% | 95.24% |
| brats_gli | 625 | 47.04% | 100.00% | 97.76% | 93.44% |
| brats_ped | 49 | 12.24% | 100.00% | 100.00% | 100.00% |
| brats_ssa | 30 | 76.67% | 100.00% | 100.00% | 93.33% |
| cfb_gbm | 95 | 67.37% | 100.00% | 93.68% | 87.37% |
| mu_glioma_post | 297 | 23.91% | 97.98% | 92.59% | 93.60% |
| remind | 85 | 0.00% | 28.24% | 77.65% | 94.12% |
| ucsd_ptgbm | 92 | 15.22% | 100.00% | 96.74% | 94.57% |
| utsw_glioma | 312 | 19.55% | 100.00% | 97.44% | 92.31% |
| brats_met | 119 | 0.00% | 0.00% | 40.34% | 31.09% |
| molab_mri_long | 186 | 0.00% | 0.00% | 56.99% | 54.30% |
| ucluk_mri_long | 43 | 0.00% | 0.00% | 34.88% | 48.84% |
| brats_men | 500 | 14.20% | 0.00% | 27.40% | 12.20% |
| ucsd_vs_longitudinal | 143 | 0.00% | 0.00% | 98.60% | 97.20% |
| vestibular_schwannoma_mc_rc2 | 271 | 0.00% | 0.00% | 87.82% | 94.10% |
| ms_les_seg | 29 | 100.00% | 65.52% | 89.66% | 89.66% |
| open_ms | 15 | 100.00% | 66.67% | 93.33% | 93.33% |

| CQ500 | Positive studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intracranial hemorrhage (ICH) | 205 | 66.51% | 71.07% | 89.60% | 94.19% |
| Intraparenchymal hemorrhage (IPH) | 134 | 77.11% | 76.09% | 88.56% | 93.99% |
| Intraventricular hemorrhage (IVH) | 28 | 77.69% | 92.35% | 95.77% | 99.10% |
| Subdural hemorrhage (SDH) | 53 | 57.39% | 73.43% | 85.27% | 89.26% |
| Extradural hemorrhage (EDH) | 13 | 61.51% | 79.15% | 91.09% | 95.96% |
| Subarachnoid hemorrhage (SAH) | 60 | 71.38% | 83.94% | 92.63% | 95.65% |
| Left-sided intracranial hemorrhage | 131 | 63.81% | 64.76% | 86.16% | 88.93% |
| Right-sided intracranial hemorrhage | 129 | 59.74% | 59.43% | 89.24% | 89.43% |
| Skull fracture | 39 | 55.90% | 70.51% | 67.85% | 92.89% |
| Mass effect | 127 | 57.38% | 40.86% | 95.04% | 96.58% |
| Midline shift | 65 | 78.38% | 81.11% | 98.02% | 97.90% |

| RSNA | Positive studies | BiomedCLIP | ConceptCLIP | ItemizedCLIP | HLIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Any intracranial hemorrhage | 8,882 | 71.17% | 67.00% | 92.58% | 94.26% |
| Epidural hemorrhage | 354 | 65.20% | 70.20% | 81.44% | 83.30% |
| Intraparenchymal hemorrhage | 5,321 | 83.55% | 80.25% | 91.53% | 94.28% |
| Intraventricular hemorrhage | 3,692 | 74.51% | 82.26% | 94.65% | 95.67% |
| Subarachnoid hemorrhage | 3,932 | 64.46% | 65.93% | 88.56% | 90.26% |
| Subdural hemorrhage | 3,812 | 55.20% | 61.21% | 87.22% | 89.74% |
