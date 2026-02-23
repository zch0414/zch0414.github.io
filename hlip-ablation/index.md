---
layout: null
---

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

  hr {
    margin: 2rem 0;
  }

  img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
  }

  .figure-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 1rem 0;
  }

  .figure {
    flex: 1;
    text-align: center;
  }

  .figure-caption {
    margin-top: 0.4rem;
    font-size: 0.85rem;
    color: #666;
  }

  .figure-empty {
    visibility: hidden;
  }

  .paper-table-wrap {
    display: flex;
    justify-content: center;
    margin: 1rem 0 1.25rem;
  }

  .paper-table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    margin: 0 auto;
    font-size: 0.9rem;
  }

  .paper-table col.model-col {
    width: 18%;
  }

  .paper-table col.metric-col {
    width: calc(82% / 13);
  }

  .paper-table th,
  .paper-table td {
    padding: 6px 4px;
    text-align: center;
    border: none;
    vertical-align: middle;
  }

  .paper-table thead tr:first-child th {
    border-top: 1.5px solid #222;
    font-weight: 600;
    position: relative;
  }

  .paper-table thead tr:first-child th[colspan] {
    border-bottom: none;
  }

  .paper-table thead tr:first-child th.group-pub::after,
  .paper-table thead tr:first-child th.group-rsna::after,
  .paper-table thead tr:first-child th.group-prospective::after {
    content: "";
    position: absolute;
    bottom: 0;
    border-bottom: 1px solid #aaa;
  }

  .paper-table thead tr:first-child th.group-prospective::after {
    left: 4px;
    right: 10px;
  }

  .paper-table thead tr:first-child th.group-pub::after {
    left: 10px;
    right: 10px;
  }

  .paper-table thead tr:first-child th.group-rsna::after {
    left: 10px;
    right: 4px;
  }

  .paper-table thead tr:last-child th {
    border-bottom: 1.5px solid #222;
    font-weight: 600;
  }

  .paper-table thead th[rowspan] {
    border-bottom: 1.5px solid #222;
  }

  .paper-table tbody tr td {
    border-bottom: 1px solid #ddd;
  }

  .paper-table tbody tr:last-child td {
    border-bottom: 1.5px solid #222;
  }

  .citation-block {
    border: 1px solid #d0d7de;
    border-radius: 8px;
    background: #f6f8fa;
    padding: 14px 16px;
    margin: 0.75rem 0 1rem;
  }

  .citation-block p {
    margin: 0 0 0.75rem 0;
  }

  .citation-block pre {
    margin: 0.6rem 0;
    background: #fff;
    border: 1px solid #d8dee4;
    border-radius: 6px;
    padding: 10px 12px;
    overflow-x: auto;
  }
</style>

# HLIP Ablation
2025-11-17 (updated 2026-02-22) · Chenhui Zhao · MLiNS @ Univeristy of Michigan

In [HLIP](https://arxiv.org/abs/2505.21862), we present a language–image pre-training framework designed for uncurated 3D medical data that incorporates a hierarchical attention mechanism. HLIP achieves state-of-the-art results on both curated and uncurated 3D medical datasets spanning brain MRI, head CT, and chest CT. We attribute these gains to the effective modeling, careful implementation, and scalability. In this blog, building on HLIP's conclusions and implementation, we push scalability one step further for uncurated 3D medical data. To this end, **we conduct five ablation studies that appear not to improve performance yet are crucial for scalability and for advancing vision–language modeling, including visual instruction tuning.** This yields new HLIP models trained on the combined BrainMRI220K and HeadCT240K datasets. We further introduce a **simple yet effective adjustment to the language supervision, resulting in updated HLIP models.**

The code [![github repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/zch0414/hlip/tree/hlip-ablation) and model [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/collections/zch0414/hlip) presented in this blog have been published.


## Experiments
While HLIP uses the external Pub-Brain-5 dataset to ablate different model designs, this dataset contains only five classes (normal, stroke, glioma, meningioma, metastasis), which is not sufficiently comprehensive to assess model capacity. The same limitation applies to the external RSNA dataset. In the following experiments, **we instead evaluate on our prospective dataset, which contains 23K studies covering 74 diagnoses for brain MRI and approximately 15K studies covering 83 diagnoses for head CT.** Moreover, the linear-probe protocol can introduce additional bias during evaluation. Therefore, **we instead use a zero-shot evaluation protocol, averaging over multiple prompts for stability (similar to the implementation in [open-clip](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/zero_shot_metadata.py)).** Given the scale of our evaluation, even a 0.5 or 1.0 AUC gain would be considered significant. Although the evaluation set is not publicly available, we hope that the conclusions drawn from these comprehensive evaluations can facilitate future work for the community.

<div class="figure-row">
  <div class="figure">
    <img src="images/ct.png" alt="ct">
    <div class="figure-caption">reimplementation on HeadCT240K</div>
  </div>

  <div class="figure">
    <img src="images/mri.png" alt="mri">
    <div class="figure-caption">reimplementation on BrainMRI220K</div>
  </div>
</div>

We reimplement HLIP on the HeadCT240K and BrainMRI220K datasets, achieving **75.9** AUC on head CT and **81.1** AUC on brain MRI.


### Pooling strategy, Patch size, Sequence position embedding

<div class="figure-row">
  <div class="figure">
    <img src="images/1 mri pool (cls -> dino.txt).png" alt="cls token → dino.txt">
    <div class="figure-caption">cls token → dino.txt (solid)</div>
  </div>

  <div class="figure">
    <img src="images/2 mri patch size (8,16,16 -> 6,16,16).png" alt="(8,16,16) → (6,16,16)">
    <div class="figure-caption">patch size [8, 16, 16] → [6, 16, 16] (solid)</div>
  </div>

  <div class="figure">
    <img src="images/5 mri seqposemb (with -> without).png" alt="w/ vs w/o sequence pos emb">
    <div class="figure-caption">w/ sequence position emb → w/o (solid)</div>
  </div>
</div>

All three experiments are conducted on the BrainMRI220K dataset.
- **Pooling strategy:** Advancing vision–language modeling, such as visual instruction tuning, may require visual tokens extracted from a frozen vision encoder. However, because HLIP uses a CLS-token pooling strategy, the visual tokens in the final layer do not receive gradients during pre-training. One could instead use the visual tokens from the second-to-last layer, but this is undesirable because the final layer of HLIP performs study-level attention. Here, we instead ablate the pooling strategy proposed by [DINO.TXT](https://arxiv.org/abs/2412.16334v1), which concatenates the CLS token with the average-pooled visual token. Although this does not improve performance in our setting, we retain this design because it can benefit downstream tasks like segmentation and visual instruction tuning.
- **Patch size:** Smaller patch sizes have been widely shown to benefit many perception tasks. Here, we find that HLIP also benefits from using smaller patch sizes along the inter-slice (z) dimension.
- **Sequence position embedding:** We find that the sequence position embedding is not necessary, likely because HLIP first applies scan-level attention, which is sufficient for the model to distinguish between different scans. Moreover, removing the sequence positional embedding makes the overall architecture more compatible with advanced positional embedding strategies, such as rotary position embedding. Although we have not yet observed benefits in our experiments, we make this change to facilitate future exploration.


### Patch dropout, Number of scans per study

<div class="figure-row">
  <div class="figure">
    <img src="images/3 mri patch dropout (0.25 -> 0.50).png" alt="patch dropout 0.25 → 0.50">
    <div class="figure-caption">patch dropout 0.25 → 0.50 (solid)</div>
  </div>

  <div class="figure">
    <img src="images/4 mri patch dropout (0.50 -> 0.75).png" alt="patch dropout 0.50 → 0.75">
    <div class="figure-caption">patch dropout 0.50 → 0.75 (solid)</div>
  </div>

  <div class="figure">
    <img src="images/6 mri scans (10 -> 8).png" alt="10 scans → 8 scans">
    <div class="figure-caption">10 scans → 8 scans (solid)</div>
  </div>
</div>

All three experiments are conducted on the BrainMRI220K dataset.
- **Patch dropout:** HLIP already uses a 0.25 patch dropout rate, primarily for acceleration and regularization. Here, we further explore this factor to enable larger batch sizes under fixed computational resources. We find that a 0.5 patch dropout rate still offers a favorable precision-to–batch-size trade-off when batch size is 384, whereas a 0.75 rate does not when batch size is 512. Therefore, we adopt a 0.5 patch dropout rate in subsequent experiments.
- **# scans per study:** We also find that reducing the number of scans per study during training from 10 to 8 does not affect performance, while it accelerates the training process and alleviates memory consumption.


### Pushing scalability one step further

<div class="figure-row">
  <div class="figure">
    <img src="images/ct&mri vs ct.png" alt="ct&mri vs ct">
    <div class="figure-caption">ct&mri (green) vs ct only (yellow)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri vs mri.png" alt="ct&mri vs mri">
    <div class="figure-caption">ct&mri (green) vs mri only (blue)</div>
  </div>
</div>

Keeping all five subtle but meaningful changes, we train HLIP on the combined BrainMRI220K and HeadCT240K datasets. Using a batch size of 768 achieved through a gradient-accumulation step of 2, the training process takes approximately two days on eight L40 GPUs. With these changes, HLIP achieves **79.2** AUC on head CT and **80.6** AUC on brain MRI.


### Sentence dropout

<div class="figure-row">
  <div class="figure">
    <img src="images/ct&mri sentence dropout (ct).png" alt="sentence dropout ct">
    <div class="figure-caption">full report vs sentence dropout (solid)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri sentence dropout (mri).png" alt="sentence dropout mri">
    <div class="figure-caption">full report vs sentence dropout (solid)</div>
  </div>
</div>

Image captions used in the original CLIP are short, often fewer than 60 words, whereas radiology reports are substantially longer, even when using an LLM-generated summary or the impression section. Motivated by this mismatch, we randomly select a single sentence at each training step during language–image pre-training. We find that this simple change yields a significant improvement. With this change, HLIP achieves **88.9** AUC on both head CT and brain MRI. We hypothesize that the gain stems from the limited representational capacity of the language model and the distribution shift between training (long text) and zero-shot evaluation (short prompts).

### Daul contrastive loss

<div class="figure-row">
  <div class="figure">
    <img src="images/ct&mri dual (ct).png" alt="dual contrastive loss ct">
    <div class="figure-caption">sentence dropout vs dual contrastive loss (orange)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri dual (mri).png" alt="dual contrastive loss mri">
    <div class="figure-caption">sentence dropout vs dual contrastive loss (orange)</div>
  </div>
</div>

One limitation of sentence dropout is that it can destabilize training. In practice, to ensure stable optimization, we must use a smaller learning rate (from 6e-4 to 4e-4). This instability increases the risk when scaling to a ViT-Large model, which is typically more difficult to train than a ViT-Base model. To improve training stability and avoid unnecessary hyperparameter tuning when training ViT-Large, we introduce a dual contrastive loss, following the strategy proposed in [TIPS](https://arxiv.org/abs/2410.16512). Specifically, at each step, we introduce two CLS tokens in the ViT architecture and contrast them with a randomly selected sentence and the full report, respectively.We observe faster convergence for the sentence contrastive loss. Intuitively, the model learns global features from sentence supervision and dense features from full-report supervision.

## Models
Building on the incremental designs introduced so far, we train four HLIP variants: (1) ViT-Base with scan attention (block indices: 0, 1, 3, 4, 6, 7, 9, 10) and study attention (block indices: 2, 5, 8, 11); (2) ViT-Base with slice attention (block indices: 0, 3, 6, 9), scan attention (block indices: 1, 4, 7, 10), and study attention (block indices: 2, 5, 8, 11); (3) ViT-Large with scan attention (block indices: 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22) and study attention (block indices: 5, 11, 17, 23); and (4) ViT-Large with slice attention (block indices: 0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21), scan attention (block indices: 4, 10, 16, 22), and study attention (block indices: 5, 11, 17, 23).

All models are trained for 20 epochs on the combined BrainMRI220K and HeadCT240K dataset with an initial learning rate of 5e-4 and a batch size of 768, followed by an additional 5 epochs of unmasked fine-tuning.

<div class="paper-table-wrap">
  <table class="paper-table">
    <colgroup>
      <col class="model-col">
      <col class="metric-col"><col class="metric-col">
      <col class="metric-col"><col class="metric-col"><col class="metric-col"><col class="metric-col"><col class="metric-col">
      <col class="metric-col"><col class="metric-col"><col class="metric-col"><col class="metric-col"><col class="metric-col"><col class="metric-col">
    </colgroup>
    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th colspan="2" class="group-prospective">Prospective</th>
        <th colspan="5" class="group-pub">Pub-Brain-5 (Anomaly Detection)</th>
        <th colspan="6" class="group-rsna">RSNA (Full Set)</th>
      </tr>
      <tr>
        <th>CT</th>
        <th>MRI</th>
        <th>STR</th>
        <th>GLI</th>
        <th>MEN</th>
        <th>MET</th>
        <th><em>mean</em></th>
        <th>IPH</th>
        <th>IVH</th>
        <th>SAH</th>
        <th>SDH</th>
        <th>Any</th>
        <th><em>mean</em></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>in the paper</td>
        <td>75.9</td>
        <td>81.1</td>
        <td>91.5</td>
        <td>89.2</td>
        <td>79.2</td>
        <td>78.1</td>
        <td>84.5</td>
        <td>88.2</td>
        <td>91.4</td>
        <td>84.1</td>
        <td>83.4</td>
        <td>81.5</td>
        <td>85.7</td>
      </tr>
      <tr>
        <td>2025-10-08</td>
        <td>89.1</td>
        <td>89.1</td>
        <td>94.8</td>
        <td>94.8</td>
        <td>86.0</td>
        <td>86.2</td>
        <td>90.5</td>
        <td>93.5</td>
        <td>96.4</td>
        <td>90.2</td>
        <td>89.1</td>
        <td>90.8</td>
        <td>92.0</td>
      </tr>
      <tr>
        <td>ViT Base<br>(scan + study)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ViT Base<br>(slice + scan + study)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ViT Large<br>(scan + study)</td>
        <td>89.0</td>
        <td>89.7</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ViT Large<br>(slice + scan + study)</td>
        <td>89.6</td>
        <td>89.6</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>

We evaluate this new model on Pub-Brain-5’s anomaly detection task and on the full RSNA dataset, demonstrating superior performance compared with the HLIP model in the original paper. **Note that these experiments are conducted under the zero-shot setting.**

## Findings
At the end of this blog post, we share several interesting findings and unsuccessful attempts from our experiments. We hope these observations provide new insights for researchers interested in this line of research.

### Supervised by different LLM-summarized reports

<div class="figure-row">
  <div class="figure">
    <img src="images/ct report (gpt3.5turbo -> gpt4omini).png" alt="ct report 4o mini">
    <div class="figure-caption">gpt3.5turbo vs gpt4omini (solid)</div>
  </div>

  <div class="figure">
    <img src="images/ct report (gpt3.5turbo -> gpt4.1mini).png" alt="ct report 4.1 mini">
    <div class="figure-caption">gpt3.5turbo vs gpt4.1mini (solid)</div>
  </div>

  <div class="figure">
    <img src="images/mri report (gpt3.5turbo -> gpt4omini).png" alt="mri report 4o mini">
    <div class="figure-caption">gpt3.5turbo vs gpt4omini (solid)</div>
  </div>

  <div class="figure">
    <img src="images/mri report (gpt3.5turbo -> gpt4.1mini).png" alt="mri report 4.1 mini">
    <div class="figure-caption">gpt3.5turbo vs gpt4.1mini (solid)</div>
  </div>
</div>

Although GPT-4omini and GPT-4.1mini are more advanced models than GPT-3.5, we find that supervising on reports summarized by these two models can lead to a significant decrease in zero-shot performance.

<div class="figure-row">
  <div class="figure">
    <img src="images/ct&mri sentence dropout (gpt3.5turbo -> gpt4omini) (ct).png" alt="ct sentence dropout report 4o mini">
    <div class="figure-caption">gpt3.5turbo w/sentence dropout vs gpt4omini w/ sentence dropout (solid)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri sentence dropout (gpt3.5turbo -> gpt4omini) (mri).png" alt="mri sentence dropout report 4o mini">
    <div class="figure-caption">gpt3.5turbo w/sentence dropout vs gpt4omini w/ sentence dropout (solid)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri dual (gpt3.5turbo -> gpt4omini) (ct).png" alt="ct dual report 4o mini">
    <div class="figure-caption">gpt3.5turbo w/dual contrastive loss vs gpt4omini w/ dual contrastive loss (solid)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri dual (gpt3.5turbo -> gpt4omini) (mri).png" alt="mri dual report 4o mini">
    <div class="figure-caption">gpt3.5turbo w/dual contrastive loss vs gpt4omini w/ dual contrastive loss (solid)</div>
  </div>
</div>

We find that either sentence dropout or dual contrastive loss can largely alleviate this issue.


### Unsuccessful attempts

<div class="figure-row">
  <div class="figure">
    <img src="images/fail mri initialization (avg -> central).png" alt="initialization">
    <div class="figure-caption">patch embedding initialization average → central (solid)</div>
  </div>

  <div class="figure">
    <img src="images/fail mri patch size (8,16,16 -> 8,14,14).png" alt="smaller patch size">
    <div class="figure-caption">patch size [8, 16, 16] → [8, 14, 14] (solid)</div>
  </div>

  <div class="figure">
    <img src="images/fail ct&mri rope (ct).png" alt="rope ct">
    <div class="figure-caption">rotary position embedding (solid)</div>
  </div>

  <div class="figure">
    <img src="images/fail ct&mri rope (mri).png" alt="rope mri">
    <div class="figure-caption">rotary position embedding (solid)</div>
  </div>
</div>

We introduce four designs that we find do not provide benefits in our current setting.
- **Initialization of patch embedding layer:** While central-inflation initialization has been shown to perform better for video ViTs, we find that average-inflation initialization performs better in our setting.
- **Patch size:** We find that using smaller patch sizes along the intra-slice (x and y) dimension does not improve performance.
- **Axial rotary position embedding:** We implement an axial rotary position embedding following [V-JEPA 2](https://github.com/facebookresearch/vjepa2). However, we do not observe clear benefits.

## Citation

<div class="citation-block">
<p>If this blog or the HLIP work is useful in your research, please consider citing:</p>

```bibtex
@article{zhao2026towards,
  title={Towards Scalable Language-Image Pre-training for 3D Medical Imaging},
  author={Chenhui Zhao and Yiwei Lyu and Asadur Zaman Chowdury and Edward S Harake and Akhil Kondepudi and Akshay T Rao and Xinhai Hou and Honglak Lee and Todd C Hollon},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=WxHf4EcBWA}
}
```

```bibtex
@misc{zhao2026hlipablationblog,
  author = {Chenhui Zhao},
  title = {HLIP Ablation},
  year = {2026},
  url = {https://zch0414.github.io/hlip-ablation/},
  note = {Accessed: 2026-02-23}
}
```
</div>
