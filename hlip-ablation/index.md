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

  table, th, td {
    border: 1px solid #ccc;
    border-collapse: collapse;
  }

  th, td {
    padding: 4px 8px;
  }
</style>

# HLIP Ablation
2025-11-17 · Chenhui Zhao · MLiNS @ Univeristy of Michigan

In [HLIP](https://arxiv.org/abs/2505.21862), we present a language–image pre-training framework designed for uncurated 3D medical data that incorporates a hierarchical attention mechanism. HLIP achieves state-of-the-art results on both curated and uncurated 3D medical datasets spanning brain MRI, head CT, and chest CT. We attribute these gains to the effective modeling, careful implementation, and scalability. In this blog, building on HLIP’s conclusions and implementation, we push scalability one step further for uncurated 3D medical data. To this end, **we conduct five ablation studies that appear not to improve performance yet are crucial for scalability and for advancing vision–language modeling, including visual instruction tuning.** This yields a new HLIP model trained on the combined BrainMRI220K and HeadCT240K datasets. We further introduce a **simple yet effective adjustment to the language supervision, resulting in an updated HLIP model.**

The code [![github repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/Zch0414/hlip/tree/hlip-ablation) and model [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/Zch0414/hlip-2025-10-08) presented in this blog have been published.


## Experimental setup
While HLIP uses the external Pub-Brain-5 dataset to ablate different model designs, this dataset contains only five classes (normal, stroke, glioma, meningioma, metastasis), which is not sufficiently comprehensive to assess model capacity. The same limitation applies to the external RSNA dataset. In the following experiments, **we instead evaluate on our prospective dataset, which contains 23K studies covering 74 diagnoses for brain MRI and approximately 15K studies covering 83 diagnoses for head CT.** Moreover, the linear-probe protocol can introduce additional bias during evaluation. Therefore, **we instead use a zero-shot evaluation protocol, averaging over multiple prompts for stability (similar to the implementation in [open-clip](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/zero_shot_metadata.py)).** Although the evaluation set is not publicly available, we hope that the conclusions drawn from these comprehensive evaluations can facilitate future work for the community.

<div class="figure-row">
  <div class="figure">
    <img src="images/ct.png" alt="ct">
    <div class="figure-caption">reimplementation on HeadCT240K</div>
  </div>

  <div class="figure">
    <img src="images/mri.png" alt="mri">
    <div class="figure-caption">reimplementation on BrainMRI220K</div>
  </div>

  <div class="figure figure-empty">
    <img src="images/ct.png" alt="">
    <div class="figure-caption">&nbsp;</div>
  </div>
</div>

We first reimplement the HLIP model on the HeadCT240K and BrainMRI220K datasets, respectively.


## Pooling strategy, Patch size, Sequence position embedding

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
- Advancing vision–language modeling, such as visual instruction tuning, may require visual tokens extracted from a frozen vision encoder. However, because HLIP uses a CLS-token pooling strategy, the visual tokens in the final layer do not receive gradients during pre-training. One could instead use the visual tokens from the second-to-last layer, but this is undesirable because the final layer of HLIP performs study-level attention. Here, we instead ablate the pooling strategy proposed by [DINO.TXT](https://arxiv.org/abs/2412.16334v1), which concatenates the CLS token with the average-pooled visual token. Although this does not improve performance in our setting, we retain this design because it can benefit downstream tasks like segmentation and visual instruction tuning.
- Smaller patch sizes have been widely shown to benefit many perception tasks. Here, we find that HLIP also benefits from smaller patch sizes.
- We find that the sequence position embedding is not necessary, likely because HLIP first applies scan-level attention, which is sufficient for the model to distinguish between different scans. Moreover, removing the sequence position embedding also makes the overall architecture more compatible with advanced positional embedding strategies, such as rotary position embedding, which we discuss later.


## Patch dropout, Number of scans per study

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
- HLIP already uses a 0.25 patch dropout rate, primarily for acceleration and regularization. Here, we further explore this factor to enable larger batch sizes under fixed computational resources. We find that a 0.5 patch dropout rate still offers a favorable precision–batch-size trade-off when batch size is 384, whereas a 0.75 rate does not when batch size is 512. Therefore, we adopt a 0.5 patch dropout rate in subsequent experiments.
- We also find that reducing the number of scans per study during training from ten to eight does not affect performance, while it accelerates the training process and alleviates memory consumption.


## Pushing scalability one step further

<div class="figure-row">
  <div class="figure">
    <img src="images/ct&mri vs ct.png" alt="ct&mri vs ct">
    <div class="figure-caption">ct&mri (green) vs ct only</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri vs mri.png" alt="ct&mri vs mri">
    <div class="figure-caption">ct&mri (green) vs mri only</div>
  </div>

  <div class="figure figure-empty">
    <img src="images/ct" alt="">
    <div class="figure-caption">&nbsp;</div>
  </div>
</div>

Keeping all five subtle but meaningful changes, we train HLIP on the combined BrainMRI220K and HeadCT240K datasets. Using a batch size of 768 achieved through a gradient-accumulation step of 2, the training process takes approximately two days on eight L40 GPUs. Combining these two datasets yields a significant advantage for head CT.


## Sentence dropout

<div class="figure-row">
  <div class="figure">
    <img src="images/ct&mri sentence dropout (ct).png" alt="sentence dropout ct">
    <div class="figure-caption">sentence dropout (solid)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri sentence dropout (mri).png" alt="sentence dropout mri">
    <div class="figure-caption">sentence dropout (solid)</div>
  </div>

  <div class="figure figure-empty">
    <img src="images/ct" alt="">
    <div class="figure-caption">&nbsp;</div>
  </div>
</div>

Image captions used in the original CLIP are very short, whereas radiology reports are much longer, even when using an LLM-generated summary or the impression section. Intuitively, we randomly select a single sentence at each training step during language–image pre-training. We find that this simple adjustment yields a significant improvement. We hypothesize that this improvement arises from the limited representational capacity of the language model and from the distribution shift between training (long text) and zero-shot evaluation (short prompt).


## Umasked fine-tuning

<div class="figure-row">
  <div class="figure">
    <img src="images/ct&mri unmasked finetune (ct).png" alt="unmasked finetune ct">
    <div class="figure-caption">unmasked finetune (deep green)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri unmasked finetune (mri).png" alt="unmasked finetune mri">
    <div class="figure-caption">unmasked finetune (deep green)</div>
  </div>

  <div class="figure figure-empty">
    <img src="images/ct" alt="">
    <div class="figure-caption">&nbsp;</div>
  </div>
</div>

We further perform unmasked fine-tuning, maintaining the same batch size of 768 by increasing the gradient-accumulation steps to 6. Unmasked fine-tuning further improves performance. This yields our updated HLIP model.


## External evaluation

**Pub-Brain-5 (Anomaly Detection)**

|  | Stroke | Glioma | Meningioma | Metastasis | Mean |
|:---------------------------------:|:--------:|:--------:|:------------:|:------------:|:----------:|
| **HLIP**                        | 91.5   | 89.2   | 79.2       | 78.1       | 84.5     |
| **HLIP-2025-10-08**             | 94.8   | 94.8   | 86.0       | 86.2       | **90.5**     |

**RSNA (Full Set)**

|      | Intraparenchymal | Intraventricular  | Subarachnoid | Subdural | Any  | Mean |
|:---------------------:|:------------------:|:-------------------:|:--------------:|:----------:|:------:|:----------:|
| **HLIP**            | 88.2             | 91.4              | 84.1         | 83.4     | 81.5 | 85.7     |
| **HLIP-2025-10-08** | 93.5             | 96.4              | 90.2         | 89.1     | 90.8 | **92.0**     |

We evaluate this new model on Pub-Brain-5’s anomaly detection task and on the full RSNA dataset, demonstrating superior performance compared with the HLIP model in the original paper. **Note that these experiments are conducted under the zero-shot setting.**


## Supervised by LLM-summarized report

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

Here, we report a phenomenon observed when supervising with LLM-summarized reports. Although GPT-4omini and GPT-4.1mini are more advanced models than GPT-3.5, we find that supervising on reports summarized by these two models can lead to a significant decrease in zero-shot performance.

<div class="figure-row">
  <div class="figure">
    <img src="images/ct&mri sentence dropout report (gpt3.5turbo -> gpt4omini) (ct).png" alt="ct sentence dropout report 4o mini">
    <div class="figure-caption">gpt4omini w/ sentence dropout (solid)</div>
  </div>

  <div class="figure">
    <img src="images/ct&mri sentence dropout report (gpt3.5turbo -> gpt4omini) (mri).png" alt="mri sentence dropout report 4o mini">
    <div class="figure-caption">gpt4omini w/ sentence dropout (solid)</div>
  </div>

  <div class="figure figure-empty">
    <img src="images/ct" alt="">
    <div class="figure-caption">&nbsp;</div>
  </div>
</div>

We find that with sentence dropout, this issue can be largely alleviated.


## Unsuccessful attempts

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

At the end of this blog, we introduce four designs that we find do not provide benefits in our setting.
- For the patch embedding layer, while central-inflation initialization has been shown to perform better for video ViTs, we find that average-inflation initialization performs better in our setting.
- We find that using smaller patch sizes along the x and y axes does not improve performance.
- We implement a rotary position embedding following [V-JEPA 2](https://github.com/facebookresearch/vjepa2). However, we do not observe clear benefits. We hypothesize that rotary position embeddings may be more beneficial for larger models.