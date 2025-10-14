# Impact of Hyperparameter Optimization on Lightweight Models for Real-Time Image Classification

> *“Why change the architecture when you can just tweak the knobs?”* — Someone smart (probably you after reading this)

![models-optimized](https://img.shields.io/badge/Optimized-Yes-21c55d)
![accuracy-boom](https://img.shields.io/badge/Accuracy%2B-Verified-orange)

Official repository for the paper:

**📄 “Analysis of Hyperparameter Optimization Effects on Lightweight Deep Models for Real-Time Image Classification”**  
🧪 *Accepted in* **Scientific Reports (Nature Portfolio)**  
🔬 *By:* [Vineet Kumar Rakesh](mailto:vineet@vecc.gov.in)*, [Soumya Mazumdar](mailto:reachme@soumyamazumdar.com), [Tapas Samanta](mailto:tsamanta@vecc.gov.in), [Hemendra Kumar Pandey](mailto:hemendra@vecc.gov.in), and [Amitabha Das](mailto:amitabhad.snsa@jadavpuruniversity.in)  
*Corresponding Author — Variable Energy Cyclotron Centre (VECC), DAE, Govt. of India*  

---

## 🚀 What’s Inside

This repository contains all code and configuration files used in our **Scientific Reports** study:

* ✅ Training & evaluation scripts for **7 lightweight CNN and transformer-based models**  
* ✅ Comprehensive ablation studies on **learning rate**, **batch size**, **optimizers**, and **augmentation**  
* ✅ Real-time deployment metrics: **latency**, **FPS**, **model size**, **GFLOPs**  
* ✅ Subset-based reproducible benchmark (90k ImageNet-1K images)  
* ✅ PyTorch 2.5.1 + CUDA 12.6 with Automatic Mixed Precision (AMP)

---

## 🧠 Summary of Findings

| Model             | Top-1 (%) | Top-5 (%) | Latency (ms) ↓ | FPS ↑ | Params (M) | FLOPs (G) |
|--------------------|-----------|-----------|----------------|-------|-------------|------------|
| ConvNeXt-Tiny      | 83.85     | 95.09     | 0.51 (B=32)    | 1964.99 | 28.57 | 4.46 |
| EfficientNetV2-S   | 88.50     | 97.15     | 0.31 (B=32)    | 3226.66 | 21.31 | 2.85 |
| MobileNetV3-Large  | 86.99     | 96.93     | **0.10 (B=32)** | **10034.10** | 4.18 | 0.21 |
| MobileViT v2 (S)   | 87.82     | 97.19     | 0.40 (B=32)    | 2516.01 | 4.88 | 1.41 |
| MobileViT v2 (XS)  | 87.36     | 96.80     | 0.33 (B=32)    | 3007.27 | 1.36 | 0.36 |
| RepVGG-A2          | 88.45     | 97.16     | 0.26 (B=16)    | 3862.14 | 28.21 | 5.69 |
| TinyViT-21M        | **90.94** | **97.74** | 0.59 (B=16)    | 1687.04 | 33.21 | 4.09 |

> 🧩 Hyperparameter tuning alone improved accuracy by **1.5–3.5%** without modifying architectures.  
> 🧠 Models like **MobileNetV3-L** and **RepVGG-A2** achieved **sub-5 ms latency** and **9,000+ FPS** on NVIDIA L40s GPUs.

---

## 📊 Key Insights

- Learning rate is the single most sensitive hyperparameter — optimal range ≈ 0.01–0.1.  
- Composite augmentations (RandAugment + Mixup + CutMix + Label Smoothing) yielded the largest performance gains.  
- Cosine LR scheduling with 5-epoch warm-up ensured stable convergence.  
- Mixed precision training provided up to **40% faster training** with negligible accuracy drop.  
- Subset-based ImageNet training (90k balanced samples) gave reproducible and scalable results.

---

### 🎨 Cumulative Augmentation Effects

**Table: Top-1 Validation Accuracy (%) of Representative Models with Cumulative Augmentation Strategies  
(Trained for 300 Epochs on ImageNet–1K Subset)**

| **Model** | **Baseline** | **+ RandAug** | **+ Mixup** | **+ CutMix** | **+ Label Smooth** |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| ConvNeXt-Tiny | 83.85 | 86.24 | 86.90 | **88.50** | 88.00 |
| EfficientNetV2-S | 88.50 | 91.34 | **92.72** | 92.63 | 92.56 |
| MobileNetV3-Large | 86.99 | 89.15 | **90.97** | 90.45 | 90.20 |
| MobileViT v2 (S) | 87.83 | 89.91 | 91.47 | **92.63** | 91.28 |
| MobileViT v2 (XS) | 87.36 | 88.88 | **90.56** | 90.18 | 90.31 |
| RepVGG–A2 | 88.45 | 89.61 | **91.54** | 91.48 | 91.43 |
| TinyViT–21M | 90.94 | 92.11 | 93.30 | 93.35 | **93.84** |

> 📈 Composite augmentation consistently enhanced validation accuracy across all architectures,  
> with **CutMix** and **Label Smoothing** yielding the strongest late-epoch gains.

---

## 🧰 Installation

```bash
git clone https://github.com/VineetKumarRakesh/lcnn-opt.git
cd lcnn-opt
conda env create -f env.yml
conda activate lcnn-opt
```

> 🧪 Tested with PyTorch 2.5.1 + CUDA 12.6 on NVIDIA L40s (48 GB), Python 3.10.18.

---

## 🏋️ Train a Model

```bash
python train.py --model repvgg_a2 --config configs/repvgg.yaml --amp


* Use --amp for mixed precision training
* All training logs are automatically saved to /logs/

---

## 🧪 Evaluate a Model

```bash
python eval.py --checkpoint outputs/repvgg_best.pt --data-path /path/to/imagenet-val
```

Want to recreate the full ablation madness?

```bash
python scripts/ablation_study.py --config configs/convnext.yaml
```

---

## 📁 Project Structure (because we’re organized)

```
configs/     # YAML configurations for all models
data/        # Dataset loaders & preprocessing
models/      # Model wrappers
scripts/     # Ablation, profiling, and visualization tools
logs/        # Training logs
plots/       # Curves and figures
outputs/     # Saved checkpoints
```

---

## 📸 Results & Visualizations

📈 All results are saved in the outputs folder.

* Plots: `/plots`  
* Training logs: `/logs`

> 💡 All graphs were created without harming any matplotlib instances.

---

## 🔐 Checkpoints Access Policy

We don’t ship checkpoints with the repo because, well, storage is expensive and email is free.

🧬 **To request checkpoints**, kindly:

* Submit a pull request (with a note)
* OR email [vineet@vecc.gov.in](mailto:vineet@vecc.gov.in)

Please include:

* Your full name  
* Institutional affiliation  
* Your favorite optimizer (optional but highly encouraged)

---

## 🙏 Acknowledgements

* 💻 NVIDIA L40s — the silent workhorse  
* 🔥 PyTorch, `timm`, and `decord`  
* ☕ Coffee — the original batch size booster

---

## 📜 Citation

If this repository saved you time, compute, or reviewer wrath, please consider citing:

```bibtex
@misc{rakesh2025impacthyperparameteroptimizationaccuracy,
      title={Impact of Hyperparameter Optimization on the Accuracy of Lightweight Deep Learning Models for Real-Time Image Classification}, 
      author={Vineet Kumar Rakesh and Soumya Mazumdar and Tapas Samanta and Sarbajit Pal and Amitabha Das},
      year={2025},
      eprint={2507.23315},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.23315}, 
}
```

---

## 🌐 External Links

* **arXiv**: [arXiv:2507.23315](https://doi.org/10.48550/arXiv.2507.23315)
* **DOI**: _[To be minted]_  
* **Project Page**: [https://github.com/VineetKumarRakesh/lcnn-opt](https://github.com/VineetKumarRakesh/lcnn-opt)

---

## 🤓 Final Words

> “Real-time is not just fast. It’s fast *with purpose*.”

Train smart, tune wisely, and may your Top-1 be ever rising. 🚀
