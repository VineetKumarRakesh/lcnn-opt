# Impact of Hyperparameter Optimization on Lightweight Models for Real-Time Image Classification

> *“Why change the architecture when you can just tweak the knobs?”* — Someone smart (probably you after reading this)

![models-optimized](https://img.shields.io/badge/Optimized-Yes-21c55d)
![accuracy-boom](https://img.shields.io/badge/Accuracy%2B-Verified-orange)

Welcome to the official codebase of our paper:

**📄 “Impact of Hyperparameter Optimization on the Accuracy of Lightweight Deep Learning Models for Real-Time Image Classification”**  
🧪 *Submitted for Publication at Secret*  
🔬 *By: [Vineet Kumar Rakesh](mailto:vineet@vecc.gov.in), [Soumya Mazumdar](mailto:reachme@soumyamazumdar.com), [Tapas Samanta](mailto:tsamanta@vecc.gov.in), [Sarbajit Pal](mailto:sarbajit@vecc.gov.in), [Amitabha Das](mailto:amitabhad.snsa@jadavpuruniversity.in)*

---

## 🚀 What’s Inside

This repository contains:

* ✅ Training & evaluation scripts for **7 lightweight SOTA models**
* ✅ Ablation studies on learning rate, batch size, optimizers, and augmentations
* ✅ Real-time deployment metrics: **FPS**, **Inference Time**, **Model Size**
* ✅ Reproducible pipelines using PyTorch + AMP
* ✅ All training logs, plots, and performance tables

---

## 📊 TL;DR: What We Found

| Model             | Baseline Top-1 | Optimized Top-1 | GPU Hours | Verdict                |
|------------------|----------------|-----------------|-----------|------------------------|
| ConvNeXt-T       | 78.5%          | 80.7%           | 93.4      | Heavy but hopeful      |
| EfficientNetV2-S | 80.2%          | 82.5%           | 46.3      | Fast and fabulous      |
| MobileNetV3-L    | 75.3%          | 77.8%           | 92.2      | Small, but mighty-ish  |
| MobileViT v2 (S) | 85.45%         | 89.45%          | 89.2      | Transformer with wings |
| MobileViT v2 (XS)| 81.51%         | 85.51%          | 92.6      | Tiny but precise       |
| RepVGG-A2        | 80.4%          | 86.6%           | 91.9      | VGG’s glow-up          |
| TinyViT-21M      | 83.6%          | 89.49%          | 46.0      | Tiny? More like beast  |

> 🧠 Trained on ImageNet-1K with cosine LR, batch size 512, and mixed-precision PyTorch 2.5.1 + CUDA 12.6.

---

### 📈 Accuracy Curves

**Accuracy vs Epochs**
<img src="https://drive.google.com/file/d/1hqVnfSbrh2ygK1CZv2sjE3Gvy3DflfSY" alt="Accuracy vs Epochs" width="600"/>

**Accuracy vs Learning Rate**  
<img src="https://drive.google.com/file/d/1ARZ1RV774dxigeVUpHH2OKlGnEGSVm3k" alt="Accuracy vs LR" width="600"/>

---

## 🧰 Installation

```bash
git clone https://github.com/VineetKumarRakesh/lcnn-opt.git
cd lcnn-opt
conda env create -f env.yml
conda activate lcnn-opt
```

> 🧪 Tested on PyTorch 2.5.1 + CUDA 12.6 with NVIDIA L40s (but works on lesser mortals too)

---

## 🏋️ Train a Model

```bash
python train.py --model repvgg_a2 --config configs/repvgg.yaml --amp
```

* Use `--amp` for mixed precision.  
* Use `--log` to save your training logs to `/outputs/`.

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
1. configs/              # YAML files for each model and experiment
2. plots/                # Outputs and graphs
3. logs/                 # Output logs of each model
4. data/                 # Data used for training and validation
5. models/               # Model loading wrappers
6. scripts/              # Ablation, plotting, main scripts
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
@article{rakesh2025hyperopt,
  title={Impact of Hyperparameter Optimization on the Accuracy of Lightweight Deep Learning Models for Real-Time Image Classification},
  author={Vineet Kumar Rakesh and others},
  journal={arXiv},
  year={2025},
  note={Published}
}
```

---

## 🌐 External Links

* **arXiv**: _[Coming Soon]_  
* **DOI**: _[To be minted]_  
* **Project Page**: [https://github.com/VineetKumarRakesh/lcnn-opt](https://github.com/VineetKumarRakesh/lcnn-opt)

---

## 🤓 Final Words

> “Real-time is not just fast. It’s fast *with purpose*.”

Train smart, tune wisely, and may your Top-1 be ever rising. 🚀
