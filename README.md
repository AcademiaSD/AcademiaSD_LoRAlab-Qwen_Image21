
# AcademiaSD Qwen-Image 2.1 LoRAlab (Work in Progress)

> [!WARNING]
> **🚧 EVERYTHING IS PENDING / TODO ESTÁ PENDIENTE 🚧**
>
> This repository is under active development and **does not contain working code yet**. Every feature, script, number and step described below is a **planned design**, not a released implementation. VRAM and speed figures are **estimates** that have not been measured yet.
>
> Este repositorio está en desarrollo y **todavía no contiene código funcional**. Todas las funciones, scripts, cifras y pasos descritos abajo son un **diseño planificado**, no una implementación publicada. Las cifras de VRAM y velocidad son **estimaciones** aún sin medir.

<p align="center">
  <b>An ultra-fast, low-resource Web GUI & pipeline for training Qwen-Image 2.1 (NF4) LoRAs.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Work%20in%20Progress-red.svg" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-NVIDIA-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/UI-Flask%20%2B%20HTML5-purple.svg" alt="Web UI">
</p>

---

## 🗺️ Roadmap

| Stage | Status |
| :--- | :--- |
| Repository & README | ✅ Done |
| Pre-quantized NF4 checkpoint (DiT + Text Encoder) | ⏳ Pending |
| `1_pre_cache_qwen_image21.py` (Text embeddings + VAE latents) | ⏳ Pending |
| `2_train_lora_qwen_image21.py` (DiT NF4 LoRA training) | ⏳ Pending |
| LoRA export to ComfyUI key format | ⏳ Pending |
| Training previews | ⏳ Pending |
| Web GUI (Flask + HTML5), launchers & installer | ⏳ Pending |
| VRAM / speed benchmarks | ⏳ Pending |
| Edit / reference-image LoRAs | 🔮 Future (phase 2) |

---

## 🔬 Technical Design: How it will be fast, light & high quality

**Qwen-Image 2.1** is a ~7-Billion parameter single-stream Diffusion Transformer (32 blocks, 4096 wide) conditioned on **Qwen3-VL-8B** and paired with a new 64-channel VAE with 16x spatial compression. **AcademiaSD Qwen-Image 2.1 LoRAlab** will follow the same proven pipeline as [AcademiaSD Krea2 LoRAlab](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-Krea2) to make training possible on consumer GPUs.

---

### 1. 📉 Planned VRAM Budget (estimated)

| Memory Component | Standard Training | Qwen-Image 2.1 LoRAlab (target) |
| :--- | :--- | :--- |
| **DiT Model (7B)** | ~14.0 GB (BF16) | **~4.5 GB (4-bit NF4)** |
| **Text Encoder (Qwen3-VL-8B)** | ~17.0 GB | **0.0 GB (Offloaded via Pre-Cache)** |
| **VAE (Qwen-Image 2.1, 64ch)** | loaded | **0.0 GB (Offloaded via Pre-Cache)** |
| **Optimizer States (AdamW)** | several GB | **~0.2 GB (8-Bit AdamW on LoRA)** |
| **Activation Memory** | high | **~1.0 GB (Gradient Checkpointing)** |
| **Total VRAM Peak** | **30+ GB** | **~6–8 GB (to be measured)** |

* **4-Bit NormalFloat (NF4) Quantization (`bitsandbytes`)**: The DiT backbone and the Qwen3-VL-8B text encoder will be quantized to NF4 (`Linear4bit`). Precision-critical modules (the shared `modulation`, `txt_in`, `img_in`, `proj_out`, `norm_out` and the timestep embedder) stay in BF16.
* **Zero VRAM Wasted on Encoders (Offline Pre-Caching)**: Neither the text encoder nor the VAE will be loaded during training. Text embeddings and image latents are computed once and stored on disk.
* **8-Bit AdamW Optimizer** and **Gradient Checkpointing** to keep optimizer and activation memory small.

---

### 2. ⚡ Why Training Will Be Fast

* **No Per-Step Encoding Overhead**: With everything pre-cached, 100% of GPU compute during training goes to the DiT.
* **Short Sequences**: The 16x VAE with patch size 1 turns a 1024×1024 image into only **4096 image tokens**, and the DiT is smaller than Krea-2 (7B vs 12B).
* **Pinned RAM & Non-Blocking CUDA Transfers** for cached latents and embeddings.

---

### 3. 🎨 How Generation Quality Will Be Preserved

* **Exact Text Conditioning**: Embeddings will match what ComfyUI feeds the model at inference: Qwen-Image 2.1 chat template, system turn removed, and the last hidden layer **without** the final RMSNorm.
* **Exact Channel-Wise VAE Normalization**: 64-channel `latents_mean` / `latents_std` applied as `(z - mean) / std`.
* **Native Architecture Behaviour**: Block-causal attention and t = 0 modulation of the text prefix, exactly as in the reference implementation.
* **Resolution-Aware Noise Shift**: Dynamic timestep shift matching the model's scheduler.
* **Full-Layer Target Coverage**: LoRA adapters on all attention and MLP linears of the 32 DiT blocks.

---

## ✨ Planned Features

- **🌐 Modern Web GUI** (Flask) for pre-caching, dataset editing, training, checkpointing and export.
- **🚀 1-Click Auto Launch** with `Run_LoRAlab-Qwen_Image21.bat`.
- **🖼️ Training Previews**.
- **📊 Real-Time Hardware Telemetry**: RAM, VRAM and GPU temperature.
- **🔑 Hugging Face Token Support** (`HF_token.json`) with download progress.
- **🖼️ Dataset Inspector & Inline Caption Editor**, including batch Trigger Word injection.
- **⏱️ Exact Step Resume Checkpoints**.
- **📂 Automatic Project Folder Management**.
- **🚀 One-Click Export ("Send to Models")** to ComfyUI and other WebUIs.
- **🌐 Fully Bilingual (English / Español)**.

---

## 🖥️ System Requirements (tentative)

| Requirement | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10/11 | Windows 11 |
| **GPU** | NVIDIA GPU with **8 GB VRAM** (to be confirmed) | NVIDIA GPU with **12 GB–24 GB VRAM** |
| **Python** | Python 3.10+ (inside `venv`) | Python 3.10 / 3.11 |
| **CUDA Toolkit** | CUDA 12.1+ | CUDA 12.8+ |

Qwen-Image 2.1 requires `torch>=2.4`, `transformers>=5.17` and a recent Diffusers build, so this project will use its own virtual environment, separate from other LoRAlab projects.

---

## 📦 Installation

⏳ **Pending.** There is nothing to install yet. / **Pendiente.** Todavía no hay nada que instalar.

---

## ⚡ Usage Guide

⏳ **Pending.** The workflow will mirror Krea2 LoRAlab: **Pre-Cache → Train → Export to ComfyUI**.

---

## 📁 Planned Project Structure

```text
AcademiaSD_LoRAlab-Qwen_Image21/
├── assets/
├── 1_pre_cache_qwen_image21.py      # Text embedding & VAE latent pre-caching
├── 2_train_lora_qwen_image21.py     # DiT 7B NF4 LoRA training
├── server.py                        # Flask backend web server
├── trainer_ui.html                  # HTML5 / CSS3 / JS Web GUI
├── Install_LoRAlab-Qwen_Image21.bat # Installer
├── Run_LoRAlab-Qwen_Image21.bat     # Windows 1-click launcher
└── Update_LoRAlab-Qwen_Image21.bat  # Updater
```

---

## 💬 Community & Support

Join the **AcademiaSD** community to learn more about local image and video AI!

- ▶ **YouTube**: [youtube.com/@Academia_SD](https://www.youtube.com/@Academia_SD)
- 𝕏 **X (Twitter)**: [twitter.com/Academia_S_D](https://twitter.com/Academia_S_D)
- 💬 **Discord**: [discord.gg/Syuaduy678](https://discord.gg/Syuaduy678)
- ☕ **Ko-Fi**: [ko-fi.com/academiasd](https://ko-fi.com/academiasd)

---

## 📜 Credits & License

Developed with ❤️ by **AcademiaSD**. Built upon PyTorch, Diffusers, PEFT, Bitsandbytes, and Hugging Face Hub.

The Qwen-Image 2.1 model weights are distributed by Qwen under the **Qwen Research License**; check its terms before using or sharing trained LoRAs.
