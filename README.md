<p align="center">
  <img src="https://raw.githubusercontent.com/aethercore/aethercore/main/assets/logo.png" alt="AetherCore Logo" width="200"/>
</p>

<h1 align="center">AetherCore</h1>
<p align="center">
  <b>The Ultimate Zero-Config AI Engine. Beyond Transformers.</b><br>
  <i>Auto-Hardware Routing • Native SSM & MoE • Infinite Context</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/aethercore/"><img src="https://img.shields.io/pypi/v/aethercore?color=blue&label=pypi" alt="PyPI"></a>
  <a href="https://github.com/aethercore/aethercore/blob/main/LICENSE"><img src="https://img.shields.io/github/license/aethercore/aethercore?color=green" alt="License"></a>
  <a href="https://discord.gg/aethercore"><img src="https://img.shields.io/discord/123456789?color=7289da&label=discord" alt="Discord"></a>
</p>

---

## 🚀 Overview

**AetherCore** is a next-generation AI execution engine designed to obsolete the limitations of traditional Transformer-based frameworks. It provides a **zero-config, install-and-play** environment that automatically detects your hardware, dynamically quantizes models, and natively supports architectures beyond standard Attention mechanisms.

Whether you are running on a single MacBook, a multi-GPU cluster, or edge devices, AetherCore handles the heavy lifting so you can focus on building.

## ⚡ Core Features

*   **🧠 Beyond Transformers:** Native, highly-optimized support for **State Space Models (Mamba/S4)**, **Mixture of Experts (MoE)**, and **RWKV**. Break free from $O(n^2)$ attention bottlenecks.
*   **⚙️ Zero-Config Hardware Routing:** Automatically detects NVIDIA (CUDA), AMD (ROCm), Apple (MPS), and Intel (NPU). Selects the fastest custom kernels at runtime.
*   **🧠 Auto-Memory & Quantization:** Out of VRAM? AetherCore automatically applies 4-bit/8-bit quantization and seamlessly offloads layers to CPU/RAM without crashing or manual setup.
*   **🌐 Infinite Context:** Native support for linear-complexity architectures, allowing you to process millions of tokens without memory explosions.
*   **⚡ Custom Fused Kernels:** Written in Rust and Triton, our backend fuses operations to maximize memory bandwidth and minimize latency.

## 📦 Installation

Install AetherCore directly via pip. The core C++/Rust bindings are pre-compiled for your OS.

```bash
pip install aethercore
```

*For bleeding-edge features:*
```bash
pip install git+https://github.com/aethercore/aethercore.git
```

## 🛠️ Quick Start

Experience the power of AetherCore in just 3 lines of code. No config files, no manual device mapping.

```python
import aethercore as ac

# 1. Auto-detects hardware, auto-quantizes to fit VRAM, loads optimal architecture
model = ac.load("aethercore/mamba-7b-instruct")

# 2. Generate text with infinite context capabilities
response = model.generate("Explain quantum entanglement like I'm five.", max_tokens=512)
print(response)

# 3. Fine-tune with auto-distributed sharding (No DeepSpeed config needed!)
model.train(dataset="my_dataset.jsonl", mode="auto-lora")
```

## 🏗️ Architecture

AetherCore is built on a tri-layer architecture:
1.  **Python Frontend:** Intuitive, PyTorch-like API for rapid prototyping.
2.  **Graph Compiler (Rust):** Analyzes the computation graph and fuses operations dynamically.
3.  **Hardware Backend (C++/Triton):** Executes highly optimized, hardware-specific kernels (FlashAttention-3, Mamba-CUDA, MoE-Routing).

## 🗺️ Roadmap

- [x] Auto-Hardware Detection & Routing
- [x] Native Mamba/SSM Support
- [x] Auto-Quantization (4-bit/8-bit) & CPU Offloading
- [ ] Native Vision-Language (VLM) Auto-Routing
- [ ] Custom Triton Kernel Auto-Generator
- [ ] Distributed Training via native P2P GPU networking

## 🤝 Contributing

We welcome elite developers and researchers. Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs. 

## 📄 License

AetherCore is released under the [Apache 2.0 License](LICENSE). Commercial use is highly encouraged.

---
<p align="center">Built for the future of AI. Engineered by the AetherCore Team.</p>
