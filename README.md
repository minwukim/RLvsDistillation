
# 🧠 RL vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning

This repository contains the code used in the paper:

> **Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning**  
> Minwu Kim*, Anubhav Shrestha*, Safal Shrestha, Aadim Nepal, Keith Ross  

📄 [arXiv:2505.14216](https://arxiv.org/abs/2505.14216) | 🤗 [Hugging Face](https://huggingface.co/papers/2505.14216)

---

## 🧩 About the Paper

This paper investigates *why* reinforcement learning with verifiable rewards (**RLVR**) and **distillation** have different impacts on large language model (LLM) reasoning. Key findings:
- RLVR improves **accuracy** but does **not** improve **capability**, due to selective reinforcement of easier questions.
- Distillation improves both **accuracy** and **capability**, but only when it introduces **new knowledge**.
- Distilling only reasoning patterns behaves similarly to RLVR, helping with easy questions but failing on hard ones.

Read the full paper 👉 [arXiv:2505.14216](https://arxiv.org/abs/2505.14216)

---

## 📁 Code Structure

```bash
.
├── environment.yml           # 🧪 Conda environment file
├── zero3.yaml               # ⚙️ DeepSpeed Zero3 config for RLVR training
│
├── train/
│   ├── RLVR/
│   │   ├── grpo_trainer.py     # RL training script using TRL (GRPO)
│   │   └── grpoconfig.yaml     # Config file for GRPO training
│   │
│   └── Distillation/
│       ├── sft_trainer.py      # Supervised fine-tuning script
│       └── sftconfig.yaml      # SFT config file
│
└── test/
    └── MATH500_eval.py        # 🎯 Evaluation script on MATH500
````

---

## ⚙️ Environment Setup

Create the conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate rlvsdistill
```

Ensure your system supports [DeepSpeed](https://www.deepspeed.ai/) and has `accelerate` configured.

---

## 🚀 How to Run

### 🔍 Evaluation (MATH500)

```bash
python test/MATH500_eval.py
```

---

### 🧪 RLVR Training (GRPO via TRL + VLLM)

We use **TRL** for policy optimization and **VLLM** for fast multi-process inference during reward evaluation.

```bash
accelerate launch \
  --config_file zero3.yaml \
  --num_processes <NUM_PROCESSES> \
  train/RLVR/grpo_trainer.py
```

💡 **Note:**

* `<NUM_PROCESSES>` should be set to the number of available CPU cores **minus one (–1)**.
  This is because **VLLM** internally uses one process for fast batched model inference, and the remaining processes will be used for parallel reward evaluation.
* Example: If your machine has 16 logical cores, use `--num_processes 15`.

🛠️ Edit `train/RLVR/grpoconfig.yaml` to configure:

* Base model and reward model paths
* Dataset locations
* Sampling parameters
* Reward functions

---

### 📘 Distillation Training (SFT)

```bash
python train/Distillation/sft_trainer.py --config train/Distillation/sftconfig.yaml
```

Use the config file to specify:

* Teacher and student model
* Number of training steps
* Prompt format and dataset
* Saving & logging behavior

---

## 📦 Libraries Used

* 🤗 **[Transformers](https://github.com/huggingface/transformers)** – model loading and generation
* 🤗 **[TRL](https://github.com/huggingface/trl)** – for RL fine-tuning via GRPO
* ⚡ **[VLLM](https://github.com/vllm-project/vllm)** – for fast batched generation and reward model inference
* 🧪 **DeepSpeed** – memory-efficient distributed training


---

## 🧪 Citation

If cite this paper in any form, please refer to the BibTeX citation shown below:

```bibtex
@misc{kim2024reinforcement,
      title={Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning}, 
      author={Minwu Kim and Anubhav Shrestha and Safal Shrestha and Aadim Nepal and Keith Ross},
      year={2024},
      eprint={2505.14216},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
