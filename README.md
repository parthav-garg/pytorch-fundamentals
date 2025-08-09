# Sequence Models: From Classic RNNs to Modern Mamba

This repository is a hands-on exploration into the architecture of deep learning models for sequence data. It contains from-scratch PyTorch implementations of two key architectures: a foundational Recurrent Neural Network (RNN) and the modern Mamba model (a State Space Model).

The goal of this project is to deconstruct these architectures to understand their core mechanisms, strengths, and weaknesses, inspired by educational deep-dives like Andrej Karpathy's `nanogpt`.

---

## Implementations

This repository contains two primary, self-contained projects:

### 1. Mamba Language Model (`Mamba/s6.py`)

A from-scratch implementation of the core `MambaBlock` in PyTorch, assembled into a full language model. This project dives into the details of modern, hardware-aware State Space Models (SSMs).

**Key Features:**
*   **Selective Scan Mechanism:** A from-scratch implementation of the core `_ssm_scan` logic, which is Mamba's key innovation for handling long-range dependencies.
*   **State-Space Discretization:** The logic for discretizing the continuous state-space parameters (A, B) into their discrete counterparts (Ā, B̄) is implemented manually.
*   **Complete Model:** The `MambaBlock` is integrated into a full `MambaLanguageModel` capable of training on text data like Tiny Shakespeare.
*   **Performance Profiling:** The script includes a function to profile the model's performance using `torch.profiler`, identifying the computational cost of different components.

### 2. Foundational RNN (`RNN/rnn.py`)

A classic Recurrent Neural Network built from scratch in PyTorch to demonstrate the fundamental principles of sequence modeling.

**Key Features:**
*   **Core Recurrence:** The `forward` pass contains an explicit `for` loop over the sequence length, clearly illustrating the sequential, step-by-step nature of a traditional RNN.
*   **From-Scratch Layers:** The RNN cell logic, including the weight matrices (`Wx`, `Wh`) and hidden state updates, is built using basic PyTorch linear layers.
*   **Demonstrative:** This implementation serves as a clear educational baseline for understanding the challenges (like sequential processing bottlenecks) that architectures like Mamba aim to solve.

---

## Repository Structure
```
.
├── Mamba/
│ └── s6.py # Mamba Language Model implementation
├── RNN/
│ └── rnn.py # Foundational RNN implementation
├── nanogpt/ # Inspirations and related materials
├── savez/
│ └── *.pth # Directory for saved model weights
├── .gitignore
├── input.txt # Sample text data (e.g., Tiny Shakespeare) for language models
└── README.md # This file
```