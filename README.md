<div align="center">

# ENIGMA

### **E**ncrypted **N**eural **I**nference on **G**raphs for **M**olecular **A**nalysis

*A secure, Kaggle-style GNN competition for molecular property prediction*

[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-blue)](leaderboard.md)
[![Dataset](https://img.shields.io/badge/Dataset-OGB_MolBACE-green)](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-Automated_Evaluation-orange)](.github/workflows/evaluate.yml)
[![Encryption](https://img.shields.io/badge/Submissions-RSA--2048_Encrypted-red)](encryption/)

</div>

---

## Overview

**ENIGMA** is a competitive benchmarking platform for Graph Neural Networks on molecular graph classification. Participants build GNN models to predict BACE-1 enzyme inhibition — a target relevant to Alzheimer's disease drug discovery — while all submissions are protected by RSA-2048 encryption.

### The Task

Given a molecular graph $G = (V, E)$ where:
- **Nodes** $V$ represent atoms with features $\mathbf{x}_v \in \mathbb{R}^d$ encoding atomic properties
- **Edges** $E$ represent chemical bonds with features encoding bond types

Learn a graph-level representation and predict a **binary label** $y \in \{0, 1\}$ indicating whether the molecule is an active inhibitor of BACE-1 (Beta-secretase 1).

### Prize

> **Top performers** will be invited to join a high-level research project aiming for publication at **NeurIPS 2026**.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Graph Specification](#graph-specification-adjacency-matrix-a--node-features-x)
3. [Evaluation Metric](#evaluation-metric)
4. [Security Architecture](#security-architecture)
5. [Getting Started](#getting-started)
6. [Submission Process](#submission-process)
7. [Baseline Architectures](#baseline-gnn-architectures)
8. [Advanced Architectures](#advanced-gnn-architectures)
9. [Evaluation Dimensions](#evaluation-dimensions)
10. [Repository Structure](#repository-structure)
11. [Rules](#rules)
12. [References](#references-and-citations)

---

## Dataset

We use the **OGB MolBACE** dataset from the [Open Graph Benchmark](https://ogb.stanford.edu/):

| Property | Value |
|----------|-------|
| **Source** | [OGB MolBACE](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) |
| **Task** | Binary classification (BACE-1 inhibitor: yes/no) |
| **Split** | Scaffold-based (prevents structural leakage) |
| **Class balance** | ~30% positive (imbalanced) |

| Split | Molecules | Labels | Description |
|-------|-----------|--------|-------------|
| Train | 1,210 | ✅ Provided | Model training |
| Valid | 151 | ✅ Provided | Hyperparameter tuning |
| Test | 152 | 🔒 Hidden | Final evaluation (CI-only) |

### Molecular Features

Each molecule is a graph with:
- **Node features**: 9-dimensional vectors $\mathbf{x}_v \in \mathbb{R}^9$ — atomic number, chirality, degree, formal charge, hydrogen count, hybridization, aromaticity, ring membership
- **Edge features**: 3-dimensional vectors — bond type, stereochemistry, conjugation

### Scaffold Split

The **scaffold split** groups molecules by Bemis-Murcko scaffolds, ensuring structurally different molecules in train/test. This simulates real-world drug discovery where novel molecular scaffolds must be classified.

### Class Imbalance

With ~30% positive class, a naive all-zeros classifier achieves ~70% accuracy but poor F1. Participants are encouraged to use class weighting, focal loss, or oversampling.

---

## Graph Specification (Adjacency Matrix A & Node Features X)

Every molecular graph is pre-computed and stored as dense NumPy matrices in `data/graphs/`:

| File | Molecules | Contents |
|------|-----------|----------|
| `data/graphs/train_graphs.npz` | 1,210 | $A$, $X$, $y$ |
| `data/graphs/valid_graphs.npz` | 151 | $A$, $X$, $y$ |
| `data/graphs/test_graphs.npz` | 152 | $A$, $X$ only (labels hidden) |

For each molecule $i$:

$$A_i \in \{0,1\}^{n_i \times n_i}, \quad X_i \in \mathbb{R}^{n_i \times 9}$$

where $n_i$ = number of atoms in molecule $i$.

<details>
<summary><strong>Loading example (Python)</strong></summary>

```python
import numpy as np

data = np.load('data/graphs/train_graphs.npz', allow_pickle=False)
indices = data['indices']   # molecule IDs

A = data['adj_2']   # (n, n) binary adjacency matrix
X = data['x_2']     # (n, 9) node feature matrix
y = data['y_2']     # label: 0 or 1

print(f"Molecule 2: {A.shape[0]} atoms, label = {y[0]}")
```

</details>

The same data is available via the OGB API (`data.edge_index`, `data.x`). See `data/graphs/README_graphs.md` for the full format specification.

---

## Evaluation Metric

**Primary metric: Macro F1 Score** — equally weights performance on both classes:

$$F1_{\text{macro}} = \frac{1}{2}\left(F1_{\text{class}_0} + F1_{\text{class}_1}\right)$$

where $F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$, $P_c = \frac{TP_c}{TP_c + FP_c}$, $R_c = \frac{TP_c}{TP_c + FN_c}$

**Why Macro F1?** — Treats both classes equally regardless of sample size. Penalises poor performance on the minority class. Standard in molecular property prediction benchmarks.

### Secondary Metric: Efficiency Score

$$\text{Efficiency} = \frac{F_1^2}{\log_{10}(\text{time}_{ms}) \times \log_{10}(\text{params})}$$

Logarithmic scaling ensures diminishing-return hardware gains; squaring F1 heavily rewards prediction quality. The leaderboard shows both metrics.

---

## Security Architecture

ENIGMA uses a layered security model to ensure fair, tamper-proof evaluation.

### Submission Encryption (RSA-2048)

```
┌──────────────┐    public_key.pem    ┌──────────────┐    GitHub Secret    ┌──────────────┐
│              │  ─────────────────▶  │              │  ─────────────────▶ │              │
│  Participant │    encrypt.py        │  GitHub PR   │    RSA_PRIVATE_KEY  │   CI Runner  │
│  predictions │  (RSA-2048, OAEP,   │  (.enc file) │    decrypt.py       │  (plaintext) │
│  (.csv)      │   SHA-256, chunked) │              │                     │              │
└──────────────┘                      └──────────────┘                     └──────────────┘
                                                                                 │
                                                                          score + rank
                                                                                 │
                                                                                 ▼
                                                                       Public Leaderboard
```

**How it works:**
1. **Encrypt** — You run `encryption/encrypt.py` with the public key. Your CSV is split into 190-byte chunks, each encrypted with OAEP/SHA-256 padding. The `.enc` file is unreadable without the private key.
2. **Submit** — You open a Pull Request containing only the `.enc` file. Other participants cannot see your predictions.
3. **Decrypt** — GitHub Actions injects the private key from a repository secret, decrypts, scores, and deletes the key — all within an ephemeral CI runner.
4. **Publish** — Only the final score and rank appear on the public leaderboard.

### Label Security

Test labels are **never** committed to this repository. During CI they are injected via:
1. **GitHub Secret** `TEST_LABELS_CSV` (base64-encoded CSV) — preferred
2. **Private repository** `enigma-private` (cloned with `PRIVATE_REPO_TOKEN`) — fallback

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/muuki2/enigma.git
cd enigma
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r starter_code/requirements.txt
```

### 3. Run Baselines

```bash
cd starter_code
python baseline.py --all       # run GCN, GIN, GraphSAGE
python baseline.py --model gcn # or a specific model
```

This downloads OGB MolBACE, trains for 50 epochs, generates `submissions/{model}_submission.csv`, and reports validation F1.

| Model | Validation Macro F1 |
|-------|---------------------|
| GCN | 0.6153 |
| GIN | 0.6103 |
| GraphSAGE | 0.5835 |

### 4. Explore the Data

```python
from ogb.graphproppred import PygGraphPropPredDataset

dataset = PygGraphPropPredDataset(name='ogbg-molbace')
graph = dataset[0]
print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
print(f"Node features: {graph.x.shape}, Label: {graph.y.item()}")
```

---

## Submission Process

### Step 1 — Generate Predictions

```csv
id,y_pred
0,1
1,0
6,1
...
```

- `id`: molecule index from `data/public/test.csv`
- `y_pred`: binary prediction (0 or 1). Legacy column name `target` is still accepted.

### Step 2 — Encrypt

```bash
python encryption/encrypt.py \
    submissions/inbox/my_team/run_01/predictions.csv \
    encryption/public_key.pem \
    submissions/inbox/my_team/run_01/predictions.enc
```

### Step 3 — Submit via Pull Request

```bash
git add submissions/inbox/my_team/run_01/predictions.enc
git commit -m "Submission: My Team Name"
git push origin my-branch && gh pr create   # or open PR on GitHub
```

<details>
<summary><strong>What happens after you submit</strong></summary>

1. **Decrypt** — CI decrypts your `.enc` using the private key from GitHub Secrets
2. **Validate** — `competition/validate_submission.py` checks format
3. **Score** — `competition/evaluate.py` computes Macro F1 against hidden labels
4. **Comment** — A bot comments on your PR with the score
5. **Leaderboard** — [Leaderboard](leaderboard/leaderboard.md) and [interactive board](https://muuki2.github.io/enigma/leaderboard.html) are updated automatically

</details>

### Optional: Efficiency Metadata

Include `metadata.json` alongside your submission to appear with efficiency metrics:

```json
{
  "team_name": "alice",
  "model_name": "MyGNN",
  "submission_type": "human",
  "efficiency_metrics": {"inference_time_ms": 5.2, "total_params": 45000}
}
```

Use `evaluation/speed_benchmark.py` to measure these values. See `schema/submission_metadata.json` for the full schema.

### Submission Layout

```
submissions/inbox/<team>/<run_id>/
├── predictions.enc    # Required (RSA-encrypted)
└── metadata.json      # Optional (efficiency + model info)
```

---

## Current Leaderboard

| Rank | Team | Macro-F1 | Efficiency | Params |
|------|------|----------|------------|--------|
| 🥇 1 | Baseline-Spectral | 0.7215 | 0.6360 | 40.4K |
| 🥈 2 | Baseline-DMPNN | 0.6674 | 0.0833 | 53.6K |
| 🥉 3 | Baseline-GCN | 0.6153 | - | - |

[Interactive Leaderboard](https://muuki2.github.io/enigma/leaderboard.html)

---

## Baseline GNN Architectures

The competition provides three baseline GNN architectures. Below are their message-passing formulations.

### Graph Convolutional Network (GCN)

GCN (Kipf & Welling, 2017) performs spectral graph convolutions using a first-order approximation:

$$\mathbf{h}_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\hat{d}_v \hat{d}_u}} \mathbf{W}^{(l)} \mathbf{h}_u^{(l)}\right)$$

where $\hat{d}_v = 1 + |\mathcal{N}(v)|$ is the augmented degree and $\mathbf{W}^{(l)}$ is a learnable weight matrix.

### GraphSAGE

GraphSAGE (Hamilton et al., 2017) learns to aggregate neighborhood features:

$$\mathbf{h}_v^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \cdot \text{CONCAT}\left(\mathbf{h}_v^{(l)}, \text{AGG}\left(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)\right)$$

where AGG can be mean, max-pool, or LSTM aggregation. Our baseline uses mean aggregation.

### Graph Isomorphism Network (GIN)

GIN (Xu et al., 2019) achieves maximal expressive power among message-passing GNNs:

$$\mathbf{h}_v^{(l+1)} = \text{MLP}^{(l)}\left((1 + \epsilon^{(l)}) \cdot \mathbf{h}_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}\right)$$

where $\epsilon$ is a learnable scalar. GIN is as powerful as the Weisfeiler-Lehman graph isomorphism test.

### Graph-Level Readout

All models use global mean pooling for graph-level prediction:

$$\mathbf{h}_G = \frac{1}{|V|} \sum_{v \in V} \mathbf{h}_v^{(L)}$$

followed by a linear classifier: $\hat{y} = \sigma(\mathbf{w}^\top \mathbf{h}_G + b)$

---

## Advanced GNN Architectures

Beyond the baselines, we provide two advanced architectures with stronger mathematical foundations.

### Directed Message Passing Neural Network (D-MPNN)

D-MPNN (Yang et al., 2019) is an edge-centric GNN designed for molecular graphs that prevents "message backflow" — a key limitation of standard MPNNs.

**Message Passing:**

$$\mathbf{m}_{uv}^{(l+1)} = \sum_{w \in \mathcal{N}(u) \setminus \{v\}} f\left(\mathbf{h}_u^{(l)}, \mathbf{m}_{wu}^{(l)}, \mathbf{e}_{uv}\right)$$

$$\mathbf{h}_v^{(l+1)} = g\left(\mathbf{h}_v^{(l)}, \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{uv}^{(l+1)}\right)$$

**Key Features:**
- Messages flow along directed edges
- Prevents information from immediately flowing back to source
- Edge features are first-class citizens
- Particularly effective for molecular property prediction

**Implementation:** `advanced_baselines/dmpnn.py`

### Spectral GNN with Laplacian Regularization

Our Spectral GNN operates in the graph frequency domain using Chebyshev polynomial approximations.

**Chebyshev Convolution:**

$$\mathbf{x} * g_\theta \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\mathbf{L}}) \mathbf{x}$$

where:
- $\tilde{\mathbf{L}} = \frac{2}{\lambda_{max}} \mathbf{L} - \mathbf{I}$ is the scaled Laplacian
- $T_k$ are Chebyshev polynomials: $T_0 = 1, T_1 = x, T_k = 2xT_{k-1} - T_{k-2}$
- $\theta_k$ are learnable spectral coefficients

**Laplacian Regularization Loss:**

We minimize the Dirichlet energy to encourage smoothness:

$$\mathcal{L}_{smooth} = \frac{1}{|V|} \mathbf{h}^\top \mathbf{L} \mathbf{h} = \frac{1}{|V|} \sum_{(i,j) \in E} \|\mathbf{h}_i - \mathbf{h}_j\|^2$$

**Laplacian Positional Encodings:**

Optional positional features from Laplacian eigenvectors:

$$\mathbf{L} \mathbf{u}_k = \lambda_k \mathbf{u}_k$$

The first $k$ eigenvectors provide structural position information.

**Implementation:** `advanced_baselines/spectral_gnn.py`

---

## Evaluation Dimensions

We evaluate submissions along multiple dimensions beyond raw accuracy.

### 1. Prediction Quality (Primary)

**Macro F1 Score** is the primary ranking metric (see [Evaluation Metrics](#evaluation-metric)).

### 2. Computational Efficiency

Tracked via the efficiency formula above. We record:
- Inference time (ms per batch)
- Parameter count
- Memory usage
- FLOPs estimate

Use the profiler in `evaluation/speed_benchmark.py` to measure your model.

### 3. Uncertainty Quantification

Good models should know when they don't know. We provide tools to evaluate:

**MC Dropout:** Epistemic uncertainty via multiple forward passes with dropout enabled:

$$\sigma^2_{\text{epistemic}} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t^2 - \left(\frac{1}{T} \sum_{t=1}^{T} \hat{y}_t\right)^2$$

**Conformal Prediction:** Distribution-free prediction sets with coverage guarantees:

$$C(x) = \{y : s(x, y) \leq \hat{q}\}$$

where $\hat{q}$ is calibrated on a holdout set to achieve $(1-\alpha)$ coverage.

**Temperature Scaling:** Post-hoc calibration via:

$$p(y|x) = \text{softmax}(z/T)$$

with temperature $T$ optimized on validation data.

**Metrics:**
- Expected Calibration Error (ECE)
- Brier Score
- Empirical Coverage at 90%

**Implementation:** `evaluation/uncertainty.py`

### 4. Adversarial Robustness

We evaluate model robustness to graph perturbations:

**Attack Types:**
1. **Random Edge Perturbation:** Add/remove random edges
2. **Gradient-Based Attack:** Remove high-importance edges
3. **Feature Noise:** Gaussian noise on node features
4. **Feature Masking:** Zero out random features

**Metrics:**
- Robust Accuracy under attack
- Attack Success Rate (ASR)

$$\text{ASR} = \frac{|\{x : f(x + \delta) \neq y, f(x) = y\}|}{|\{x : f(x) = y\}|}$$

**Implementation:** `evaluation/adversarial.py`

### 5. Pareto Efficiency

We visualize the accuracy-efficiency trade-off:

A model is **Pareto optimal** if no other model is:
- Better in accuracy AND equally efficient, OR
- Equally accurate AND more efficient, OR
- Better in both

**Hypervolume Indicator:**

$$\text{HV}(S) = \text{volume dominated by Pareto front } S$$

Higher hypervolume indicates better overall performance.

**Visualization:** `visualization/pareto_plot.py`

---

## Tips and Ideas

### Additional GNN Architectures
- **GAT** (Graph Attention Network) — attention-weighted message passing
- **MPNN** (Message Passing Neural Network) — edge-conditioned convolutions
- **AttentiveFP** — designed specifically for molecular property prediction
- **D-MPNN** — see our implementation in `advanced_baselines/dmpnn.py`
- **Spectral GNN** — see our implementation in `advanced_baselines/spectral_gnn.py`
- **Ensemble methods** — combine multiple architectures

### Techniques to Consider
- **Class weighting** — address class imbalance via weighted cross-entropy
- **Focal loss** — down-weight easy examples, focus on hard ones
- **Laplacian regularization** — encourage smooth representations (see Spectral GNN)
- **Data augmentation** — random edge dropping, node feature masking
- **Different pooling** — sum pooling, attention-based pooling, Set2Set
- **Virtual nodes** — add a global node connected to all atoms
- **Positional encodings** — Laplacian eigenvectors, random walk features
- **Learning rate scheduling** — cosine annealing, warm restarts
- **Early stopping** — monitor validation F1 to prevent overfitting

### Evaluation Tools
- **Speed benchmark**: `evaluation/speed_benchmark.py` — profile inference time
- **Uncertainty**: `evaluation/uncertainty.py` — MC Dropout, Conformal Prediction
- **Adversarial**: `evaluation/adversarial.py` — robustness testing
- **Visualization**: `visualization/pareto_plot.py` — Pareto front analysis

### Resources
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [OGB Leaderboard for MolBACE](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molbace)
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1901.00596)
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [GIN Paper](https://arxiv.org/abs/1810.00826)

---

## Repository Structure

```
enigma/
├── competition/                # Competition infrastructure
│   ├── config.yaml             # Single source of truth for all settings
│   ├── evaluate.py             # Scoring entry-point (CI)
│   ├── metrics.py              # Metric computation (Macro-F1, Efficiency)
│   ├── validate_submission.py  # Submission format validation
│   └── render_leaderboard.py   # Leaderboard renderer (Markdown + JS)
├── encryption/                 # RSA-2048 submission encryption
│   ├── encrypt.py              # Encrypt predictions (participant-facing)
│   ├── decrypt.py              # Decrypt submissions (CI-only)
│   └── public_key.pem          # RSA public key (safe to publish)
├── data/
│   ├── public/                 # Public data for participants
│   │   ├── train.csv, valid.csv, test.csv
│   ├── graphs/                 # Pre-computed A and X matrices (.npz)
│   │   ├── train_graphs.npz, valid_graphs.npz, test_graphs.npz
│   │   └── README_graphs.md
│   ├── mmp_split/              # MMP-OOD activity-cliff split
│   └── ogb/                    # OGB dataset (auto-downloaded)
├── submissions/inbox/          # Submit here: inbox/<team>/<run_id>/
├── leaderboard/                # Authoritative CSV + auto-generated Markdown
├── docs/                       # GitHub Pages interactive leaderboard
├── starter_code/               # Baseline GNNs (GCN, GIN, GraphSAGE)
├── advanced_baselines/         # D-MPNN + Spectral GNN
├── evaluation/                 # Speed, uncertainty, adversarial, MMP-OOD
├── visualization/              # Pareto front analysis
├── scripts/                    # Label generation, local tests, MMP evaluation
├── schema/                     # Submission metadata JSON schema
├── .github/workflows/          # CI: decrypt → validate → score → leaderboard
├── requirements.txt            # CI dependencies
└── README.md                   # This file
```

---

## Rules

1. **No external data**: Use only the provided OGB MolBACE dataset
2. **No pre-trained models**: Train from scratch; pre-trained molecular embeddings are not allowed
3. **One submission per team**: Each team may submit **only once** — make it count!
4. **One submission per PR**: Each pull request should contain exactly one predictions file
5. **Code sharing encouraged**: You may share code and ideas, but submit individually
6. **Fair play**: Do not attempt to access test labels or exploit the evaluation system
7. **Submission privacy**: All submissions must be encrypted using the provided RSA public key. Only final scores and ranks appear on the public leaderboard — private submissions must not be visible
8. **LLM usage restriction**: Large Language Models must not be used to fully design the competition, including dataset creation, task definition, or evaluation logic. This competition's dataset (OGB MolBACE) was created by the academic community, and the evaluation logic was designed by the organizer independently
9. **Computational affordability**: Full model training must not exceed **3 hours on CPU**. The provided dataset (1,210 training molecules, ~30 atoms each) and baseline models (~40K–54K parameters) train in minutes on CPU. Participants should keep model complexity within this budget
10. **Kaggle-style ranking**: Tied scores share the same rank on the leaderboard (min method). The next rank after a tie skips accordingly

---

## FAQ

**Q: Can I use libraries other than PyTorch Geometric?**
> Yes. You can use DGL, Spektral, JAX, or any other framework. Ensure your final predictions follow the CSV format.

**Q: How do I test locally before submitting?**
> Use the validation set to evaluate your model locally. Training labels are available via OGB; only test labels are hidden.

**Q: Can I submit multiple times?**
> No. Each team is limited to **one submission** — make it count! If you need to correct an error, contact the organisers.

**Q: How does the automated scoring work?**
> When you open a PR, GitHub Actions fetches the hidden test labels from a private repository, runs the scoring script, and comments on your PR with the result.

**Q: When does the competition end?**
> This is an ongoing challenge. Top performers will be contacted for the research opportunity.

---

## Acknowledgments

- **Dataset**: [Open Graph Benchmark](https://ogb.stanford.edu/)
- **Original BACE data**: [MoleculeNet](https://moleculenet.org/)

---

## References and Citations

If you use this challenge or the methods implemented here, please cite the following:

### Dataset

**Open Graph Benchmark (OGB)**
```bibtex
@article{hu2020ogb,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={22118--22133},
  year={2020}
}
```

**MoleculeNet**
```bibtex
@article{wu2018moleculenet,
  title={MoleculeNet: A Benchmark for Molecular Machine Learning},
  author={Wu, Zhenqin and Ramsundar, Bharath and Feinberg, Evan N and Gomes, Joseph and Geniesse, Caleb and Pappu, Aneesh S and Leswing, Karl and Pande, Vijay},
  journal={Chemical Science},
  volume={9},
  number={2},
  pages={513--530},
  year={2018},
  publisher={Royal Society of Chemistry}
}
```

### GNN Architectures

**GraphSAGE**
```bibtex
@inproceedings{hamilton2017inductive,
  title={Inductive Representation Learning on Large Graphs},
  author={Hamilton, William L and Ying, Rex and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

**Graph Convolutional Networks (GCN)**
```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={International Conference on Learning Representations},
  year={2017}
}
```

**Graph Isomorphism Network (GIN)**
```bibtex
@inproceedings{xu2019powerful,
  title={How Powerful are Graph Neural Networks?},
  author={Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

**Directed Message Passing Neural Network (D-MPNN)**
```bibtex
@article{yang2019analyzing,
  title={Analyzing Learned Molecular Representations for Property Prediction},
  author={Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and 
          Eiden, Philipp and Gao, Hua and Guzman-Perez, Angel and Hopper, Timothy and 
          Kelley, Brian and Mathea, Miriam and others},
  journal={Journal of Chemical Information and Modeling},
  volume={59},
  number={8},
  pages={3370--3388},
  year={2019},
  publisher={ACS Publications}
}
```

**Spectral Graph Theory**
```bibtex
@book{chung1997spectral,
  title={Spectral Graph Theory},
  author={Chung, Fan RK},
  year={1997},
  publisher={American Mathematical Society}
}
```

**Chebyshev Spectral Convolutions**
```bibtex
@inproceedings{defferrard2016convolutional,
  title={Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering},
  author={Defferrard, Micha{\"e}l and Bresson, Xavier and Vandergheynst, Pierre},
  booktitle={Advances in Neural Information Processing Systems},
  volume={29},
  year={2016}
}
```

**Conformal Prediction**
```bibtex
@article{romano2020classification,
  title={Classification with Valid and Adaptive Coverage},
  author={Romano, Yaniv and Sesia, Matteo and Candes, Emmanuel},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={3581--3591},
  year={2020}
}
```

### Libraries

**PyTorch Geometric**
```bibtex
@inproceedings{fey2019fast,
  title={Fast Graph Representation Learning with PyTorch Geometric},
  author={Fey, Matthias and Lenssen, Jan Eric},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019}
}
```

---

## Credits

### Dataset Creators
- **Jure Leskovec** (Stanford University) — Open Graph Benchmark, GraphSAGE
- **Weihua Hu** (Stanford University) — Open Graph Benchmark
- **Zhenqin Wu** and **Vijay Pande** (Stanford University) — MoleculeNet

### GNN Architecture Authors
- **William L. Hamilton**, **Rex Ying**, **Jure Leskovec** — GraphSAGE
- **Thomas N. Kipf**, **Max Welling** — Graph Convolutional Networks
- **Keyulu Xu**, **Weihua Hu**, **Jure Leskovec**, **Stefanie Jegelka** — Graph Isomorphism Network

### Library Developers
- **Matthias Fey**, **Jan Eric Lenssen** — PyTorch Geometric
- **Deep Graph Library (DGL) Team** — DGL Framework

### Special Thanks
- **[BASIRA Lab](https://basira-lab.com/)** — Research collaboration and support
- **Prof. Islem Rekik** (Imperial College London) — Mentorship and guidance

### Competition Organizer
- **Murat Kolic** — Sarajevo, Bosnia and Herzegovina

---

## Contact

For questions or issues, please open a [GitHub Issue](../../issues).

**Organizer:** Murat Kolic ([@muuki2](https://github.com/muuki2))
**Affiliation:** [BASIRA Lab](https://basira-lab.com/)
**Location:** Sarajevo, Bosnia and Herzegovina

---

<div align="center">

*ENIGMA — Encrypted Neural Inference on Graphs for Molecular Analysis*

</div>