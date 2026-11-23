# System-1 Guided Path Planning (DS-S1-RRT* & S1-RRT*)

This repository explores the integration of fast, amortized "System 1" decision models (learned neural heuristics, Vision-Language Models) into asymptotically optimal sampling-based motion planning (RRT*), maintaining theoretical guarantees (probabilistic completeness and asymptotic optimality) even when the learned model is adversarial or arbitrary.

## Key Theoretical Formulations

- **[`ds_s1_rrt_star.md`](ds_s1_rrt_star.md)**: **Dual-Stratum System-1 RRT\* (DS-S1-RRT\*)**
  - Features stratum-tagged vertex architecture ($V_{\text{unif}}$ and $V_{\text{sys1}}$).
  - Dual-stratum neighborhood: metric ball query $r_u(n_u)$ on $V_{\text{unif}}$ (no $\varepsilon^{-1/d}$ inflation needed) and $k$-NN on $V_{\text{sys1}}$ ($O(\log |V|)$ bounded degree).
  - Lazy cost-sorted `ChooseParent` with early termination and monotonic rewiring.
  - Complete proofs for probabilistic completeness and asymptotic optimality via Karaman–Frazzoli $\theta$-discretization and Lévy's conditional Borel–Cantelli lemma.
- **[`s1_rrt_star_algorithm.md`](s1_rrt_star_algorithm.md)**: Initial formulation of S1-RRT\* with $\varepsilon$-uniform mixture sampling and inflated connection radius $\gamma_\varepsilon = \gamma_* \cdot \varepsilon^{-1/d}$.

## Project Structure

```
├── ds_s1_rrt_star.md          # Theoretical formulation and proofs for DS-S1-RRT*
├── s1_rrt_star_algorithm.md   # Initial formulation of S1-RRT*
├── vlm-rrt(1)/                # Implementation of VLM-guided RRT experiments
│   ├── vlm_rrt.py             # Core VLM-RRT algorithm and visualizer
│   ├── map_gen.py             # 2D environment and map generators
│   ├── test_api.py            # API testing for visual language models
│   ├── run_experiments.py     # Experiment runner
│   └── .env.example           # Template for API credentials
├── paper_source/              # LaTeX source files for paper
├── paper_text.txt             # Extracted text of literature reference
└── arxiv_source.tar.gz        # Raw archive of literature reference
```

## Getting Started

### Prerequisites

- Python 3.10+
- Required packages: `numpy`, `matplotlib`, `requests`, `python-dotenv`, `pandas`, `openpyxl`

### Configuration

Copy `.env.example` in `vlm-rrt(1)` to `.env` and fill in your API key:
```bash
cp "vlm-rrt(1)/.env.example" "vlm-rrt(1)/.env"
```
