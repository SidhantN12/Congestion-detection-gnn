# Poster replacement text

Real OpenROAD labels • nangate45 • 12 samples / 4 designs
Train: dynamic_node + tinyRocket; validation: ibex; test: aes.

| Method | Spearman H | Spearman V |
|---|---:|---:|
| CongestionGNN | 0.7570 | 0.7308 |
| RUDY | 0.8593 | 0.8402 |
| Density-only | 0.7805 | 0.7270 |
| Capacity-only | 0.1096 | 0.0956 |

Mean over three held-out aes variants. RUDY leads; no GNN superiority claim. Contribution: validated router-label extraction, memory-conscious graph construction, and reproducible per-channel evaluation.

GNN latency: median of 25 CPU forward calls after warm-up, per aes variant: u35_s2 150.1 ms, u40_s3 143.2 ms, u45_s1 140.1 ms.

Limitations: one test design, one technology, narrow tinyRocket utilisation range, and seed/utilisation confounding. No final sample overflows.

Corrected placed-cell ranges from the verified corpus:
- aes: 14,404–14,625 cells.
- dynamic_node: 12,703–12,977 cells.
- ibex: 14,536–14,620 cells.
- tinyRocket: 28,500–28,587 cells.
