# A Deficiency Aware Feature Complementarity Framework for Source Code Vulnerability Detection

---

Official implementation of the paper **"A Deficiency Aware Feature Complementarity Framework for Source Code Vulnerability Detection"**.

## 1. Abstract

DAFC-SVD is a **diagnostic feature-selection framework** for source code vulnerability detection. Instead of directly concatenating all available feature representations, DAFC-SVD first decomposes the feature space into 10 groups (F1–F10), evaluates each group independently per sample, measures **deficiency (Def)**, **complementarity (Com)**, **overlap (Ovl)** and **cost (Cost)**, then selects feature combinations **before** fusion.

The framework:
1. Evaluates ten feature groups independently (F1–F10).
2. Generates 375 candidate combinations (size 2–4).
3. Filters candidates by **Coverage**, **Overlap**, **Cost** and ranks them by **PreScore**.
4. Retrains the Top-7 combinations with **Multi-Metric Adaptive Fusion (MMAF)**.
5. Cross-validates on Juliet and performs external testing on Big-Vul.
6. Reports results under multiple criteria: detection performance, cost, and generalization.

## 2. Repository Structure

```
dafc_svd/
├── README.md                     
├── requirements.txt
├── config.py                     # All hyperparameters
├── download_datasets.py          # Step 1+2: download + preprocess + split
├── extract_features.py           # Step 3: extract F1–F10
├── main.py                       # Steps 4–15: pipeline
├── reproduce_all_results.py      # Step 16: rebuild tables/figures
└── src/
    ├── data_loader.py
    ├── preprocess.py
    ├── splits.py
    ├── devign_orig.py            # Devign loader (Google Drive)
    ├── juliet_nist.py            # Juliet loader (NIST GitHub mirror)
    ├── bigvul_orig.py            # Big-Vul loader (Google Drive)
    ├── joern_extractor.py        # Joern graph extraction (F2–F7)
    ├── gcn.py                    # GCN encoder
    ├── bilstm.py                 # BiLSTM encoder (F2)
    ├── feature_io.py             # Save raw artifacts (Table 4)
    ├── individual_train.py       # Step 4
    ├── prediction_log.py         # Step 5
    ├── combo_generator.py        # Step 6
    ├── metrics.py                # Step 7
    ├── filter_combos.py          # Step 8
    ├── dataloader.py
    ├── mmf.py                    # MMAF model
    ├── mmf_train.py              # Steps 9–11
    ├── evaluation.py
    ├── baselines.py              # Steps 13–14
    ├── ablation.py               # Step 12
    ├── sensitivity.py            # Step 15
    ├── plotting.py
    └── features/
        ├── f1_lexical.py         # TF-IDF 5,000 → 256
        ├── f2_slice.py           # Program slice + BiLSTM
        ├── f3_ast.py             # Joern AST + GCN
        ├── f4_cfg.py             # Joern CFG + GCN
        ├── f5_dataflow.py        # Joern data-flow + GCN
        ├── f6_call.py            # Joern call graph + GCN
        ├── f7_cpg.py             # Joern CPG + GCN
        ├── f8_codebert.py        # CodeBERT 768 → 256
        ├── f9_graphcodebert.py   # GraphCodeBERT 768 → 256
        └── f10_security.py       # 32 security metrics → 256
```

## 3. Data Sources

| Dataset | Source | URL | Raw | Clean |
|---|---|---|---|---|
| **Devign** | Google Drive (official) | https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view | 27,318 | 24,754 |
| **Juliet** | NIST SARD #112 (GitHub mirror) | https://github.com/arichardson/juliet-test-suite-c | 64,099 | 58,420 |
| **Big-Vul** | Google Drive (ZeoVan MSR 2020) | https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view | 188,636 | 151,820 |

Alternative sources:
- Juliet NIST official: https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip
- Big-Vul GitHub (metadata only): https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset

The loaders download the datasets automatically. Manual download is not required (unless Google Drive quota is exceeded — fallback instructions are printed).

## 4. Requirements

- Python ≥ 3.8
- **Joern ≥ 2.0** (required for F2–F7): https://joern.io/docs/installation/
- `git` command (required for Juliet)
- Optional: NVIDIA GPU (recommended for F8/F9 and MMAF training)

Set Joern path:
```bash
export JOERN_PATH=/path/to/joern       # Linux/macOS
set JOERN_PATH=C:\path\to\joern.bat    # Windows cmd
$env:JOERN_PATH="C:\path\to\joern.bat" # Windows PowerShell
```

## 5. Installation

```bash
git clone <this-repo>
cd dafc_svd
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 6. Quick Start

```bash
# Step 1+2: Download + preprocess + split (5–30 min)
python download_datasets.py

# Step 3: Extract F1–F10 features (10–17 hours, GPU recommended)
python extract_features.py --dataset all

# Steps 4–15: Full pipeline (5–10 hours, GPU required)
python main.py

# Step 16: Rebuild tables and figures
python reproduce_all_results.py
```

Analysis-only mode:
```bash
python reproduce_all_results.py
```

## 7. Pipeline Overview (16 Steps)

| Step | Description | Script |
|---|---|---|
| 1 | Prepare three datasets | `download_datasets.py` |
| 2 | Preprocess + fixed split IDs | `download_datasets.py` |
| 3 | Extract F1–F10 features | `extract_features.py` |
| 4 | Train individual classifiers F1–F10 (5 seeds) | `main.py` |
| 5 | Generate prediction / error / cost logs | `main.py` |
| 6 | Generate 375 candidate combinations | `main.py` |
| 7 | Compute Coverage / Com / Ovl / Cost / PreScore | `main.py` |
| 8 | Filter 375 → 286 → 132 → 41 → Top-7 | `main.py` |
| 9 | Train MMAF on Top-7 (Devign) | `main.py` |
| 10 | Retrain Top-7 on Juliet | `main.py` |
| 11 | External test on Big-Vul | `main.py` |
| 12 | Direct comparison + Ablation | `main.py` |
| 13 | Random / Top-individual baselines | `main.py` |
| 14 | CPG+RGCN / CodeBERT / DWF baselines | `main.py` |
| 15 | Sensitivity: 5-feature, threshold, PreScore, seed | `main.py` |
| 16 | Rebuild tables + figures | `reproduce_all_results.py` |

## 8. Reported Metrics

All numbers below match the paper **exactly** after running the pipeline. Numbering is sequential 8.1–8.21; the original paper's table number (if different) is noted in each subsection.

### 8.1 Dataset statistics 

| Dataset | Raw | Clean | Train | Val | Test |
|---|---|---|---|---|---|
| Devign  | 27,318  | 24,754  | 17,327  | 3,713  | 3,714  |
| Juliet  | 64,099  | 58,420  | 40,894  | 8,763  | 8,763  |
| Big-Vul | 188,636 | 151,820 | —       | —      | 22,773 |

Big-Vul test: **22,773 samples** = 15% stratified split (seed 42), with **1,316 vulnerable + 21,457 non-vulnerable**.

### 8.2 Feature configuration 

| Group | Representation | Encoder | Dim |
|---|---|---|---|
| F1  | Lexical/Text | TF-IDF (token + n-gram) | 5,000 → 256 |
| F2  | Sequence/Slice | Program slice + BiLSTM | 256 |
| F3  | Syntax/AST | Joern AST + GCN | 256 |
| F4  | Control-flow | Joern CFG + GCN | 256 |
| F5  | Data-flow/Dependency | Joern data-flow + GCN | 256 |
| F6  | Call/Inter-procedural | Joern call graph + GCN | 256 |
| F7  | Composite graph | Joern CPG + GCN | 256 |
| F8  | Deep semantic embedding | CodeBERT-base | 768 → 256 |
| F9  | Deep structural embedding | GraphCodeBERT-base | 768 → 256 |
| F10 | Security-risk metrics | 32 static metrics | 32 → 256 |

### 8.3 Extraction results on Devign 

| Group | File | Input | Success | Failed | Coverage | Time | Storage |
|---|---|---|---|---|---|---|---|
| F1  | F1_lexical_text.csv | 24,754 | 24,754 | 0 | 100.00% | 18 min | 0.21 GB |
| F2  | F2_sequence_slice.jsonl | 24,754 | 23,980 | 774 | 96.87% | 46 min | 0.62 GB |
| F3  | F3_ast.jsonl | 24,754 | 24,150 | 604 | 97.56% | 52 min | 0.74 GB |
| F4  | F4_cfg.pt | 24,754 | 23,790 | 964 | 96.11% | 1.18 h | 0.96 GB |
| F5  | F5_dataflow_dependency.pt | 24,754 | 23,020 | 1,734 | 92.99% | 1.86 h | 1.28 GB |
| F6  | F6_call_interprocedural.pt | 24,754 | 22,580 | 2,174 | 91.22% | 2.15 h | 1.44 GB |
| F7  | F7_composite_graph.pt | 24,754 | 21,940 | 2,814 | 88.63% | 2.74 h | 1.92 GB |
| F8  | F8_deep_semantic.npy | 24,754 | 24,754 | 0 | 100.00% | 1.21 h | 0.82 GB |
| F9  | F9_deep_structural.npy | 24,754 | 23,820 | 934 | 96.23% | 1.68 h | 0.89 GB |
| F10 | F10_security_metrics.csv | 24,754 | 24,620 | 134 | 99.46% | 12 min | 0.08 GB |

### 8.4 Cost components on Devign 

| Group | Ext time | Inf time (ms/sample) | Dim | Peak mem (MB) | Failure rate (%) | Cost |
|---|---|---|---|---|---|---|
| F1  | 18 min | 0.88 | 5,000 | 710  | 0.00  | 0.353 |
| F2  | 46 min | 0.34 | 256   | 860  | 3.13  | 0.162 |
| F3  | 52 min | 0.51 | 256   | 1,080 | 2.44  | 0.211 |
| F4  | 1.18 h | 0.56 | 256   | 1,220 | 3.89  | 0.284 |
| F5  | 1.86 h | 0.63 | 256   | 1,450 | 7.00  | 0.427 |
| F6  | 2.15 h | 0.72 | 256   | 1,690 | 8.78  | 0.521 |
| F7  | 2.74 h | 1.20 | 256   | 2,800 | 11.37 | 0.809 |
| F8  | 1.21 h | 0.58 | 768   | 1,760 | 0.00  | 0.292 |
| F9  | 1.68 h | 0.64 | 768   | 1,900 | 3.77  | 0.420 |
| F10 | 12 min | 0.18 | 32    | 620   | 0.54  | 0.009 |

Cost = `0.2 × (ext_norm + inf_norm + dim_norm + mem_norm + fail_norm)` with Min-Max normalization over 10 groups.

### 8.5 Individual performance on Devign 

| Group | Representation | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| F1  | Lexical/Text | 0.704 ± 0.008 | 0.722 ± 0.007 | 0.713 ± 0.007 | 0.781 ± 0.006 |
| F2  | Sequence/Slice | 0.754 ± 0.006 | 0.774 ± 0.007 | 0.764 ± 0.006 | 0.832 ± 0.005 |
| F3  | Syntax/AST | 0.773 ± 0.006 | 0.790 ± 0.006 | 0.781 ± 0.006 | 0.846 ± 0.005 |
| F4  | Control-flow | 0.786 ± 0.005 | 0.803 ± 0.006 | 0.794 ± 0.005 | 0.858 ± 0.005 |
| F5  | Data-flow/Dependency | 0.801 ± 0.005 | 0.818 ± 0.005 | 0.809 ± 0.005 | 0.875 ± 0.004 |
| F6  | Call/Inter-procedural | 0.779 ± 0.006 | 0.795 ± 0.006 | 0.787 ± 0.006 | 0.851 ± 0.005 |
| F7  | Composite graph | 0.803 ± 0.005 | 0.817 ± 0.005 | 0.810 ± 0.005 | 0.881 ± 0.004 |
| F8  | Deep semantic embedding | 0.807 ± 0.005 | 0.824 ± 0.005 | 0.815 ± 0.005 | 0.886 ± 0.004 |
| F9  | Deep structural embedding | 0.819 ± 0.004 | 0.836 ± 0.004 | 0.827 ± 0.004 | 0.897 ± 0.003 |
| F10 | Security-risk metrics | 0.789 ± 0.005 | 0.806 ± 0.005 | 0.797 ± 0.005 | 0.864 ± 0.004 |

Best single feature: **F9 (F1 = 0.827 ± 0.004)**.

### 8.6 Filtering funnel 

| Step | Remaining | Retention |
|---|---|---|
| Initial candidates | 375 | 100.00% |
| After Coverage ≥ 0.80 | 286 | 76.27% |
| After Overlap ≤ 0.60 | 132 | 35.20% |
| After Cost ≤ 0.75 | 41 | 10.93% |
| After PreScore ranking (K=7) | 7 | 1.87% |

### 8.7 Top-7 selected combinations 

| Rank | Symbol | Features | Coverage | Com | Ovl | Cost | PreScore |
|---|---|---|---|---|---|---|---|
| 1 | S1 | F5 + F8 + F9 + F10 | 0.925 | 0.472 | 0.326 | 0.720 | **0.846** |
| 2 | S2 | F5 + F8            | 0.929 | 0.426 | 0.284 | 0.570 | 0.835 |
| 3 | S3 | F5 + F9            | 0.927 | 0.419 | 0.302 | 0.610 | 0.831 |
| 4 | S4 | F5 + F8 + F9       | 0.925 | 0.448 | 0.341 | 0.700 | 0.828 |
| 5 | S5 | F8 + F9 + F10      | 0.949 | 0.423 | 0.382 | 0.660 | 0.823 |
| 6 | S6 | F5 + F8 + F10      | 0.928 | 0.432 | 0.318 | 0.640 | 0.821 |
| 7 | S7 | F8 + F9            | 0.956 | 0.405 | 0.438 | 0.550 | 0.817 |

### 8.8 Threshold sensitivity 

| Configuration | θ_cov | θ_ovl | θ_cost | After Coverage | After Ovl | After Cost | Rank of S1 |
|---|---|---|---|---|---|---|---|
| Conservative | 0.83 | 0.57 | 0.72 | 273 | 118 | 35 | 2 |
| Default      | 0.80 | 0.60 | 0.75 | 286 | 132 | 41 | 1 |
| Relaxed      | 0.77 | 0.63 | 0.78 | 303 | 149 | 49 | 2 |

### 8.9 Five-feature analysis 

| Rank | Feature combination | Coverage | Com | Ovl | Cost | PreScore |
|---|---|---|---|---|---|---|
| 1 | F2 + F3 + F8 + F9 + F10 | 0.923 | 0.487 | 0.401 | 0.718 | 0.813 |
| 2 | F2 + F4 + F8 + F9 + F10 | 0.910 | 0.478 | 0.417 | 0.731 | 0.809 |
| 3 | F2 + F3 + F4 + F8 + F10 | 0.917 | 0.466 | 0.439 | 0.696 | 0.804 |
| 4 | F2 + F3 + F5 + F8 + F10 | 0.886 | 0.474 | 0.425 | 0.734 | 0.800 |
| 5 | F3 + F4 + F5 + F8 + F10 | 0.879 | 0.462 | 0.447 | 0.743 | 0.796 |
| 6 | F2 + F3 + F4 + F5 + F10 | 0.861 | 0.451 | 0.472 | 0.749 | 0.791 |

Best 5-feature PreScore = 0.813, lower than S7 = 0.817 → no 5-feature combination was promoted to MMAF.

### 8.10 PreScore sensitivity 

| Configuration | α | β | γ | η | Rank of S1 | Overlap with Default | Common combinations |
|---|---|---|---|---|---|---|---|
| Default                  | 0.40 | 0.30 | 0.20 | 0.10 | 1 | 7/7 | S1, S2, S3, S4, S5, S6, S7 |
| BasePerf-oriented        | 0.45 | 0.25 | 0.20 | 0.10 | 3 | 6/7 | S1, S2, S3, S4, S6, S7 |
| Complementarity-oriented | 0.35 | 0.35 | 0.20 | 0.10 | 1 | 6/7 | S1, S2, S3, S4, S5, S6 |
| Overlap-aware            | 0.35 | 0.30 | 0.25 | 0.10 | 2 | 5/7 | S1, S2, S3, S4, S6 |
| Cost-aware               | 0.35 | 0.30 | 0.20 | 0.15 | 4 | 5/7 | S1, S2, S3, S6, S7 |

### 8.11 Seed stability 

| Symbol | Seed 42 | Seed 52 | Seed 62 | Seed 72 | Seed 82 | Frequency |
|---|---|---|---|---|---|---|
| S1 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| S2 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| S3 | ✓ | ✓ | ✓ |   | ✓ | 4/5 |
| S4 | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| S5 | ✓ |   | ✓ | ✓ | ✓ | 4/5 |
| S6 |   | ✓ | ✓ | ✓ | ✓ | 4/5 |
| S7 | ✓ | ✓ |   |   | ✓ | 3/5 |

Per-seed Top-7 overlap with mean Top-7: **6/7, 6/7, 6/7, 5/7, 7/7** (seeds 42, 52, 62, 72, 82).

### 8.12 MMAF on Devign 

| Symbol | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| S1 | 0.879 ± 0.004 | 0.883 ± 0.004 | **0.881 ± 0.004** | 0.932 ± 0.003 |
| S2 | 0.850 ± 0.005 | 0.858 ± 0.005 | 0.854 ± 0.005 | 0.914 ± 0.004 |
| S3 | 0.854 ± 0.005 | 0.862 ± 0.005 | 0.858 ± 0.005 | 0.917 ± 0.004 |
| S4 | 0.870 ± 0.004 | 0.876 ± 0.004 | 0.873 ± 0.004 | 0.928 ± 0.003 |
| S5 | 0.866 ± 0.004 | 0.872 ± 0.004 | 0.869 ± 0.004 | 0.925 ± 0.003 |
| S6 | 0.862 ± 0.005 | 0.868 ± 0.005 | 0.865 ± 0.005 | 0.922 ± 0.004 |
| S7 | 0.848 ± 0.005 | 0.856 ± 0.005 | 0.852 ± 0.005 | 0.911 ± 0.004 |

### 8.13 MMAF on Juliet 

| Symbol | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| S1 | 0.922 ± 0.003 | 0.902 ± 0.003 | **0.912 ± 0.003** | 0.952 ± 0.002 |
| S2 | 0.902 ± 0.004 | 0.884 ± 0.004 | 0.893 ± 0.004 | 0.933 ± 0.003 |
| S3 | 0.906 ± 0.004 | 0.888 ± 0.004 | 0.897 ± 0.004 | 0.936 ± 0.003 |
| S4 | 0.916 ± 0.003 | 0.898 ± 0.003 | 0.907 ± 0.003 | 0.948 ± 0.002 |
| S5 | 0.912 ± 0.003 | 0.896 ± 0.003 | 0.904 ± 0.003 | 0.945 ± 0.002 |
| S6 | 0.908 ± 0.004 | 0.890 ± 0.004 | 0.899 ± 0.004 | 0.941 ± 0.003 |
| S7 | 0.890 ± 0.004 | 0.870 ± 0.004 | 0.880 ± 0.004 | 0.920 ± 0.003 |

### 8.14 MMAF on Big-Vul (external test) 

| Symbol | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| S1 | 0.782 ± 0.005 | 0.788 ± 0.005 | **0.785 ± 0.005** | 0.868 ± 0.004 |
| S2 | 0.742 ± 0.006 | 0.750 ± 0.006 | 0.746 ± 0.006 | 0.829 ± 0.005 |
| S3 | 0.748 ± 0.006 | 0.756 ± 0.006 | 0.752 ± 0.006 | 0.834 ± 0.005 |
| S4 | 0.770 ± 0.005 | 0.776 ± 0.005 | 0.773 ± 0.005 | 0.858 ± 0.004 |
| S5 | 0.764 ± 0.005 | 0.770 ± 0.005 | 0.767 ± 0.005 | 0.852 ± 0.004 |
| S6 | 0.758 ± 0.006 | 0.764 ± 0.006 | 0.761 ± 0.006 | 0.846 ± 0.005 |
| S7 | 0.738 ± 0.006 | 0.746 ± 0.006 | 0.742 ± 0.006 | 0.824 ± 0.005 |

### 8.15 F1 summary across datasets 

| Symbol | Devign | Juliet | Big-Vul | Average | Drop |
|---|---|---|---|---|---|
| S1 | 0.881 | 0.912 | 0.785 | **0.859** | 0.096 |
| S2 | 0.854 | 0.893 | 0.746 | 0.831 | 0.108 |
| S3 | 0.858 | 0.897 | 0.752 | 0.836 | 0.106 |
| S4 | 0.873 | 0.907 | 0.773 | 0.851 | 0.100 |
| S5 | 0.869 | 0.904 | 0.767 | 0.847 | 0.102 |
| S6 | 0.865 | 0.899 | 0.761 | 0.842 | 0.104 |
| S7 | 0.852 | 0.880 | 0.742 | 0.825 | 0.110 |

### 8.16 Direct comparison 

| Method | Features | Devign | Juliet | Big-Vul | Extraction time | Cost |
|---|---|---|---|---|---|---|
| Best single feature | F9 | 0.827 ± 0.004 | 0.862 ± 0.004 | 0.704 ± 0.006 | 1.68 h  | 0.420 |
| Main embedding combo | F8 + F9 | 0.852 ± 0.005 | 0.880 ± 0.004 | 0.742 ± 0.006 | 2.89 h  | 0.550 |
| Full concatenation | F1–F10 | 0.864 ± 0.005 | 0.889 ± 0.004 | 0.738 ± 0.006 | 12.95 h | 1.000 |
| **DAFC-SVD + MMAF** | **S1** | **0.881 ± 0.004** | **0.912 ± 0.003** | **0.785 ± 0.005** | 4.95 h | 0.720 |

### 8.17 Internal comparison and ablation 

| Variant | Selected combo | Devign | Juliet | Big-Vul | Cost |
|---|---|---|---|---|---|
| Full DAFC-SVD | F5+F8+F9+F10 | 0.881 ± 0.004 | 0.912 ± 0.003 | 0.785 ± 0.005 | 0.720 |
| Without Coverage | F5+F6+F8+F9 | 0.872 ± 0.005 | 0.902 ± 0.004 | 0.765 ± 0.006 | **0.740** |
| Without Com | F3+F5+F8+F9 | 0.868 ± 0.005 | 0.898 ± 0.004 | 0.758 ± 0.006 | 0.710 |
| Without Ovl | F4+F8+F9+F10 | 0.866 ± 0.005 | 0.895 ± 0.004 | 0.754 ± 0.006 | 0.700 |
| Without Cost | F5+F7+F8+F9 | 0.879 ± 0.004 | 0.913 ± 0.003 | 0.782 ± 0.005 | 0.870 |
| Random + MMAF | F1+F2+F5+F10 | 0.852 ± 0.006 | 0.881 ± 0.005 | 0.731 ± 0.007 | 0.590 |
| Top-individual + MMAF | F5+F7+F8+F9 | 0.869 ± 0.005 | 0.899 ± 0.004 | 0.757 ± 0.006 | 0.870 |

### 8.18 Ablation fusion 

| Variant | Selected combo | Devign | Juliet | Big-Vul | Cost |
|---|---|---|---|---|---|
| Full DAFC-SVD (MMAF) | F5+F8+F9+F10 | 0.881 ± 0.004 | 0.912 ± 0.003 | 0.785 ± 0.005 | 0.720 |
| DAFC-SVD + Concatenation | F5+F8+F9+F10 | 0.861 ± 0.005 | 0.892 ± 0.004 | 0.746 ± 0.006 | 0.720 |
| DAFC-SVD + Average fusion | F5+F8+F9+F10 | 0.855 ± 0.005 | 0.884 ± 0.004 | 0.735 ± 0.006 | 0.720 |

### 8.19 Selection by usage objective 

| Objective | Combo | Devign | Juliet | Big-Vul | Avg | Cost |
|---|---|---|---|---|---|---|
| Detection-oriented | S1 | 0.881 | 0.912 | 0.785 | 0.859 | 0.720 |
| Low-cost-oriented | S7 | 0.852 | 0.880 | 0.742 | 0.825 | 0.550 |
| Generalization-oriented | S1 | 0.881 | 0.912 | 0.785 | 0.859 | 0.720 |
| Balanced multi-criteria | S4 | 0.873 | 0.907 | 0.773 | 0.851 | 0.700 |

### 8.20 Selection baselines (Random / Top-individual)

| Method | Combination | Devign | Juliet | Big-Vul | Cost |
|---|---|---|---|---|---|
| Random feature selection + MMAF | F1 + F2 + F5 + F10 | 0.852 ± 0.006 | 0.881 ± 0.005 | 0.731 ± 0.007 | 0.590 |
| Top-individual-feature selection + MMAF | F5 + F7 + F8 + F9 | 0.869 ± 0.005 | 0.899 ± 0.004 | 0.757 ± 0.006 | 0.870 |

### 8.21 External baselines

| Method | Approach | Dataset / setting | Reported result |
|---|---|---|---|
| PVDetector | Graph-based + pretraining | PVData / PVData+ / Big-Vul | F1 = 93.01% / 76.19% / 47.85% |
| ZSVulD | Transformer-based / zero-shot | Devign → ReVeal | F1 = 65.00% |
| FusionVul | Multimodal feature fusion | Devign / ReVeal / SVulD / DiverseVul | F1 = 58.42% / 47.09% / 53.86% / 25.12% |
| RLV | LLM + repository context | FFmpeg+QEMU / DiverseVul; unseen-project | F1 improvement = 26.83% over SOTA |

**Same-protocol baselines on Devign (paper Table 15, Update.docx):**

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| CPG + RGCN | 0.637 | 0.917 | 0.752 | 0.887 |
| CodeBERT   | 0.855 | 0.790 | 0.821 | 0.950 |
| DWF        | 0.910 | 0.870 | 0.890 | 0.985 |
| **DAFC-SVD (S1)** | 0.879 | 0.883 | 0.881 | 0.932 |

## 9. Output Files

```
data/{devign,juliet,bigvul}/{raw,clean,splits,features,joern_raw}
logs/{extraction,resource}/F1..F10.csv
selection/{candidates_2to4,qualified_41,top7}.csv
results/
├── tables/                    
├── figures/                   
├── models/                    
├── prediction_logs/{F}/seed*.csv
├── error_vectors/{F}/seed*.npy
└── cost_logs/{cost_log.csv, valid_sample_ids.json}
```


## 10. Environment

- CPU: Intel Core i9 (12 cores / 24 threads)
- RAM: 64 GB
- GPU: NVIDIA RTX 3090 (24 GB)
- OS: Ubuntu 20.04
- Python 3.9, PyTorch, Hugging Face Transformers, scikit-learn

#

## 11. License

[License here — e.g., MIT / Apache-2.0 / CC BY 4.0]

## 12. Contact

Open an issue or email [hlvnin88@gmail.com].