# Fine-tuning Evo2-7B for Synthetic Nitrogen-Fixation Gene Generation

## 1. Project Objective

### 1.1 Problem Statement
The goal of this project is to fine-tune the Evo2-7B DNA language model to generate synthetic cyanobacterial genome segments containing functional nitrogen-fixation (nif) gene clusters.

**Specific Objectives:**
- Generate biologically plausible DNA sequences
- Preserve gene ordering, motifs, and context required for nitrogen fixation

### 1.2 Biological Background
Cyanobacteria are photosynthetic bacteria capable of fixing atmospheric nitrogen through the nitrogenase enzyme complex encoded by nif genes.

**Key Facts:**
- Earth's atmosphere contains ~78% nitrogen gas (N₂), which plants cannot use directly
- Nitrogen must be converted into ammonia (NH₃) through biological nitrogen fixation
- The nitrogenase enzyme complex is encoded by nifH, nifD, and nifK genes plus accessory genes
- Synthetic biology aims to engineer organisms with improved nitrogen fixation efficiency

**Why This Matters:**
- Nitrogen fixation is essential for ecosystems and agriculture
- Generative DNA models could accelerate discovery and design of optimized genomes
- Potential applications in sustainable agriculture and environmental biotechnology

## 2. Technical Specifications

### 2.1 Model Architecture
- **Base Model**: Evo2-7B (StripedHyena 2 architecture)
- **Context Length**: 8,192 tokens (base) → reduced to 4,096 for memory constraints
- **Parameters**: 7 billion
- **Pre-training**: OpenGenome2 (8.8 trillion tokens across all domains of life)

### 2.2 Compute Infrastructure
- **HPC Cluster**: University of South Dakota Innovator Cluster
- **GPUs**: NVIDIA A100 80GB (2-4 GPUs per node)
- **Nodes**: 14 GPU nodes (28 total GPUs total)
- **Scheduler**: SLURM 21.08.8
- **Containerization**: Singularity CE 4.1.2 / Apptainer 1.4.1
- **Container Image**: NVIDIA BioNeMo Framework (nvcr.io/nvidia/clara/bionemo-framework:nightly)

### 2.3 Software Stack
- **Core Framework**: NVIDIA BioNeMo Framework 24.09+
- **Model Source**: ARC Institute Evo2 (savanna_evo2_7b_base)
- **Data Processing**: Custom bioinformatics pipeline (BLAST, HMMER, bedtools)
- **Monitoring**: SLURM sacct, nvidia-smi, PyTorch memory profiler

## 3. Dataset Construction

### 3.1 Data Source
- **Source**: NCBI GenBank complete cyanobacterial genomes
- **Taxonomic Group**: Cyanobacteriota (taxid: 1117)
- **Assembly Level**: Complete genomes only
- **Retrieval Method**: NCBI datasets CLI tool

### 3.2 Dataset Statistics
- **Initial genomes**: 476 complete cyanobacterial genomes
- **Filtered genomes**: Genomes containing confirmed nif gene clusters
- **Final sequences**: 4,428 nif-enriched genomic segments
- **Sequence length range**: 500-2,000 base pairs
- **Total dataset size**: ~20 MB of FASTA data
- **Train/Val/Test split**: 90%/5%/5%

### 3.3 Target Signal: nif Gene Clusters
The fine-tuning task focuses on nitrogen fixation gene clusters, including:
- Core nitrogenase genes: nifH, nifD, nifK
- Associated regulatory genes
- Accessory and maturation genes
- Conserved promoter regions and operon structures

### 3.4 Tooling Rationale
| Tool | Purpose |
|------|---------|
| ncbi-blast+ | Identify known nif genes by sequence similarity |
| hmmer | Sensitive detection of homologous nif genes using profile HMMs |
| jq / wget | Metadata handling and automated downloads |

### 3.5 Pipeline Architecture
NCBI Genomes (476 complete) → BLAST/HMMER Filtering → nif Gene Extraction
↓
FASTA Dataset (4,428 sequences) → Quality Control & Deduplication
↓
BioNeMo Preprocessing → Binary Training Format → Evo2-7B Fine-tuning
↓
Checkpoints → Evaluation → Synthetic nif Sequence Generation


### 3.6 Dataset Output
**Final cleaned dataset**: `evo2_nif_dataset_final_cleaned.fna`

This file contains extracted genome segments enriched for nif-related regions, ready for model training.

## 4. Fine-Tuning Pipeline Overview

### 4.1 High-Level Stages
1. **Data Preparation** → Model Initialization → Training Setup
2. **Fine-Tuning** → Validation & Evaluation → Deployment
3. **Monitoring** → Iterative Improvement

## 5. Training Configuration

### 5.1 Data Preprocessing Configuration (`preprocess_config.yaml`)
```yaml
- datapaths: ["evo2_nif_dataset_final_cleaned.fna"]
  output_dir: "./preprocessed_data"
  output_prefix: nif_genes
  train_split: 0.9
  valid_split: 0.05
  test_split: 0.05
  overwrite: True
  embed_reverse_complement: true
  indexed_dataset_dtype: "uint8"
  tokenizer_type: "Byte-Level"
  append_eod: true
  workers: 8
```
### 5.2 Training Configuration (job_finetune_7b.slurm)
```yaml
#!/bin/bash
#SBATCH --job-name=evo2_7b_nitrogen
#SBATCH --nodes=2                    # 2 nodes
#SBATCH --gres=gpu:2                 # 2 GPUs per node (most common config)
#SBATCH --ntasks-per-node=2
#SBATCH --time=3-00:00:00            # 3 days for slower training
#SBATCH --mem=300G
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=gpu

module load singularity

echo "=== Starting fine-tuning with 2 nodes, 2 GPUs each (4 GPUs total) ==="

srun singularity exec --nv ~/bionemo-framework-nightly.sif bash << 'EOF'
train_evo2 \
    -d training_data_config.yaml \
    --dataset-dir ./preprocessed_data \
    --result-dir ./finetune_results \
    --model-size 7b \
    --devices 2 \
    --num-nodes 2 \
    --seq-length 8192 \
    --micro-batch-size 1 \
    --lr 0.000015 \
    --min-lr 1e-6 \
    --max-steps 2000 \
    --ckpt-dir ./checkpoints/nemo2_evo2_7b_8k \
    --val-check-interval 100 \
    --activation-checkpoint-recompute-num-layers 5
EOF
```
### 5.3 Training Data Config (training_data_config.yaml)
```yaml
- dataset_prefix: nif_genes_byte-level_train
  dataset_split: train
  dataset_weight: 1.0
- dataset_prefix: nif_genes_byte-level_val
  dataset_split: validation
  dataset_weight: 1.0
- dataset_prefix: nif_genes_byte-level_test
  dataset_split: test
  dataset_weight: 1.0
```
## 6. Training Environment Setup
### 6.1 Containerized Execution
```yaml
# Model conversion from HuggingFace to NeMo format
singularity exec --nv ~/bionemo-framework-nightly.sif \
    evo2_convert_to_nemo2 \
    --model-path hf://arcinstitute/savanna_evo2_7b_base \
    --model-size 7b \
    --output-dir ./checkpoints/nemo2_evo2_7b_8k

# Training execution
srun singularity exec --nv ~/bionemo-framework-nightly.sif \
    train_evo2 [arguments...]
```
## 7. Roadblocks Encountered
### 7.1 GPU Out-of-Memory (OOM) Errors

| Field | Details |
| :--- | :--- |
| **Error Encountered** | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 344 MiB. GPU total: 79.25 GiB. Used by process: 78.95 GiB` |
| **Observations** | * Evo2-7B nearly saturated available GPU memory (80GB A100). |
| | * Memory fragmentation further reduced usable capacity. |
| | * PyTorch reserved but unallocated memory accumulated over time. |
| **Root Causes** | * Full fine-tuning of a 7B-parameter model requires substantial memory. |
| | * Long DNA sequences (4,096 context) with large attention matrices. |
| | * No parameter-efficient fine-tuning (LoRA/QLoRA) applied initially. |

---

### 7.2 Distributed Training Issues

* **NCCL Communication Errors:**
    ```text
    torch.distributed.DistBackendError: NCCL error
    nvmlInit_v2() failed: Driver/library version mismatch
    ncclSystemError: System call failed or device error
    ```
* **Impact:**
    * Multi-node training failed completely.
    * Even single-node, multi-GPU training encountered communication issues.
    * Required fallback to single-GPU configuration.
* **Root Cause:** Version incompatibility between container CUDA drivers (12.x) and host drivers.

---

### 7.3 Job Timeouts

| Job ID | State | Runtime | Resource Request |
| :--- | :--- | :--- | :--- |
| 7232055 | CANCELLED | ~2 days | 4 nodes $\times$ 8 GPUs |
| 7234640 | TIMEOUT | ~3 days | 2 nodes $\times$ 4 GPUs |
| 7069533 | TIMEOUT | 3 days | 2 nodes $\times$ 2 GPUs |

* **Impact:**
    * Training did not complete sufficient steps for meaningful learning.
    * No stable checkpoints produced.
    * GPU resources reclaimed by scheduler before completion.

---

### 7.4 Container Compatibility Issues

* **Problems:**
    * `preprocess_evo2` command not found in base HPC environment.
    * Container tags outdated (24.09 not available, switched to `:nightly`).
    * Driver/library mismatches between host and container.
* **Solutions:**
    * Explicit **Singularity execution** for all BioNeMo commands.
    * Used **`:nightly` tag** for latest compatible container.
    * **Single-GPU mode** to avoid NCCL issues.
