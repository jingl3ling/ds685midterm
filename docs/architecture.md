# TCC Architecture

Architecture documentation for the Temporal Cycle-Consistency Learning codebase.

**Papers:**
- [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846) (Dwibedi et al., CVPR 2019)
- [Time-Contrastive Networks](https://arxiv.org/abs/1704.06888) (Sermanet et al., 2017)

## Module Dependency Diagram

```mermaid
graph TD
    subgraph Entry Points
        TR[train.py]
        EV[evaluate.py]
        EX[extract_embeddings.py]
        VI[visualize_alignment.py]
    end

    subgraph Algorithm Registry
        AL[algorithms.py<br/>get_algo]
    end

    subgraph Algorithms
        AA[algos/alignment.py<br/>TCC]
        AS[algos/sal.py<br/>Shuffle & Learn]
        AT[algos/tcn.py<br/>TCN]
        AC[algos/classification.py<br/>Supervised]
        AST[algos/alignment_sal_tcn.py<br/>Combined]
        AB[algos/algorithm.py<br/>Base Class]
    end

    subgraph Core Losses
        TA[tcc/alignment.py<br/>compute_alignment_loss]
        TD[tcc/deterministic_alignment.py<br/>align_pair_of_sequences]
        TS[tcc/stochastic_alignment.py<br/>gen_cycles, _align]
        TL[tcc/losses.py<br/>classification & regression]
    end

    subgraph Models
        MO[models.py<br/>BaseModel + Embedder]
    end

    subgraph Data
        DS[datasets.py<br/>create_dataset]
        PP[preprocessors/<br/>sequence preprocessing]
        DP[dataset_preparation/<br/>video/image to records]
    end

    subgraph Evaluation
        EC[evaluation/classification.py]
        EF[evaluation/few_shot_classification.py]
        EE[evaluation/event_completion.py]
        EK[evaluation/kendalls_tau.py]
    end

    CF[config.py]

    TR --> AL
    EV --> AL
    EX --> AL
    VI --> AL
    AL --> AA & AS & AT & AC & AST
    AA & AS & AT & AC & AST --> AB
    AB --> MO
    AB --> CF
    AA --> TA
    AST --> TA
    TA --> TD & TS
    TD --> TL
    TS --> TL
    TR --> DS
    EV --> DS
    DS --> PP
    EV --> EC & EF & EE & EK

    classDef entry fill:#0277bd,color:#fff,stroke:#01579b
    classDef algo fill:#00695c,color:#fff,stroke:#004d40
    classDef loss fill:#6a1b9a,color:#fff,stroke:#4a148c
    classDef model fill:#e65100,color:#fff,stroke:#bf360c
    classDef data fill:#2e7d32,color:#fff,stroke:#1b5e20
    classDef eval fill:#c62828,color:#fff,stroke:#b71c1c
    classDef config fill:#37474f,color:#fff,stroke:#263238
    classDef registry fill:#455a64,color:#fff,stroke:#37474f

    class TR,EV,EX,VI entry
    class AA,AS,AT,AC,AST,AB algo
    class TA,TD,TS,TL loss
    class MO model
    class DS,PP,DP data
    class EC,EF,EE,EK eval
    class CF config
    class AL registry
```

## Embedding Network Architecture

The embedder transforms raw video frames into 128-dimensional embeddings used for alignment.

```mermaid
graph LR
    subgraph Input
        FR["Frames<br/>[B, T, 224, 224, 3]"]
    end

    subgraph "BaseModel (ResNet50V2)"
        R5["ResNet50V2<br/>pretrained ImageNet"]
        CF["Conv4c features<br/>[B, T, 14, 14, 1024]"]
    end

    subgraph "ConvEmbedder"
        CX["Context stacking<br/>k-1 neighbor frames"]
        C3["Conv3D layers<br/>temporal aggregation"]
        MP["3D MaxPooling"]
        F1["FC 512 + ReLU"]
        F2["FC 512 + ReLU"]
        LP["Linear projection"]
        LN["L2 Normalize"]
    end

    subgraph Output
        EM["Embeddings<br/>[B, T, 128]"]
    end

    FR --> R5 --> CF --> CX --> C3 --> MP --> F1 --> F2 --> LP --> LN --> EM

    classDef input fill:#0277bd,color:#fff,stroke:#01579b
    classDef backbone fill:#e65100,color:#fff,stroke:#bf360c
    classDef embedder fill:#6a1b9a,color:#fff,stroke:#4a148c
    classDef output fill:#2e7d32,color:#fff,stroke:#1b5e20

    class FR input
    class R5,CF backbone
    class CX,C3,MP,F1,F2,LP,LN embedder
    class EM output
```

**Alternative embedder:** ConvGRUEmbedder replaces Conv3D + MaxPool with Conv2D per frame followed by GRU layers for temporal modeling.

## TCC Loss Computation

The core TCC loss enforces cycle-consistency across video pairs through soft nearest neighbor matching.

### Cycle-Consistency Principle

```mermaid
graph LR
    subgraph "Sequence U"
        U["u_i<br/>selected frame"]
        UB["û_k<br/>cycle-back"]
    end

    subgraph "Sequence V"
        V["ṽ<br/>soft nearest neighbor"]
    end

    U -->|"softmax(-‖u_i - v_j‖²)"| V
    V -->|"softmax(-‖ṽ - u_k‖²)"| UB
    UB -.->|"loss: i should equal k"| U

    classDef seqU fill:#0277bd,color:#fff,stroke:#01579b
    classDef seqV fill:#e65100,color:#fff,stroke:#bf360c

    class U,UB seqU
    class V seqV
```

### Deterministic Alignment Pipeline

```mermaid
graph TD
    E1["embs1 [M, D]<br/>Sequence U embeddings"]
    E2["embs2 [N, D]<br/>Sequence V embeddings"]

    SIM12["Similarity matrix sim_12<br/>[M, N]<br/>L2 or cosine / temperature"]
    SM1["Softmax over columns<br/>α_j = softmax(sim_12)"]
    WE["Weighted embeddings<br/>ṽ = Σ α_j · v_j"]
    SIM21["Return similarity sim_21<br/>[M, M]"]

    LAB["One-hot labels<br/>diagonal identity"]
    LOSS{"Loss type"}
    CBC["CBC: Cross-entropy<br/>-Σ y_j log(ŷ_j)"]
    CBR["CBR: Regression<br/>|i - μ|² / σ² + λ log(σ)"]

    E1 & E2 --> SIM12
    SIM12 --> SM1
    SM1 --> WE
    E1 --> SIM21
    WE --> SIM21
    E1 --> LAB
    SIM21 --> LOSS
    LAB --> LOSS
    LOSS -->|classification| CBC
    LOSS -->|regression| CBR

    classDef emb fill:#0277bd,color:#fff,stroke:#01579b
    classDef comp fill:#6a1b9a,color:#fff,stroke:#4a148c
    classDef decision fill:#e65100,color:#fff,stroke:#bf360c
    classDef lossnode fill:#c62828,color:#fff,stroke:#b71c1c

    class E1,E2 emb
    class SIM12,SM1,WE,SIM21,LAB comp
    class LOSS decision
    class CBC,CBR lossnode
```

### Loss Variants

**Cycle-Back Classification (CBC):**
Cross-entropy loss on the cycle-back logits, treating each frame position as a class.

```
L_cbc = -Σⱼ yⱼ log(ŷⱼ)
```

**Cycle-Back Regression (CBR):**
Fits a Gaussian to the cycle-back distribution and penalizes distance from the true index with variance regularization.

```
β = softmax(logits)
μ = Σ(steps · β)        # predicted time index
σ² = Σ(steps² · β) - μ² # predicted variance
L_cbr = |i - μ|² / σ² + λ · log(σ)
```

CBR with λ=0.001 outperforms CBC and MSE variants (paper ablation).

### Stochastic vs Deterministic

| Mode | Description |
|------|------------|
| **Deterministic** | Aligns all N*(N-1) pairs in batch. Concatenates logits/labels from all pairs. |
| **Stochastic** | Generates random cycles of configurable length (2+). Randomly selects frames and cycles through sequences. More scalable for large batches. |

## Training Pipeline

```mermaid
graph TD
    subgraph "Data Pipeline"
        VID["Video files"]
        DEC["Decode frames"]
        SAM["Sample T frames<br/>stride / offset_uniform"]
        AUG["Augment<br/>flip, crop, color jitter"]
        BAT["Batch [B, T, 224, 224, 3]"]
    end

    subgraph "Forward Pass"
        CNN["BaseModel<br/>ResNet50V2"]
        EMB["ConvEmbedder<br/>Conv3D → FC → 128-d"]
        OUT["Embeddings [B, T, 128]"]
    end

    subgraph "Loss Computation"
        TCC["TCC alignment loss"]
        SAL["SAL loss<br/>(optional)"]
        TCN["TCN loss<br/>(optional)"]
        TOT["Total loss<br/>w₁·TCC + w₂·SAL + w₃·TCN"]
    end

    subgraph "Optimization"
        GRD["Compute gradients"]
        OPT["Adam / Momentum<br/>LR schedule"]
        CKP["Checkpoint + TensorBoard"]
    end

    VID --> DEC --> SAM --> AUG --> BAT
    BAT --> CNN --> EMB --> OUT
    OUT --> TCC & SAL & TCN
    TCC & SAL & TCN --> TOT
    TOT --> GRD --> OPT --> CKP

    classDef data fill:#2e7d32,color:#fff,stroke:#1b5e20
    classDef forward fill:#0277bd,color:#fff,stroke:#01579b
    classDef loss fill:#6a1b9a,color:#fff,stroke:#4a148c
    classDef optim fill:#e65100,color:#fff,stroke:#bf360c

    class VID,DEC,SAM,AUG,BAT data
    class CNN,EMB,OUT forward
    class TCC,SAL,TCN,TOT loss
    class GRD,OPT,CKP optim
```

### Training Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Image size | 224×224 | ResNet50 input |
| Embedding dim | 128 | L2-normalized |
| Context frames | k (configurable) | Temporal context for Conv3D |
| Batch size | configurable | N sequences per batch |
| Optimizer | Adam / Momentum | With LR decay (fixed/exp/manual) |
| Similarity | L2 or cosine | Scaled by temperature |
| Loss type | CBR (regression_mse_var) | Best performing variant |

## Evaluation Pipeline

Evaluation extracts frozen embeddings and trains lightweight classifiers on top — no fine-tuning of the pretrained model.

```mermaid
graph TD
    subgraph "Embedding Extraction"
        CK["Load checkpoint"]
        VD["All video frames"]
        FW["Forward pass<br/>(frozen model)"]
        EB["Embeddings<br/>[N_frames, 128]"]
    end

    subgraph "Evaluation Tasks"
        PC["Phase Classification<br/>SVM on embeddings → accuracy"]
        FS["Few-Shot Classification<br/>1-shot SVM → accuracy"]
        PP["Phase Progression<br/>Linear regression → R²"]
        KT["Kendall's Tau<br/>NN alignment → τ score"]
    end

    CK --> FW
    VD --> FW
    FW --> EB
    EB --> PC & FS & PP & KT

    classDef extract fill:#0277bd,color:#fff,stroke:#01579b
    classDef task fill:#c62828,color:#fff,stroke:#b71c1c

    class CK,VD,FW,EB extract
    class PC,FS,PP,KT task
```

### Evaluation Tasks

| Task | Method | Metric | Description |
|------|--------|--------|-------------|
| Phase Classification | SVM on embeddings | Accuracy | Classify action phase per frame |
| Few-Shot Classification | 1-shot SVM | Accuracy | Single labeled video for training |
| Phase Progression | Linear regression | R² | Predict fraction of task completion |
| Kendall's Tau | NN alignment | τ ∈ [-1, 1] | Temporal concordance between video pairs |

## Shuffle and Learn (SAL) Loss

SAL trains the embedder to distinguish temporally ordered from shuffled frame sequences.

```mermaid
graph TD
    EB["Embeddings [B, T, 128]"]
    S5["Sample 5 frames<br/>per sequence"]
    TP["Create triplets<br/>ordered + shuffled"]
    CT["Concatenate triplet<br/>embeddings [3×128]"]
    FC["FC classifier<br/>→ 2 classes"]
    CE["Binary cross-entropy<br/>+ label smoothing"]

    EB --> S5 --> TP --> CT --> FC --> CE

    classDef emb fill:#0277bd,color:#fff,stroke:#01579b
    classDef proc fill:#6a1b9a,color:#fff,stroke:#4a148c
    classDef loss fill:#c62828,color:#fff,stroke:#b71c1c

    class EB emb
    class S5,TP,CT,FC proc
    class CE loss
```

**Algorithm:**
1. Randomly select 5 frame indices from each sequence
2. Form ordered triplets (f₁, f₂, f₃) where f₁ < f₂ < f₃
3. Form shuffled triplets by swapping positions
4. Concatenate 3 frame embeddings → 384-dim vector
5. FC classifier predicts ordered vs shuffled → binary cross-entropy

## TCN (Time-Contrastive Networks) Loss

TCN uses an N-pairs metric learning loss on temporally sampled anchor/positive frame pairs.

```mermaid
graph TD
    EB["Embeddings [B, 2T, 128]<br/>doubled for anchor/positive"]
    SP["Split even/odd indices"]
    AN["Anchors<br/>even frames"]
    PO["Positives<br/>odd frames<br/>(within temporal window)"]
    SM["Similarity matrix<br/>dot product [T, T]"]
    LB["Label matrix<br/>from temporal indices"]
    NP["N-pairs loss<br/>softmax cross-entropy<br/>+ L2 regularization"]

    EB --> SP
    SP --> AN & PO
    AN & PO --> SM
    AN --> LB
    SM & LB --> NP

    classDef emb fill:#0277bd,color:#fff,stroke:#01579b
    classDef proc fill:#6a1b9a,color:#fff,stroke:#4a148c
    classDef loss fill:#c62828,color:#fff,stroke:#b71c1c

    class EB emb
    class SP,AN,PO,SM,LB proc
    class NP loss
```

**N-pairs loss:**
```
L_tcn = softmax_cross_entropy(anchors · positives^T, labels) + λ · (‖anchors‖² + ‖positives‖²)
```

The positive window controls how far apart anchor/positive frames can be temporally. L2 regularization (λ) prevents embedding collapse.

## Combined Loss (TCC + SAL + TCN)

The combined algorithm (`alignment_sal_tcn`) computes a weighted sum:

```
L_total = w_align · L_tcc + w_sal · L_sal + w_tcn · L_tcn
```

Weights are configurable via `CONFIG.ALIGNMENT_SAL_TCN.ALIGNMENT_LOSS_WEIGHT` and `SAL_LOSS_WEIGHT`. The paper shows that TCC+TCN achieves the best performance on fine-grained temporal tasks.

## Configuration Structure

```mermaid
graph TD
    CFG["CONFIG"]

    TRAIN["TRAIN<br/>max_iters, batch_size<br/>num_frames, visualize_interval"]
    EVAL["EVAL<br/>batch_size, tasks<br/>kendalls_tau_stride<br/>classification_fractions"]
    MODEL["MODEL<br/>embedder_type: conv/convgru<br/>base_model: ResNet50/VGGM<br/>train_base: frozen/only_bn/train_all<br/>embedding_size: 128"]
    ALIGN["ALIGNMENT<br/>loss_type: classification/regression<br/>similarity_type: l2/cosine<br/>temperature, cycle_length<br/>variance_lambda, label_smoothing"]
    SALL["SAL<br/>dropout_rate, fc_layers<br/>shuffle_fraction, num_samples"]
    TCNC["TCN<br/>positive_window<br/>reg_lambda"]
    OPT["OPTIMIZER<br/>type: Adam/Momentum<br/>lr: initial, decay_type"]
    DATA["DATA<br/>sampling_strategy<br/>stride, num_steps<br/>augmentation flags"]

    CFG --> TRAIN & EVAL & MODEL & ALIGN & SALL & TCNC & OPT & DATA

    classDef root fill:#37474f,color:#fff,stroke:#263238
    classDef section fill:#455a64,color:#fff,stroke:#37474f

    class CFG root
    class TRAIN,EVAL,MODEL,ALIGN,SALL,TCNC,OPT,DATA section
```

## PyTorch Migration Map

Key TensorFlow → PyTorch equivalences for the port:

| TensorFlow | PyTorch | Used in |
|------------|---------|---------|
| `tf.keras.Model` | `nn.Module` | All algorithm/model classes |
| `tf.nn.l2_normalize` | `F.normalize` | Embedding normalization |
| `tf.nn.softmax` | `F.softmax` | Soft nearest neighbor |
| `tf.matmul` | `torch.matmul` / `@` | Similarity computation |
| `tf.GradientTape` | `loss.backward()` | Training loop |
| `tf.stop_gradient` | `.detach()` | Loss computation |
| `tf.data.TFRecordDataset` | `torch.utils.data.Dataset` | Data loading |
| `tf.keras.layers.Conv3D` | `nn.Conv3d` | Temporal convolutions |
| `tf.keras.layers.Dense` | `nn.Linear` | FC layers |
| `CuDNNGRU` | `nn.GRU` | ConvGRU embedder |
| `tf.distribute.MirroredStrategy` | `DistributedDataParallel` | Multi-GPU |
| `tf.summary` | `torch.utils.tensorboard` | Logging |
| `tf.keras.backend.learning_phase()` | `model.train()/eval()` | Train/eval mode |
| `tf.image.*` augmentations | `torchvision.transforms` | Data augmentation |
| EasyDict config | dataclasses / OmegaConf | Configuration |
