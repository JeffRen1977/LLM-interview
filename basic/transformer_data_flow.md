# Transformer Encoder Data Flow Diagram

## Overview
This document provides a comprehensive visualization of how data flows through a Transformer Encoder block, showing the step-by-step process from input to output.

## Data Flow Architecture

### 1. Input Processing Stage
```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT STAGE                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   Input Embeddings  │    │  Position Encoding  │             │
│  │     (d_model)       │    │     (d_model)       │             │
│  └─────────┬───────────┘    └─────────┬───────────┘             │
│            │                          │                         │
│            └──────────┐    ┌──────────┘                         │
│                       │    │                                    │
│                       ▼    ▼                                    │
│              ┌─────────────────────┐                            │
│              │      ADDITION       │                            │
│              │   (Element-wise)    │                            │
│              └─────────┬───────────┘                            │
│                        │                                        │
│                        ▼                                        │
│              ┌─────────────────────┐                            │
│              │  Positional Embedding│                           │
│              │     Vector (d_model) │                           │
└──────────────┴─────────────────────┴──────────────────────────┘
```

**What happens here:**
- Input tokens are converted to dense vectors (embeddings)
- Positional information is added to preserve sequence order
- The two vectors are combined through element-wise addition

---

### 2. Multi-Head Self-Attention Sub-layer

```
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-HEAD ATTENTION SUB-LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 RESIDUAL CONNECTION                     │    │
│  │              (Skip Connection Input)                    │    │
│  └─────────────────────┬───────────────────────────────────┘    │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              POSITIONAL EMBEDDING VECTOR                │    │
│  │                     (d_model)                           │    │
│  └─────────────────────┬───────────────────────────────────┘    │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MULTI-HEAD ATTENTION MECHANISM             │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │    │
│  │  │   Head 1    │ │   Head 2    │ │   Head h    │        │    │
│  │  │ (d_k, d_v)  │ │ (d_k, d_v)  │ │ (d_k, d_v)  │        │    │
│  │  └─────┬───────┘ └─────┬───────┘ └─────┬───────┘        │   │
│  │        │                │               │               │   │
│  │        └────────────────┼───────────────┘               │   │
│  │                         │                               │   │
│  │                         ▼                               │   │
│  │              ┌─────────────────────┐                    │   │
│  │              │   CONCATENATE HEADS │                    │   │
│  │              │      (d_model)      │                    │   │
│  │              └─────────┬───────────┘                    │   │
│  │                        │                                │   │
│  │                        ▼                                │   │
│  │              ┌─────────────────────┐                    │   │
│  │              │   LINEAR PROJECTION │                    │   │
│  │              │      (d_model)      │                    │   │
│  └──────────────┴─────────┬───────────┴────────────────────┘   │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ATTENTION OUTPUT                          │   │
│  │                   (d_model)                           │   │
└──┴─────────────────────────────────────────────────────────┴───┘
```

**Key Components:**
- **Query (Q)**: Represents what the token is looking for
- **Key (K)**: Represents what the token offers
- **Value (V)**: Represents the actual content
- **Attention Score**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

---

### 3. First Add & Layer Normalization

```
┌─────────────────────────────────────────────────────────────────┐
│              FIRST ADD & LAYER NORMALIZATION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐           │
│  │  RESIDUAL INPUT     │    │ ATTENTION OUTPUT    │           │
│  │   (d_model)         │    │     (d_model)       │           │
│  └─────────┬───────────┘    └─────────┬───────────┘           │
│            │                          │                       │
│            └──────────┐    ┌──────────┘                       │
│                       │    │                                  │
│                       ▼    ▼                                  │
│              ┌─────────────────────┐                          │
│              │      ADDITION       │                          │
│              │   (Element-wise)    │                          │
│              └─────────┬───────────┘                          │
│                        │                                      │
│                        ▼                                      │
│              ┌─────────────────────┐                          │
│              │   LAYER NORMALIZATION│                          │
│              │     (d_model)       │                          │
└──────────────┴─────────────────────┴──────────────────────────┘
```

**What happens here:**
- **Residual Connection**: Adds the original input to the attention output
- **Layer Normalization**: Normalizes the sum to stabilize training
- **Formula**: `LayerNorm(x + Sublayer(x))`

---

### 4. Feed-Forward Network Sub-layer

```
┌─────────────────────────────────────────────────────────────────┐
│              FEED-FORWARD NETWORK SUB-LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 RESIDUAL CONNECTION                     │    │
│  │              (Skip Connection Input)                    │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                       │
│                        ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              NORMALIZED VECTOR                          │   │
│  │                   (d_model)                             │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                       │
│                        ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              FEED-FORWARD NETWORK                       │   │
│  │  ┌─────────────────────┐ ┌─────────────────────┐        │   │
│  │  │   LINEAR LAYER 1    │ │   LINEAR LAYER 2    │        │   │
│  │  │   (d_model → d_ff)  │ │   (d_ff → d_model)  │        │   │
│  │  └─────────┬───────────┘ └─────────┬───────────┘        │   │
│  │            │                        │                   │   │
│  │            └────────────┐  ┌────────┘                   │   │
│  │                         │  │                            │   │
│  │                         ▼  ▼                            │   │
│  │              ┌─────────────────────┐                    │   │
│  │              │   ACTIVATION (ReLU) │                    │   │
│  │              └─────────┬───────────┘                    │   │
│  └────────────────────────┼────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              FFN OUTPUT                                 │   │
│  │                   (d_model)                             │   │
└──┴─────────────────────────────────────────────────────────┴───┘
```

**FFN Architecture:**
- **First Linear Layer**: Expands from `d_model` to `d_ff` (typically 4x larger)
- **ReLU Activation**: Introduces non-linearity
- **Second Linear Layer**: Contracts back to `d_model`

---

### 5. Second Add & Layer Normalization

```
┌─────────────────────────────────────────────────────────────────┐
│              SECOND ADD & LAYER NORMALIZATION                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐           │
│  │  RESIDUAL INPUT     │    │    FFN OUTPUT       │           │
│  │   (d_model)         │    │     (d_model)       │           │
│  └─────────┬───────────┘    └─────────┬───────────┘           │
│            │                          │                       │
│            └──────────┐    ┌──────────┘                       │
│                       │    │                                  │
│                       ▼    ▼                                  │
│              ┌─────────────────────┐                          │
│              │      ADDITION       │                          │
│              │   (Element-wise)    │                          │
│              └─────────┬───────────┘                          │
│                        │                                      │
│                        ▼                                      │
│              ┌─────────────────────┐                          │
│              │   LAYER NORMALIZATION│                          │
│              │     (d_model)       │                          │
└──────────────┴─────────────────────┴──────────────────────────┘
```

---

### 6. Complete Transformer Encoder Block Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐                                        │
│  │   INPUT TOKENS      │                                        │
│  └─────────┬───────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────┐                                        │
│  │  EMBEDDING + POS    │                                        │
│  │     (d_model)       │                                        │
│  └─────────┬───────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              SUB-LAYER 1: MULTI-HEAD ATTENTION          │    │
│  │  ┌─────────────────┐  ┌─────────────────┐               │    │
│  │  │   RESIDUAL      │  │   ATTENTION     │               │    │
│  │  │   CONNECTION    │  │   MECHANISM     │               │    │
│  │  └───────┬─────────┘  └───────┬─────────┘               │    │
│  │          │                    │                         │    │
│  │          └────────┐  ┌────────┘                         │    │
│  │                   │  │                                  │    │
│  │                   ▼  ▼                                  │    │
│  │          ┌─────────────────┐                            │    │
│  │          │   ADD + NORM    │                            │    │
│  │          └───────┬─────────┘                            │    │
│  └──────────────────┼──────────────────────────────────────┘    │
│                     │                                           │
│                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              SUB-LAYER 2: FEED-FORWARD NETWORK          │    │
│  │  ┌─────────────────┐  ┌─────────────────┐               │    │
│  │  │   RESIDUAL      │  │   FFN LAYERS    │               │    │
│  │  │   CONNECTION    │  │                 │               │    │
│  │  └───────┬─────────┘  └───────┬─────────┘               │    │
│  │          │                    │                         │    │
│  │          └────────┐  ┌────────┘                         │    │
│  │                   │  │                                  │    │
│  │                   ▼  ▼                                  │    │
│  │          ┌─────────────────┐                            │    │
│  │          │   ADD + NORM    │                            │    │
│  │          └───────┬─────────┘                            │    │
│  └──────────────────┼──────────────────────────────────────┘    │
│                     │                                           │
│                     ▼                                           │
│  ┌─────────────────────┐                                        │
│  │   ENCODER OUTPUT    │                                        │
│  │     (d_model)       │                                        │
│  └─────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulations

### 1. Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 2. Scaled Dot-Product Attention
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

### 3. Feed-Forward Network
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### 4. Layer Normalization
```
LayerNorm(x) = γ * (x - μ)/√(σ² + ε) + β
```

---

## Key Design Principles

### 1. **Residual Connections**
- Help with gradient flow during backpropagation
- Allow the model to learn residual functions
- Formula: `x + Sublayer(x)`

### 2. **Layer Normalization**
- Stabilizes training by normalizing activations
- Applied after each sub-layer
- Helps with training deep networks

### 3. **Multi-Head Attention**
- Allows the model to attend to different positions simultaneously
- Each head learns different attention patterns
- Enables parallel processing

### 4. **Positional Encoding**
- Provides sequence order information
- Uses sinusoidal functions for generalization
- Added to input embeddings

---

## Data Dimensions Throughout the Flow

| Stage | Dimension | Description |
|-------|-----------|-------------|
| Input Tokens | `(seq_len,)` | Sequence of token IDs |
| Embeddings | `(seq_len, d_model)` | Dense vector representations |
| Positional Encoding | `(seq_len, d_model)` | Position information |
| Combined Input | `(seq_len, d_model)` | Embeddings + Position |
| Query/Key/Value | `(seq_len, d_k)` or `(seq_len, d_v)` | Per-head dimensions |
| Attention Output | `(seq_len, d_model)` | After concatenation and projection |
| FFN Hidden | `(seq_len, d_ff)` | Expanded dimension (usually 4×d_model) |
| Final Output | `(seq_len, d_model)` | Ready for next encoder block |

---

## Training Considerations

### 1. **Gradient Flow**
- Residual connections help maintain gradient flow
- Layer normalization stabilizes training
- Proper initialization is crucial

### 2. **Computational Complexity**
- Self-attention: O(n²) where n is sequence length
- FFN: O(n × d_model × d_ff)
- Memory usage scales with sequence length

### 3. **Hyperparameters**
- `d_model`: Model dimension (typically 512, 768, or 1024)
- `d_ff`: Feed-forward dimension (typically 4×d_model)
- `h`: Number of attention heads (typically 8 or 16)
- `d_k`, `d_v`: Per-head dimensions (d_model/h)

---

## Summary

The Transformer Encoder processes data through a sophisticated pipeline:

1. **Input Processing**: Tokens → Embeddings + Positional Encoding
2. **Self-Attention**: Captures relationships between all positions
3. **Residual + Norm**: Stabilizes training and maintains information flow
4. **Feed-Forward**: Adds non-linearity and capacity
5. **Final Residual + Norm**: Prepares output for next layer

This architecture enables the model to:
- Process sequences in parallel (unlike RNNs)
- Capture long-range dependencies effectively
- Scale to very deep architectures
- Maintain stable training dynamics

The key innovation lies in the self-attention mechanism, which allows each position to attend to all other positions, making it possible to capture complex relationships regardless of distance in the sequence.

