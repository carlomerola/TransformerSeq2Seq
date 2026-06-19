# Transformer Seq2Seq — Sentence Reordering

A from-scratch **Transformer (encoder–decoder)** built in Keras/TensorFlow that takes a
**random permutation of an English sentence** as input and reconstructs the **original
word order**.


> The Transformer was chosen over an LSTM seq2seq because attention can capture
> dependencies between tokens **regardless of their distance in the sequence** and
> trains with far more parallelism.

The full implementation, training trace and evaluation live in
[Carlo_Merola_14_06_24_Sentence_Reordering_spec_4_256_1024_8_02.ipynb](Carlo_Merola_14_06_24_Sentence_Reordering_spec_4_256_1024_8_02.ipynb).
Trained weights are in [transformer_teachforce_4_256_1024_8_02.h5](transformer_teachforce_4_256_1024_8_02.h5)
(the filename encodes the config: *teacher-forcing, 4 layers, d_model 256, dff 1024, 8 heads, 0.2 dropout*).

---

## Problem & constraints

Given a shuffled bag of words, output the correctly ordered sentence. Output may be produced
in a single shot or autoregressively, one token at a time.

The exam imposed hard constraints, all of which are respected here:

| Constraint | This project |
|---|---|
| No pretrained models | Trained from scratch |
| < 20M parameters | **≈ 12.5M parameters** |
| No post-processing (e.g. no beam search) | Greedy autoregressive decoding only |
| No additional training data | Uses only the provided `generics_kb` corpus |

A bonus is awarded for low parameter counts, which motivated the parameter-saving choices below
(shared embeddings, a compact 4-layer model).

---

## Dataset & preprocessing

Source: the `generics_kb` dataset from Hugging Face, restricted to the **10K most frequent words**.

Preprocessing pipeline (notebook cells 7–25):

1. **Filter** out sentences with ≤ 8 words.
2. **Augment markup**: wrap each sentence with `<start>` / `<end>` tokens and replace every comma
   with a dedicated `<comma>` token.
3. **Tokenize** with a Keras `TextVectorization` layer (`max_tokens=10000`,
   `lower_and_strip_punctuation`). The vocabulary is ordered most- to least-frequent, so the
   special tokens land at known indices used throughout the code:

   | index | 0 | 1 | 2 | 3 | 7 |
   |---|---|---|---|---|---|
   | token | *(padding)* | `[UNK]` | `<end>` | `<start>` | `<comma>` |

4. **Detokenizer** — a custom `TextDetokenizer` class maps indices back to words, restoring the
   special tokens and dropping padding.
5. **Drop noisy rows**: any sentence containing an `[UNK]` token (index 1) is removed.
6. **Pad** every sentence to a fixed length of **28 tokens** → final tensor shape **(241236, 28)**.

### On-the-fly shuffling — `DataGenerator`

A custom `keras.utils.Sequence` (`DataGenerator`) produces `(shuffled_input, ordered_target)`
batches. For each sentence it copies the ordered version, then shuffles **only the real word
positions** — `<start>`, `<end>` and padding are left in place
(`np.random.shuffle(data_batch[i, 1:argmin-1])`). This means new permutations are seen across
epochs rather than a single fixed shuffle.

The data is split into **~220K training / ~21K test** sentences.

---

## Evaluation metric

Quality is measured by the **longest common substring ratio** between the source `s` and the
prediction `p`:

```
score = |longest_common_substring(s, p)| / max(|s|, |p|)
```

implemented with `difflib.SequenceMatcher(...).find_longest_match()`. `<start>`/`<end>` tokens are
excluded before scoring, and the reported figure is the mean over the whole test set
(well above the required 3K examples).

**Baseline** — scoring the raw shuffled inputs against their targets gives a mean of only
**0.1585**, which the trained model must beat.

---

## Model architecture

A standard Transformer encoder–decoder, implemented as a set of custom Keras layers. Total
parameters: **12,507,920 (≈ 12.5M)**.

| Block | Params |
|---|---|
| `TokenAndPositionEmbedding` (shared) | 2,567,168 |
| `Encoder` (4 layers) | 3,159,040 |
| `Decoder` (4 layers) | 4,211,712 |
| Final `Dense` → vocab (softmax) | 2,570,000 |

### Implemented components

**`TokenAndPositionEmbedding`**
- Token embedding + **learned** positional embedding (added together). Position information is
  essential here — without it the model would treat the input as a bag of words, which is exactly
  what we are trying to reorder.
- A sinusoidal positional-encoding method is also implemented as an alternative.
- **Shared between encoder and decoder** to cut the parameter count.
- `mask_zero=True` generates a **padding mask**; `compute_mask` is overridden so the mask
  propagates automatically to every downstream layer that supports it.

**`AttentionMechanism`** — one class wrapping the three attention modes used in the network,
each followed by a residual `Add` and `LayerNormalization`:
- **Self-attention** (encoder): the padding mask from the embedding layer is applied
  automatically, so attention only ever looks at meaningful tokens.
- **Causal self-attention** (decoder): receives a **causal mask** so each position can attend only
  to itself and earlier positions — it can't peek at future tokens. The code also supports
  **combining the causal mask with the padding mask** (`combined_mask = padding_mask * causal_mask`)
  when both are supplied.
- **Cross-attention** (decoder → encoder): queries come from the decoder, keys/values from the
  encoder output, letting the decoder focus on the relevant parts of the input sentence.

**Causal mask — `get_causal_attention_mask`**
Builds a lower-triangular matrix (`i >= j`) and tiles it across the batch. For a length-4 sequence:

```
pos 1 → [1,0,0,0]
pos 2 → [1,1,0,0]
pos 3 → [1,1,1,0]
pos 4 → [1,1,1,1]
```

**`PositionWiseFeedForward`**
Two dense layers (`d_model → dff → d_model`, ReLU in between) with **GlorotUniform** initialization
to limit vanishing/exploding gradients, followed by dropout, a residual connection and layer norm.

**`Encoder`** — `num_layers` stacked blocks, each = self-attention + position-wise FFN, with dropout
on the (already embedded) input.

**`DecoderLayer`** — computes its causal mask, then runs *causal self-attention → cross-attention →
FFN*.

**`Decoder`** — `num_layers` stacked `DecoderLayer`s, consuming the target embeddings and the encoder
output.

**`Transformer`** — ties it together: shared embedding → encoder → decoder → final `Dense(vocab,
softmax)` producing per-token probabilities.

### Hyperparameters

| `num_layers` | `d_model` | `dff` | `num_heads` | `dropout` | `maxlen` | `vocab_size` |
|---|---|---|---|---|---|---|
| 4 | 256 | 1024 | 8 | 0.2 | 28 | 10,000 |

---

## Training

**Teacher forcing.** At each decoder step the ground-truth previous token is fed in. The dataset is
pre-shifted by one position and cached locally: `dec_in = ordered[:, :-1]`,
`target = ordered[:, 1:]`. Saving the shifted dataset locally also saved compute on Colab.

**Masked loss & accuracy.** Both metrics mask out **padding (0)** *and* the **`<start>` token (3)**
so only meaningful predictions contribute:
- `masked_loss`: sparse categorical cross-entropy, summed over unmasked positions / count of
  unmasked positions.
- `masked_accuracy`: token-level accuracy over the same unmasked positions.

**Learning-rate schedule.** The classic Transformer warm-up schedule (`CustomSchedule`):
`lr = d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)` with 4000 warm-up steps, used with
**Adam** (`β1=0.9, β2=0.98, ε=1e-9`).

**Callbacks.** `ModelCheckpoint` (save best weights on `val_masked_accuracy`) and `EarlyStopping`
(patience 3, restore best weights).

**Multi-phase schedule.** Accuracy was prioritized over raw validation loss, because higher token
accuracy means more words land in the right place (which is what the substring metric rewards):

| Phase | Epochs | Batch size | Final `val_masked_accuracy` |
|---|---|---|---|
| 1 | 16 | 256 | ≈ 0.841 |
| 2 | +10 | 256 | ≈ 0.847 |
| 3 | up to 30 (early-stopped) | 1024 | **≈ 0.857** |

Training accuracy reached ≈ 0.986; validation accuracy plateaued around 0.857 even as validation
loss crept up — accepted on purpose for the reason above.

---

## Inference

Greedy, autoregressive decoding (no beam search, per the constraints):

- **`predict`** — single sequence: start from `[<start>]`, repeatedly take `argmax` of the last
  step's distribution, append it, and stop at `<end>` (or `maxlen`).
- **`predict_batches`** — the same loop vectorized over a batch for fast evaluation.
- **`calculate_scores`** — runs batched prediction over the test set, strips `<start>`/`<end>`, and
  averages the substring metric.

---

## Results

**Mean score: 0.5488** over **21,216** test sentences (vs. the **0.1585** shuffled baseline).

Example predictions:

| Target | Prediction | Score |
|---|---|---|
| predators like to eat banana slugs at all stages of their lives | predators like to eat banana slugs at all stages of their lives | **1.00** |
| recycling prevents pollution and helps conserve precious natural resources | recycling helps conserve precious natural resources and prevents pollution | 0.57 |
| snails are hermaphrodites but they have to mate before laying some days later | some snails have hermaphrodites but they are laying to mate later days before | 0.34 |

### Other approaches explored

- **Deep & narrow** (`d_model=128`, `dff=512`, **10 layers**, ≈ 7M params): score **0.5125** — a strong
  result for the parameter budget.
- **Deeper at the same width**: tested but kept at 4 encoder/decoder layers to avoid inflating the
  parameter count.

---

## Repository layout

| File | Description |
|---|---|
| [Carlo_Merola_14_06_24_Sentence_Reordering_spec_4_256_1024_8_02.ipynb](Carlo_Merola_14_06_24_Sentence_Reordering_spec_4_256_1024_8_02.ipynb) | Full notebook: data, model, training trace, evaluation |
| [transformer_teachforce_4_256_1024_8_02.h5](transformer_teachforce_4_256_1024_8_02.h5) | Trained weights (best `val_masked_accuracy`) |

Backend: TensorFlow / Keras. The notebook was developed on Google Colab (mounts Drive and
`pip install datasets`).
