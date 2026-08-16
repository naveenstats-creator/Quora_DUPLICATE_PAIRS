# Quora Duplicate Question Detection

Detecting whether two Quora questions are asking the same thing — a semantic similarity /
duplicate detection task, using the classic
[Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs) (~404k question
pairs, ~37% labeled duplicate).

## Approach

Two notebooks, each building on the last:

**`quora_dl.ipynb`** — classic ML baseline, then a first pass at deep learning:
1. Clean text (lowercase, remove stopwords, stem)
2. Train a **Word2Vec** model on the question corpus, mean-pool word vectors into one vector per
   question
3. Add a few simple hand-crafted features (word-length difference, common-word count)
4. **Random Forest** on the combined features — the baseline to beat
5. Concatenate both questions into a single string, tokenize, and feed through a **SimpleRNN**,
   then a **Bidirectional LSTM**

**`quora_sbert.ipynb`** — swapping in pretrained transformer embeddings instead of Word2Vec:
1. Encode every question with pretrained **Sentence-BERT** (`all-MiniLM-L6-v2`, frozen, no
   fine-tuning)
2. Build features from each embedding pair (concatenation, absolute difference, cosine
   similarity)
3. Train a small feedforward neural network classifier on top
4. Started (but didn't finish) fine-tuning a CrossEncoder for a further push — see notes below

Both oversample (SMOTE / RandomOverSampler) only on the training set, **after** splitting, to
avoid leaking duplicated rows into the test set — an issue I caught and fixed early on, since an
initial reference implementation I was following did this the other way around.

## Results

All models evaluated on the same untouched, stratified, class-imbalanced test set (never
oversampled) for a fair comparison.

| Model | Accuracy | F1 (duplicate) | Recall (duplicate) |
|---|---|---|---|
| Random Forest (Word2Vec + engineered features) | 0.821 | 0.757 | 0.76 |
| SimpleRNN (concatenated questions) | 0.744 | 0.665 | 0.69 |
| Bidirectional LSTM (concatenated questions) | 0.755 | 0.703 | 0.78 |
| **SBERT (frozen) + NN classifier** | **0.869** | **0.828** | **0.85** |

SBERT is the clear winner across every metric. The RF baseline came in ahead of both the
SimpleRNN and LSTM — worth understanding why rather than treating it as a fluke (see below).

## Key findings

- **The RF baseline beat the from-scratch RNN/LSTM models**, and I think that's mostly about
  embedding quality, not modeling power. The RF got Word2Vec vectors *plus* explicit
  length/overlap features handed to it directly. The RNN and LSTM had to learn what words even
  mean from a 32-dimensional embedding layer trained from random initialization, using only the
  duplicate/not-duplicate label as signal — a much harder problem with a lot less data to learn
  from than Word2Vec had.
- **Bidirectional LSTM needed more training than I first gave it.** An initial run stopped early
  (2 epochs) while val_loss was still dropping — retraining with a longer patience setting on
  `EarlyStopping` closed some of the gap (F1 improved from 0.673 to 0.703, recall from 0.69 to
  0.78), which suggests the concatenated-RNN approach had more headroom than the first run showed.
- **Pretrained Sentence-BERT embeddings closed the rest of the gap and then some.** Once I
  swapped in `all-MiniLM-L6-v2` — pretrained on far more text than Word2Vec ever saw — the same
  "frozen embeddings + small classifier" pattern jumped to 0.869 accuracy / 0.828 F1, beating the
  RF outright. This confirms the RF's earlier win was about embedding quality, not about tree
  ensembles being fundamentally better suited to this task.
- **A real train/test leakage bug was caught and fixed during development.** A reference
  implementation I initially followed oversampled *before* splitting into train/test, letting
  duplicated synthetic rows land on both sides of the split and inflate reported accuracy. Both
  notebooks here split first and oversample only the training data.

## Repo structure

- `quora_dl_beginner.ipynb` — Word2Vec + RF baseline, SimpleRNN, Bidirectional LSTM
- `quora_sbert.ipynb` — Sentence-BERT pipeline (best-performing model)
- `sbert_classifier.keras`, `lstm_duplicate_model.keras`, `tokenizer.pkl` — trained models saved
  for inference

## Running it

Both notebooks auto-detect the dataset path (works on Kaggle or Colab) — just add/upload the
Quora Question Pairs `train.csv`, enable GPU, and run top to bottom. Expensive steps (Word2Vec
training, RF fit, SBERT encoding, model training) are checkpointed to disk, so an interrupted
session can resume without recomputing finished steps.

## Inference example

```python
predict_duplicate_sbert(
    "What are some common examples of solid matter?",
    "What are the most common examples of solid matter?"
)
# -> ("DUPLICATE", 0.91)
```

## Limitations & possible next steps

- Recall on duplicates (0.85 with SBERT) is strong but not perfect — some heavily reworded
  paraphrases are still missed.
- I started fine-tuning a CrossEncoder (`distilbert-base-uncased`) in `quora_sbert.ipynb` — it
  jointly attends over both questions rather than encoding them independently, which is usually
  the strongest architecture for this kind of pair classification. Didn't finish it given the
  frozen SBERT model already performed well, but it's a natural next step for further gains.
- A lightweight demo (Streamlit/Gradio) would make the model interactively testable without
  running the notebook.
