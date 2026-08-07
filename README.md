# Quora Duplicate Question Detection

Detecting whether two Quora questions are asking the same thing — a semantic similarity /
duplicate detection task, using the classic
[Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs) (~400k question
pairs, ~37% labeled duplicate).

## Approach

The project moves through five stages, each addressing a limitation found in the one before it:

1. **Random Forest baseline** — hand-engineered features (word overlap, length stats,
   fuzzy-matching ratios via `fuzzywuzzy`) combined with bag-of-words text vectors.
2. **Siamese LSTM (small sample, from-scratch embeddings)** — two shared-weight LSTM branches
   encode each question separately, merged via absolute difference.
3. **Concatenated Bidirectional LSTM (full dataset)** — both questions joined into one string,
   tokenized, and passed through a single recurrent model, trained on the full ~400k pairs.
4. **Random Forest + Word2Vec** — engineered features combined with mean-pooled Word2Vec
   sentence vectors.
5. **Sentence-BERT (frozen) + neural classifier** — pretrained transformer sentence embeddings
   (`all-MiniLM-L6-v2`), no fine-tuning, with a small feedforward classifier on top.

## Results

All models evaluated on the same untouched, stratified, class-imbalanced test set (never
oversampled) for a fair comparison.

| Model | Accuracy | F1 (duplicate) | Recall (duplicate) |
|---|---|---|---|
| RF — engineered features only | 0.719 | 0.617 | 0.62 |
| RF — engineered + fuzzy + BOW(3000) | 0.778 | 0.681 | 0.65 |
| Siamese LSTM, from-scratch embeddings (30k sample) | 0.729 | 0.640 | 0.66 |
| Bidirectional LSTM, full dataset | 0.773 | 0.708 | 0.75 |
| RF + Word2Vec + engineered features, full dataset | 0.820 | 0.756 | 0.75 |
| **SBERT (frozen) + NN classifier, full dataset** | **0.869** | **0.828** | **0.85** |

The final model clearly wins on every metric — nearly 5 points of accuracy over the best
classical model, and recall on duplicates up to 0.85 from 0.75. That recall number matters most
in practice: it's the difference between a duplicate-detection system that actually catches
most duplicates versus one that lets a quarter of them slip through.

## Key findings

- **Feature importance analysis** on the RF model showed all 14 top-ranked features were
  hand-engineered (fuzzy ratios, word overlap) — none were individual bag-of-words tokens. BOW's
  contribution came from aggregating thousands of weak signals, not any single strong one.
- **From-scratch embeddings overfit badly on small data.** A Siamese LSTM trained on only 30k
  samples reached 99% train accuracy but plateaued at 73% validation accuracy — classic
  overfitting from too many trainable parameters (the embedding layer) relative to data size.
- **A classical model can beat deep learning if its embeddings are weak.** RF + Word2Vec +
  engineered features (0.820 accuracy) beat every deep learning model trained with small,
  from-scratch embeddings. This wasn't "trees beat neural nets" — it was "hand-crafted features
  beat a poorly-informed embedding layer." Worth stating explicitly rather than concluding
  deep learning "didn't work" for this task.
- **Pretrained embeddings closed the gap decisively.** Swapping in frozen Sentence-BERT
  embeddings — trained on far more data than Word2Vec ever saw — pushed accuracy to 0.869 and
  F1 to 0.828, beating the RF baseline outright. This confirms the earlier RF win was about
  embedding quality, not modeling approach.
- **A real bug was caught and fixed during development:** an earlier reference implementation
  oversampled (SMOTE / RandomOverSampler) *before* splitting into train/test, which let
  duplicated synthetic rows leak across both sets and inflated reported accuracy. Fixed by always
  splitting first and oversampling only the training set.

## Repo structure

- `quora_kaggle.ipynb` — Word2Vec + RF baseline, RNN, Bidirectional LSTM pipeline
- `quora_sbert.ipynb` — Sentence-BERT pipeline (best-performing model)
- `lstm_duplicate_model.keras`, `sbert_classifier.keras`, `tokenizer.pkl` — trained models for
  inference
- `README.md` — this file

Earlier exploratory notebooks (initial RF-only baseline, early Siamese LSTM experiments) are not
included here — their results are summarized in the table above, and their approaches were
either superseded or folded into `quora_kaggle.ipynb`.

## Running it

Built to run on Kaggle Notebooks or Google Colab (auto-detects dataset path). Add the Quora
Question Pairs dataset, enable GPU + internet access, and run top to bottom. Expensive steps
(cleaning, Word2Vec, RF, RNN, LSTM, SBERT encoding) are checkpointed to disk, so an interrupted
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

- Recall on duplicates (0.85) is strong but not perfect — some heavily reworded paraphrases are
  still missed.
- I started fine-tuning a CrossEncoder (`distilbert-base-uncased`) for a potential further push —
  it jointly attends over both questions rather than encoding them independently, which is
  usually the strongest architecture for this kind of pair classification. Didn't finish this
  given the frozen SBERT model already performed well, but it's a natural next step if chasing
  further gains.
- A lightweight demo (Streamlit/Gradio) would make the model interactively testable without
  running the notebook.
