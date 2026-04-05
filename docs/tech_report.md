
# Benchmarking Classical ML, Transformer, and LLM Approaches for Korean ABSA

# Abstract

This project benchmarks three approaches to Aspect-Based Sentiment Analysis (ABSA) on Korean restaurant reviews:
- Classical ML **(TF-IDF + Logistic Regression)**  
- Transformer **(KcELECTRA)**  
- Small LLM via Ollama **(Qwen 2.5)** 

Each model is evaluated on per-aspect F1 scores, mention and sentiment F1, inference time, model size, and parameter count. 

A key challenge was the absence of a purpose-built Korean restaurant ABSA dataset, requiring adaptation of the multilingual M-ABSA corpus. 

Results show that KcELECTRA achieves the best overall performance, TF-IDF + LR offers a strong lightweight baseline, and Qwen 2.5 demonstrates advantages on complex multi-aspect and sarcastic reviews despite underperforming on standard metrics in its few-shot setting.

# 1. Introduction

Aspect-Based Sentiment Analysis (ABSA) identifies sentiment at the aspect level rather than assigning a single label to an entire review. 

For example, in the sentence:

> "The food was delicious, but the service was terrible."

A whole-sentence sentiment classifier would struggle, often labeling it as neutral or partially correct. ABSA, on the other hand, can assign:

- FOOD → Positive
- SERVICE → Negative

This fine-grained breakdown provides more actionable insights than sentence-level sentiment, allowing restaurants to pinpoint specific strengths and weaknesses. To achieve this, models must handle both mention detection (whether an aspect is discussed) and sentiment classification (positive, negative, or neutral). This project targets four aspects in Korean restaurant reviews: FOOD, PRICE, SERVICE, and AMBIENCE.

Three approaches are benchmarked:

- **TF-IDF + Logistic Regression** — lightweight and interpretable
- **KcELECTRA** — a transformer pretrained on Korean text
- **Qwen 2.5 LLM** — a 7B-parameter model 

The goal is to identify trade-offs between predictive performance, model size, and deployment practicality.

# 2. Methodology

## 2.1 Models Overview

### 2.1.1 TF-IDF + Logistic Regression

- **Architecture:** Bag-of-Words → Multi-output Logistic Regression  
- **Parameters:** ~200K  
- **Advantages:** Fast, lightweight, interpretable  
- **Limitations:** Cannot capture complex context or implicit sentiment  

**Implementation Details (2-Stage ABSA):**

The TF-IDF + LR model is implemented as a **two-stage pipeline**:


**Vectorization:**
- `TfidfVectorizer` with `analyzer="char_wb"` and character n-grams
- Character n-grams were chosen over word-level tokenization to handle Korean morphological variation without requiring an external tokenizer such as KoNLPy. Variants like 맛있어요 and 맛있었어 share overlapping n-grams, giving the model implicit morpheme-level signal.

1. **Stage 1 — Mention Detection:**  
   - Objective: Identify whether each aspect (FOOD, PRICE, SERVICE, AMBIENCE) is discussed in a review.  
   - Approach: A multi-output Logistic Regression classifier is trained on **binary labels** per aspect (1 = mentioned, 0 = not mentioned).  
   - This ensures that sentiment is predicted **only for aspects that are actually discussed**.

2. **Stage 2 — Sentiment Classification:**  
   - Objective: Predict sentiment (Negative = 1, Positive = 2) for aspects detected as mentioned.  
   - Approach: For each aspect, a separate Logistic Regression classifier is trained **only on the subset of reviews where that aspect is mentioned**.  
   - If an aspect has only one sentiment class in the training subset, the classifier stores it as a **fallback class** to avoid training errors.

**Inference Pipeline:**
```
Korean review text
       ↓
TF-IDF vectorization
       ↓
Stage 1: Mention detection
  (multi-output LR, binary per aspect)
       ↓
  mentioned? ── no ──→ assign 0
       │
      yes
       ↓
Stage 2: Sentiment classification
  (per-aspect LR on mentioned subset)
       ↓
Output: [FOOD, PRICE, SERVICE, AMBIENCE]
  (0 = not mentioned, 1 = neg, 2 = pos) 
  ```


### 2.1.2 KcELECTRA

- **Architecture:** Transformer (ELECTRA) pretrained on Korean text, with a task-specific ABSA head  
- **Parameters:** ~128M  
- **Advantages:** Captures context and long-range dependencies, handles multi-aspect sentences  
- **Limitations:** Requires GPU for efficient training and inference  

**Implementation Details:**

KcELECTRA is fine-tuned for ABSA in a **single-stage multi-output classification** framework:

1. **Input Representation:**
   - Each review is tokenized and converted to input IDs and attention masks.
   - The `[CLS]` token representation from the final encoder layer is used as the aggregate sequence embedding.
   - A dropout layer (p=0.1) is applied to the `[CLS]` embedding before the classification heads.

2. **Model Heads:**
   - **Mention head:** A single `Linear(hidden_size, num_aspects)` layer predicts whether each aspect is discussed. Trained with binary cross-entropy loss with positive class weighting to handle class imbalance.
   - **Sentiment heads:** One independent `Linear(hidden_size, 2)` per aspect predicts sentiment polarity (negative or positive). Trained with cross-entropy loss with per-class weights. Applied only when the aspect is mentioned.
   - Both heads share the same `[CLS]` representation from a single encoder forward pass.

3. **Loss Function:**
   - Mention loss: weighted multi-label BCE loss across all aspects.
   - Sentiment loss: weighted cross-entropy loss per aspect, computed only on mentioned aspects.
   - Total loss combines both terms to train the full model end-to-end.

4. **Output:**
   - Logits are packed into a `(B, 3A)` tensor: the first `A` columns are mention logits and the remaining `2A` columns are sentiment logits reshaped from `(B, A, 2)`.
   - At inference, mention predictions gate which sentiment heads are considered for the final output vector `[FOOD, PRICE, SERVICE, AMBIENCE]`

5. **Training Setup:**  
   - Fine-tuned on the ABSA dataset for `6` epochs.  
   - Optimizer: AdamW with learning rate `3e-5`.  
   - Batch size: `16`.  
   - Validation used for **threshold tuning**.

**Inference Pipeline:**
```
Korean review text
       ↓
Tokenizer
  (input IDs + attention mask)
       ↓
KcELECTRA encoder
  ([CLS] embedding, dropout p=0.1)
       ↓
       ├──→ Mention head: Linear(hidden, 4)
       │      (BCE loss + pos class weights)
       │             ↓
       │      mentioned per aspect?
       │        yes ↓       no → assign 0
       │
       └──→ Sentiment heads: Linear(hidden, 2) × 4
              (CE loss + class weights, one per aspect)
                     ↓
              applied only on mentioned aspects
                     ↓
Output: [FOOD, PRICE, SERVICE, AMBIENCE]
  (0 = not mentioned, 1 = neg, 2 = pos)
```

This setup allows KcELECTRA to **simultaneously detect mentions and sentiment** while leveraging the pretrained contextual embeddings to handle subtle or implicit sentiments.  

### 2.1.3 Qwen 2.5 LLM

- **Architecture:** 7B-parameter causal transformer (LLM) via Ollama  
- **Parameters:** ~7B  
- **Advantages:** Handles subtle, implicit, or sarcastic sentiments; multi-aspect awareness without fine-tuning  
- **Limitations:** Very large, slower inference, high memory requirements  

**Implementation Details:**

The Qwen 2.5 model uses a **prompt-based inference pipeline** for ABSA:

1. **Prompt Engineering:**  
   - Each review is converted into a structured prompt specifying the **four aspects**: FOOD, PRICE, SERVICE, AMBIENCE.  
   - Instructions clarify **label meanings**:  
     - `0 = Not Mentioned`  
     - `1 = Negative`  
     - `2 = Positive`
  

2. **Query Pipeline:**  
- Prompts are sent to the **local Qwen API** via HTTP POST requests.  
- Model outputs a text response containing predictions in the format `[FOOD, PRICE, SERVICE, AMBIENCE]`.

```
Korean reviews
       ↓
build_batch_prompt()
  (aspect labels + instructions)
       ↓
query_qwen(batch_size=n)
  (HTTP POST → Ollama local API)
       ↓
parse_batch_output()
  (regex extraction → fallback [0,0,0,0])
       ↓
Prediction safety check
       ↓
Evaluation
  [FOOD, PRICE, SERVICE, AMBIENCE]
  ```

- The LLM **does not require fine-tuning**

## 2.2 Dataset & Preprocessing

### 2.2.1 Dataset Overview

- **Dataset:** [M-ABSA](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA)

The M‑ABSA dataset is a large multilingual resource for Aspect‑Based Sentiment Analysis (ABSA).

It covers 21 languages and 7 domains, designed to support fine‑grained sentiment tasks such as extracting aspect terms, categories, and sentiment polarities

- **Task:** Aspect-Based Sentiment Analysis (ABSA) targeting four aspects: FOOD, PRICE, SERVICE, AMBIENCE  

- **Structure:** Each entry contains a text sentence and triplet labels `[aspect term, aspect category, sentiment polarity]` separated by `"####"`  

**Example Entry:**
- Text: "The laptop battery life is disappointing."
- Triplet: [battery life, hardware quality, negative]


### 2.2.2 Preprocessing Overview

- Filtered for **Korean sentences** only  
- Parsed triplets into structured columns and mapped to target aspects  
- Converted sentiment polarity to numerical labels: `0 = Not Mentioned`, `1 = Negative`, `2 = Positive`  
- **Additional preprocessing details and dataset modifications** are documented in `dataset_logs.md`
  

## 2.3 Experimental Setup

- Hardware: CPU/GPU  
- Metrics:  
  - Macro F1 per aspect  
  - Mention F1  
  - Sentiment F1  
  - Inference time 
  - Training time  
- The dataset was split into train (70%), validation (15%), and test (15%) sets using `MultilabelStratifiedShuffleSplit` to preserve aspect label distribution across all splits.

## 2.4 Evaluation Challenges

- Multi-aspect sentences  
- Class imbalance in some aspects
- Sarcasm & implicit sentiment  


# 3. Results Summary

### 3.1 Overall Performance

| Model          | Mention F1 | Sentiment F1 | Training Time (s) | Inference Time (s) |
|----------------|-----------|--------------|-----------------|------------------|
| TF-IDF + LR    | 0.91 | 0.51 | 1.01 | 0.05 |
| KcELECTRA      | 0.94 | 0.57 | 146 | 1.04 |
| Qwen 2.5 LLM   | 0.81 | 0.44 | 0 | 1801 |

- KcELECTRA achieves the highest mention and sentiment F1 scores, while TF-IDF + LR remains competitive. Qwen 2.5 underperforms. Its inference time of 1801s reflects sequential one-by-one processing of the entire test set  making it by far the slowest option in this evaluation setup.


### 3.2 Per-Aspect Performance(F1 scores) 

| Aspect     | TF-IDF + LR | KcELECTRA | Qwen 2.5 |
|------------|------------|-----------|-----------|
| FOOD       | 0.787 | 0.876 | 0.54 |
| PRICE      | 0.788 | 0.818 | 0.69 |
| SERVICE    | 0.783 | 0.891 | 0.77 |
| AMBIENCE   | 0.719 | 0.78 | 0.522 |

- KcELECTRA leads across all four aspects. Qwen 2.5 struggles most with FOOD and AMBIENCE, suggesting that few shot prompting without fine-tuning is insufficient for fine-grained aspect detection in Korean.

### 3.3 Efficiency & Model Complexity

| Model          | Parameters | Size on Disk | Inference Time (s) |
|----------------|-----------|--------------|------------------|
| TF-IDF + LR    | 192k | 2.36 MB |  0.05 |
| KcELECTRA      | 127M       | 485.28 MB       | 1.04 |
| Qwen 2.5 LLM   | 7B         | 4700 MB      | 1801 |

- TF-IDF + LR is over 20x faster than KcELECTRA and uses less than 0.5% of its disk space, making it suitable for low-resource or real-time deployments. 
- KcELECTRA offers the best performance-to-cost ratio for production use.

### 3.4 Performance on Hard Multi-Aspect Cases
A set of 29 challenging multi-aspect reviews(hard_example_set.csv) was evaluated to test model robustness on complex and implicit sentiment.

**Misclassified Aspects**

| Aspect    | LR | KcELECTRA | Qwen |
|-----------|----|-----------|------|
| FOOD      | 12 | 6         | 8    |
| PRICE     | 5  | 3         | 2    |
| SERVICE   | 4  | 4         | 3    |
| AMBIENCE  | 13 | 7         | 4    |

Qwen performs best overall on hard cases.

# 4. Challenges and Limitations

### 4.1 Dataset Availability
The most significant challenge in this project was the lack of a Korean restaurant review dataset specifically annotated for ABSA. No off-the-shelf dataset covering the four target aspects (FOOD, PRICE, SERVICE, AMBIENCE) in Korean existed, requiring the M-ABSA multilingual dataset to be filtered, restructured, and relabeled to fit the task. This introduced noise and limited the volume of usable training samples, particularly for underrepresented aspects like AMBIENCE.

### 4.2 Class Imbalance
The dataset exhibited significant class imbalance across both mention detection and sentiment classification. Not all aspects are discussed in every review, resulting in a large number of not-mentioned (0) labels, and negative sentiment instances were considerably fewer than positive ones across most aspects.

Each model handled this differently:

- **TF-IDF + LR** used `class_weight="balanced"` in scikit-learn, which automatically adjusts weights inversely proportional to class frequencies during training
- **KcELECTRA** used computed class weights for both the mention head and sentiment heads — a scalar `mention_pos_weight` for the BCE mention loss and per-class `sentiment_class_weights` for the cross-entropy sentiment loss, both derived from the training label distribution.
- **Qwen 2.5 LLM** has no explicit mechanism to handle class imbalance, which likely contributed to its lower performance on minority classes

### 4.3 Model-Specific Limitations
- **TF-IDF + LR** struggles with implicit sentiment and long sentences due to lack of context and word order
- **KcELECTRA** requires a GPU for efficient training and inference, and is sensitive to hyperparameter choices such as learning rate and batch size
- **Qwen 2.5 LLM** is resource-heavy and requires careful prompt design. Initial batch inference attempts produced incomplete outputs with one or two results collapsing to all zeros, likely due to the model losing track of multiple reviews in a single prompt. This required switching to one-review-at-a-time processing, which resolved the output quality issue but significantly increased total inference time

# 5. Future Improvements

### 5.1 Model Improvements
- Fine-tune Qwen 2.5 or a similarly sized LLM directly on the ABSA task to close the performance gap with KcELECTRA while retaining its ability to handle implicit and sarcastic sentiment
- Explore aspect-aware attention mechanisms on top of KcELECTRA to allow the model to attend differently per aspect rather than relying on a flat linear head

### 5.2 Dataset Expansion
- Build a native Korean restaurant review dataset to reduce noise
- Extend to multi-domain ABSA (movies, hotels, e-commerce, healthcare), enabling broader real-world applicability

### 5.3 Inference & Deployment
- Implement batch inference with structured JSON outputs for Qwen 2.5 to improve efficiency and consistency
- Optimize deployment via quantization, ONNX export, or FastAPI-based serving for faster and scalable inference

# 6. Application

The project includes an interactive web application (`app.py`) built with Streamlit, serving as the main entry point for the project. It provides:

- **Live review analyzer** — input a Korean restaurant review and get real-time aspect-level sentiment predictions from all three models
- **Results dashboard** — visualizations of model performance including per-aspect F1 scores, overall metrics, and efficiency comparisons

To run the application:
```bash
streamlit run app.py
```
 # 7. Conclusion

This project benchmarked three approaches to Aspect-Based Sentiment Analysis on Korean restaurant reviews, highlighting clear trade-offs between performance and computational cost.

- KcELECTRA achieved the best overall results with a mention F1 of 0.94 and sentiment F1 of 0.57, making it the most practical choice for production use. 
- TF-IDF + LR remained competitive at a fraction of the size and inference time, making it suitable for lightweight deployments.
- Qwen 2.5 scored lowest in the few-shot. However, it showed clear advantages on reviews containing multiple aspects or sarcastic expressions, where smaller models tend to fall short — suggesting that fine-tuning Qwen on the task could yield strong gains.

A key limitation across all models was the lack of a native Korean restaurant ABSA dataset, requiring adaptation of the M-ABSA corpus and likely constraining overall performance. This remains the most impactful area for future improvement. 