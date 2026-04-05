# Dataset Logs

## 1. Source

The dataset originates from [M-ABSA](https://github.com/megagonlabs/m-absa), a large multilingual resource for Aspect-Based Sentiment Analysis covering 21 languages and 7 domains. It was loaded using the Hugging Face `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("Multilingual-NLP/M-ABSA")
```

Only Korean sentences were used for this project. No purpose-built Korean restaurant ABSA dataset exists, so the M-ABSA corpus was filtered, restructured, and relabeled to fit the task.

> Full preprocessing pipeline, dataset distribution analysis, and visualizations are available in `preprocessing.ipynb` inside notebooks.

---
---

## 2. Original Structure

Each entry contains a review sentence and one or more aspect-sentiment triplets, separated by `####`:
```
Text #### [aspect term, aspect category, sentiment polarity]
```

**Example:**
```
The laptop battery life is disappointing. #### [battery life, hardware quality, negative]
```

| Element | Description |
|---|---|
| Aspect term | The specific entity being discussed |
| Aspect category | The general category of the aspect |
| Sentiment polarity | Sentiment toward the aspect (`positive` / `negative`) |

---

## 3. Filtering

### 3.1 Language
Filtered to Korean sentences only.

### 3.2 Delimiter Validation
Only entries containing `####` were kept to ensure the sentence and triplet annotations could be properly separated. Entries without the delimiter were removed.

### 3.3 Empty Triplets
Entries with no triplet annotations were removed as they provide no aspect-level sentiment information.

---

## 4. Triplet Parsing

Triplet strings were parsed into structured columns:
```
Sentence #### [aspect term, aspect category, sentiment]
```
↓
```
Sentence | Aspect Term | Aspect Category | Sentiment
```

---

## 5. Aspect Mapping

The original aspect categories are domain-general and not specific to restaurants. Relevant categories were manually mapped to four target aspects:

- **FOOD**
- **PRICE**
- **SERVICE**
- **AMBIENCE**

---

## 6. Sentiment Label Conversion

The original dataset contains only `positive` and `negative` polarities. These were converted to numerical labels, and a third label was introduced to represent aspects not discussed in a sentence:

| Label | Meaning |
|---|---|
| 0 | Not Mentioned |
| 1 | Negative |
| 2 | Positive |

---

## 7. Multi-Aspect Label Construction

In the original format, each sentence is associated with one triplet per annotation. Since a single sentence may mention multiple aspects simultaneously, the dataset was converted into a multi-aspect representation where each sentence holds labels for all four aspects:
```
[FOOD, PRICE, SERVICE, AMBIENCE]
```

**Example:**

| Sentence | FOOD | PRICE | SERVICE | AMBIENCE |
|---|---|---|---|---|
| 음식은 맛있지만 가격이 비싸다 | 2 | 1 | 0 | 0 |

---

## 8. Manual Verification

Converting from single-triplet to multi-aspect format required manual review to ensure correctness:

- All mapped aspect labels were manually verified
- Sentences mentioning multiple aspects were individually checked
- Labels were corrected where the automatic mapping was incorrect or ambiguous

---

## 9. AI-Generated Samples for Class Imbalance

After initial preprocessing, certain aspect-sentiment remained severely underrepresented — mostly negative labels for AMBIENCE, SERVICE and PRICE.

To address this, some additional samples were generated using an AI language model and saved as `manual_samples.csv`. These samples were:

- Written in natural Korean consistent with the style of the original reviews
- Targeted specifically at underrepresented aspect-sentiment combinations

> **Note:** AI-generated samples were used strictly to mitigate extreme class imbalance and not to inflate overall dataset size. They should be interpreted as a data augmentation strategy rather than ground-truth annotations.

---

## 10. Final Dataset Construction

The final dataset was produced by merging the preprocessed M-ABSA samples with the generated samples from `manual_samples.csv`:

Each entry in the final dataset is represented as:

| Column | Description |
|---|---|
| `review` | Korean review sentence |
| `FOOD` | Sentiment label |
| `PRICE` | Sentiment label |
| `SERVICE` | Sentiment label |
| `AMBIENCE` | Sentiment label |

**Label encoding:**
```
0 = Not Mentioned
1 = Negative
2 = Positive
```

## 11. Dataset Splitting

The dataset was split into three stratified subsets using `MultilabelStratifiedShuffleSplit` from the `iterative-stratification` library, which preserves label distribution across splits for multilabel classification problems — important given the class imbalance across aspects.

The split was performed in two steps: first separating 70% for training and 30% temporary, then splitting the temporary set equally into validation and test. `random_state=42` was used for reproducibility.