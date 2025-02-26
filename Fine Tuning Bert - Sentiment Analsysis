# Fine-Tuning BERT for Sentiment Classification

## Introduction

BERT (Bidirectional Encoder Representations from Transformers) is a powerful pretrained model that can be fine-tuned for text classification tasks, such as sentiment analysis. This guide walks through the entire process of fine-tuning BERT for a binary sentiment classification task (positive/negative sentiment).

---

## 📌 1. Sample Dataset

We use a small sample dataset containing movie reviews labeled as **positive (1)** or **negative (0)**.

| Review                               | Label |
| ------------------------------------ | ----- |
| The movie was absolutely fantastic!  | 1     |
| I hated every minute of it.          | 0     |
| A wonderful story with great acting. | 1     |
| It was boring and way too long.      | 0     |
| The visuals were stunning!           | 1     |

---

## 📝 2. Tokenization

BERT requires text input to be tokenized before processing. We use the **Hugging Face Tokenizer** to achieve this.

### **Example Sentence:**

```plaintext
"The movie was absolutely fantastic!"
```

### **Tokenized Output:**

```plaintext
['[CLS]', 'the', 'movie', 'was', 'absolutely', 'fantastic', '!', '[SEP]']
```

### **Token IDs:**

```plaintext
[101, 1996, 3185, 2001, 7078, 7965, 999, 102]
```

- **[CLS]**: Special token at the start of the sequence.
- **[SEP]**: Special token marking the end of the sequence.
- Token IDs represent words in BERT's vocabulary.

---

## 🔍 3. Attention Masks

BERT uses an **attention mask** to differentiate between **actual tokens (1)** and **padding tokens (0)**.

**Example Attention Mask:**

```plaintext
[1, 1, 1, 1, 1, 1, 1, 1]
```

If padding were needed:

```plaintext
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # Padding tokens (0s) added
```

---

## 🏗️ 4. Model Architecture for Fine-Tuning

### **How BERT Processes the Input:**

1. The tokenized sentence is passed through **BERT’s transformer layers**.
2. The **final hidden state of [CLS]** represents the entire sentence.
3. A **fully connected classification layer** is added on top of `[CLS]`.
4. The model outputs logits, which are converted into **probability scores** using Softmax.

### **Mathematical Representation:**

```plaintext
z = W * [CLS] + b  # Fully connected layer output
```

- `W`: Weight matrix (768 × number of classes)
- `b`: Bias term

### **Example Logits Output (Binary Classification):**

```plaintext
[-0.85, 2.3]
```

Applying Softmax:

```plaintext
Probability (Positive) = exp(2.3) / [exp(-0.85) + exp(2.3)] ≈ 0.91
Probability (Negative) = exp(-0.85) / [exp(-0.85) + exp(2.3)] ≈ 0.09
```

Since the probability for **Positive** is higher, the prediction is **Positive**.

---

## 🎯 5. Training the Model

### **Loss Function:**

We use **Cross-Entropy Loss**, which is ideal for classification tasks.

### **Training Process:**

1. Compute **loss** using the softmax output and true labels.
2. **Backpropagate** to update BERT’s weights using gradient descent.
3. **Fine-tune BERT’s parameters** to improve classification performance.

---

## 🔮 6. Making Predictions

After fine-tuning, we can use the model for inference:

**Example Input:**

```plaintext
"I really enjoyed the movie!"
```

**Tokenized Input:**

```plaintext
['[CLS]', 'I', 'really', 'enjoyed', 'the', 'movie', '!', '[SEP]']
```

**Model Output:**

```plaintext
Positive (Confidence: 0.95)
```

---

## 🔄 7. Summary of Fine-Tuning Steps

1. **Prepare the dataset** with text and labels.
2. **Tokenize sentences** using BERT tokenizer.
3. **Create attention masks** to identify valid tokens.
4. **Pass input through BERT** to get `[CLS]` representation.
5. **Add a classification head** (fully connected layer) on top.
6. **Train the model** using cross-entropy loss.
7. **Fine-tune BERT’s parameters** using gradient descent.
8. **Make predictions** on new text inputs.

---

## 🚀 Final Thoughts

By fine-tuning BERT, we leverage its deep contextual understanding of text for sentiment classification. This approach can be extended to other classification tasks, such as spam detection, topic categorization, and more.

🎉 **Now you're ready to fine-tune BERT for text classification!**

