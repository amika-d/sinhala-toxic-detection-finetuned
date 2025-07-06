# Sinhala Toxic Comment Classification using XLM-RoBERTa

This project demonstrates the fine-tuning of the multilingual transformer model `xlm-roberta-large` for classifying Sinhala social media comments as toxic or non-toxic.

---

## 🚀 Project Overview

- **Objective**: Build a robust classifier to detect toxic comments in Sinhala, a low-resource language.
- **Model**: `xlm-roberta-large`
- **Dataset**: Custom-labeled Sinhala comments scraped from social platforms
- **Output**: Binary classification (Toxic = 1, Non-toxic = 0)

---

## 🧠 Key Features

- Tokenization analysis to choose optimal sequence length
- Basic data augmentation using `[MASK]` token injection
- Early stopping for training efficiency
- Mixed-precision (`fp16`) training for GPU speedup
- Custom dataset class with HuggingFace `Trainer` API
- Evaluation with F1-score, confusion matrix, and classification report

---

## 📁 Folder Structure

```
.
├── cleaned_sinhala_comments.csv       # Preprocessed dataset
├── fine_tuned_sinhala_model/         # Saved tokenizer and model
├── sinhala_toxic_detection.ipynb     # Full notebook with code
├── logs/                             # Training logs
└── model/                            # Output directory during training
```

---

## ⚙️ Setup Instructions

```bash
# Clone the repository
$ git clone https://github.com/yourusername/sinhala-toxic-comment-classifier.git
$ cd sinhala-toxic-comment-classifier

# Install required packages
$ pip install -r requirements.txt
```

---

## 🧪 Training Configuration

```python
TrainingArguments(
    output_dir='./model',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=True,
    eval_strategy="steps",
    save_strategy="steps",
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    seed=42
)
```

---

## 📊 Evaluation Metrics

- **Accuracy**, **F1-score**, **Precision**, **Recall**
- Confusion Matrix using `matplotlib` + `seaborn`
- Classification Report with `sklearn`

---

## 🔍 Example Predictions

```python
Text: උඹ වගේ ගොන්ජයෙක්ට කොහෙද මෙච්චර හිතෙන හැටි
Prediction: Toxic (1) | Confidence: 0.94

Text: 33ක් වගේම හොඳයි, peaceful vibe එකක් තියෙනවා
Prediction: Non-Toxic (0) | Confidence: 0.89
```

---

## 🌍 Multilingual & Inclusivity Impact

This project tackles the challenge of Sinhala NLP using a multilingual transformer model, addressing:

- Language inclusivity
- Toxic speech moderation in underrepresented communities
- Transformer-based adaptation for low-resource languages

---

## 🛠️ Optimizations

- Mixed-precision (`fp16`) training for faster convergence
- Token length truncation based on 95th percentile analysis
- Model evaluation pipeline ready for edge deployment

---

## 📈 Future Improvements

- Integrate with WandB for experiment tracking
- Quantize model for mobile/edge deployment
- Use LoRA or DistilRoBERTa for smaller variants
- Incorporate advanced augmentation (e.g., back-translation)

---

## 📌 Related Skills

- HuggingFace Transformers
- PyTorch
- NLP for low-resource languages
- Model interpretability
- Text classification pipeline

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

If you have any questions or want to collaborate:

- GitHub: [yourusername](https://github.com/yourusername)
- Email: [yourname@example.com](mailto\:yourname@example.com)

---



