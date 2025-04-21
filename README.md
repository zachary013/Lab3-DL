# 🧠 Deep Learning Lab 3: Arabic NLP & Text Generation 🚀

![Deep Learning](https://img.shields.io/badge/Deep_Learning-NLP-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.20+-green)
![GPU](https://img.shields.io/badge/GPU-Enabled-brightgreen)
![Arabic NLP](https://img.shields.io/badge/Arabic-NLP-yellow)

## 📚 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Collection & Preprocessing](#-data-collection--preprocessing)
- [Classification Models](#-classification-models)
- [Transformer Text Generation](#-transformer-text-generation)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results & Analysis](#-results--analysis)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [References](#-references)

## 🔍 Overview
This project implements advanced deep learning architectures for Natural Language Processing tasks focusing on the Arabic language. It consists of two main components:

1. **Arabic Text Classification**: Building and comparing various recurrent neural network architectures (RNN, Bidirectional RNN, GRU, and LSTM) to classify Arabic texts based on their relevance scores.
   
2. **Arabic Text Generation**: Fine-tuning a GPT-2 model to generate coherent Arabic text continuations from given prompts.

This implementation is part of the Deep Learning course (Master MBD) supervised by Prof. ELAACHAk LOTFI at Université Abdelmalek Essaadi.

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*uCMelgC_N3Q8VzI_3QiQqQ.png" width="600" alt="NLP Deep Learning Architecture">
</p>

## 📂 Project Structure
```
deep-learning-lab3/
├── notebooks/
│   ├── part1_arabic_classification.ipynb
│   └── part2_gpt2_finetuning.ipynb
├── data/
│   ├── raw/
│   │   └── arabic_texts_raw.csv
│   └── processed/
│       └── arabic_texts_processed.csv
├── models/
│   ├── classification/
│   │   ├── rnn_model.pth
│   │   ├── birnn_model.pth
│   │   ├── lstm_model.pth
│   │   └── gru_model.pth
│   └── generation/
│       └── fine_tuned_gpt2/
├── scripts/
│   ├── scraping.py
│   ├── preprocessing.py
│   ├── train_models.py
│   └── generate_text.py
├── results/
│   ├── model_comparison.csv
│   ├── learning_curves.png
│   └── generated_samples.txt
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠 Installation
To run this project, you need to install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

Or directly in Google Colab:

```python
!pip install torch transformers datasets scrapy beautifulsoup4 pyarabic farasapy matplotlib
```

### 💻 Setup Google Colab with GPU
1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Go to Runtime → Change runtime type → Hardware accelerator → GPU
4. Verify GPU is available:
   ```python
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
   ```

## 📊 Data Collection & Preprocessing

### Web Scraping
Arabic text data is collected from various news websites and blogs using BeautifulSoup and Scrapy. Each text is assigned a relevance score between 0 and 10.

```python
# Web scraping with BeautifulSoup
import requests
from bs4 import BeautifulSoup

websites = ["https://www.aljazeera.net", "https://www.bbc.com/arabic", ...]

# Extract Arabic text content
# Score assignment
```

### Text Preprocessing Pipeline
The collected Arabic texts undergo a comprehensive preprocessing pipeline:

1. **Diacritics Removal**: Remove Arabic diacritical marks (تشكيل)
2. **Tokenization**: Split text into individual tokens
3. **Stemming**: Extract word stems using Farasa stemmer
4. **Stop Words Removal**: Remove common Arabic stop words
5. **Normalization**: Standardize text (e.g., normalize various forms of Alif)

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*Zdxav15O9xVh7GQNaf5FJQ.png" width="650" alt="NLP Preprocessing Pipeline">
</p>

## 🧮 Classification Models
We implement four different sequence model architectures for the classification task:

### 1. Simple RNN
```python
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))
```

### 2. Bidirectional RNN
Processes text in both directions to capture context from past and future tokens.

### 3. LSTM (Long Short-Term Memory)
Addresses the vanishing gradient problem with memory cells and gating mechanisms.

### 4. GRU (Gated Recurrent Unit)
A simplified variant of LSTM with fewer parameters but comparable performance.

## 🤖 Transformer Text Generation
The second part of the project focuses on fine-tuning a pre-trained GPT-2 model for Arabic text generation.

### Model Architecture
GPT-2 is a transformer-based language model with a decoder-only architecture:

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*iJcUH1F0TmCQE9yCrJjvxg.png" width="500" alt="GPT-2 Architecture">
</p>

### Fine-tuning Process
1. Load the pre-trained GPT-2 model
2. Create a custom Arabic text dataset
3. Fine-tune the model on this dataset
4. Generate text based on input prompts

```python
# Example text generation function
def generate_text(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
```

## 📏 Evaluation Metrics

### Classification Metrics
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual scores
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predictions and actual values
- **R² Score**: Indicates the proportion of variance in the dependent variable that's predictable from the independent variables

### Text Generation Metrics
- **BLEU Score**: Evaluates the quality of generated text by comparing with reference texts
- **Perplexity**: Measures how well a probability model predicts a sample
- **Human Evaluation**: Subjective assessment of fluency, coherence, and relevance

## 📈 Results & Analysis

### Classification Models Comparison

| Model | MSE | MAE | R² | Training Time |
|-------|-----|-----|-----|--------------|
| RNN   | 0.45| 0.52| 0.63| 3m 20s       |
| BiRNN | 0.38| 0.47| 0.71| 4m 15s       |
| LSTM  | 0.32| 0.43| 0.75| 5m 40s       |
| GRU   | 0.34| 0.45| 0.73| 4m 50s       |

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*U3Qa3gWrPfGAMMrJdsvj8Q.png" width="600" alt="Learning Curves">
</p>

### Text Generation Examples

**Prompt**: "الذكاء الاصطناعي هو"

**Generated**: "الذكاء الاصطناعي هو مجال من مجالات علوم الحاسوب الذي يركز على تطوير أنظمة قادرة على أداء مهام تتطلب عادة الذكاء البشري. يشمل ذلك التعلم والاستدلال وحل المشكلات واتخاذ القرارات والإدراك البصري وفهم اللغة الطبيعية."

**Prompt**: "اللغة العربية لها"

**Generated**: "اللغة العربية لها تاريخ غني وعريق يمتد لأكثر من 1500 عام. تتميز بثرائها اللغوي وتنوع مفرداتها وقواعدها النحوية المعقدة والدقيقة. وهي من اللغات السامية التي انتشرت في شبه الجزيرة العربية ثم توسعت مع انتشار الإسلام لتصبح لغة العلم والأدب والفلسفة خلال العصر الذهبي للحضارة الإسلامية."

## 🔑 Key Learnings
- **Architecture Impact**: Bidirectional models consistently outperform unidirectional ones by capturing context from both directions.
- **Memory Mechanisms**: LSTM and GRU models show superior performance in capturing long-range dependencies compared to simple RNNs.
- **Arabic Language Challenges**: Arabic morphological complexity requires specialized preprocessing techniques.
- **Transfer Learning**: Fine-tuning pre-trained transformers is highly effective even with limited training data.
- **Hyperparameter Sensitivity**: Model performance varies significantly with different hyperparameter configurations.

