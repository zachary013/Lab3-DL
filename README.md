# 🧠 Deep Learning Lab 3: Arabic NLP & Text Generation 🚀

![Deep Learning](https://img.shields.io/badge/Deep_Learning-NLP-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.51+-green)
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
This project focuses on Natural Language Processing (NLP) tasks for the Arabic language, implemented in Google Colab with GPU support. It consists of two main parts:

1. **Arabic Text Classification**: We collect Arabic texts from news websites, preprocess them, and train various recurrent neural network models (Simple RNN, Bidirectional RNN, LSTM, GRU) to predict relevance scores.
2. **Arabic Text Generation**: We fine-tune a GPT-2 model on a small Arabic dataset to generate coherent text continuations from given prompts.



## 🛠 Installation
Run the project in Google Colab with the following dependencies:

```python
!pip install torch transformers datasets scrapy beautifulsoup4 pyarabic farasapy pandas numpy matplotlib scikit-learn
```

### 💻 Setup Google Colab with GPU
1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime** → **Change runtime type** → **Hardware accelerator** → **GPU**.
3. Verify GPU availability:
   ```python
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
   ```
   Expected output: GPU available (Tesla T4).

## 📊 Data Collection & Preprocessing

### Web Scraping
We scrape Arabic texts from news websites like Al Jazeera, BBC Arabic, and CNN Arabic using `requests` and `BeautifulSoup`. Each text is assigned a random relevance score (0-10) for classification.

```python
websites = [
    "https://www.aljazeera.net/news/politics",
    "https://www.bbc.com/arabic",
    "https://arabic.cnn.com/"
]
```

- **Output**: Saved as `arabic_texts.csv` with 20 texts for demonstration.

### Text Preprocessing Pipeline
We preprocess the Arabic texts using the following steps:
1. **Diacritics Removal**: Using `pyarabic` to remove tashkeel (تشكيل).
2. **Non-Arabic Removal**: Filter out non-Arabic characters.
3. **Tokenization**: Split text into tokens with `pyarabic`.
4. **Stemming**: Apply Farasa stemmer using `farasapy`.
5. **Stop Words Removal**: Remove common Arabic stop words (e.g., من, في).

```python
def preprocess_arabic_text(text):
    text = araby.strip_tashkeel(text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    tokens = araby.tokenize(text)
    stems = [stemmer.stem(token) for token in tokens]
    filtered_stems = [token for token in stems if token not in arabic_stop_words]
    return " ".join(filtered_stems)
```

## 🧮 Classification Models
We implement four recurrent neural network models to predict relevance scores for Arabic texts:

### 1. Simple RNN
Basic RNN architecture with an embedding layer, RNN layer, and a linear output layer.

### 2. Bidirectional RNN
Processes text in both directions to capture context from past and future tokens.

### 3. LSTM (Long Short-Term Memory)
Uses memory cells and gates to handle long-range dependencies.

### 4. GRU (Gated Recurrent Unit)
A lighter alternative to LSTM with update and reset gates.

```python
# Example: Simple RNN Model
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

- **Training Setup**:
  - Vocabulary size: Based on preprocessed text corpus.
  - Embedding dimension: 100.
  - Hidden dimension: 128.
  - Batch size: 16.
  - Epochs: 10.
  - Optimizer: Adam (learning rate 0.001).
  - Loss: Mean Squared Error (MSE).

## 🤖 Transformer Text Generation
We fine-tune a GPT-2 model for Arabic text generation using the `transformers` library.

### Fine-Tuning Process
1. Create a small Arabic dataset (`custom_dataset.txt`) with sample sentences.
2. Load pre-trained GPT-2 model and tokenizer.
3. Fine-tune on the dataset for 3 epochs.
4. Save the fine-tuned model to `./fine_tuned_gpt2`.

```python
# Example Text Generation
def generate_text(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=0.95, top_k=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

### Sample Prompts
- "مرحبا بك في"
- "الذكاء الاصطناعي هو"
- "اللغة العربية لها"

- **Output**: Saved in `sample_generated_texts.txt` and `text_generation_results.json`.

## 📏 Evaluation Metrics

### Classification Metrics
- **Mean Squared Error (MSE)**: Measures prediction error on original score scale.
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual scores.
- **R² Score**: Proportion of variance explained by the model.

### Text Generation Evaluation
- **Qualitative Assessment**: Evaluate generated text for fluency and coherence.
- **Output Storage**: Results saved as JSON and text files for review.

## 📈 Results & Analysis

### Classification Models
- Models are trained and evaluated on the test set.
- Loss curves are plotted using `matplotlib` to compare training and test performance.

### Text Generation Examples
**Prompt**: "الذكاء الاصطناعي هو"  
**Generated**: (Results vary; see `sample_generated_texts.txt` for examples.)

**Prompt**: "اللغة العربية لها"  
**Generated**: (Results vary; see `sample_generated_texts.txt` for examples.)

## 🔑 Key Learnings
- **Arabic NLP Challenges**: Morphological complexity requires careful preprocessing (diacritics removal, stemming).
- **Model Performance**: LSTM and GRU outperform Simple RNN due to better handling of long-range dependencies.
- **Fine-Tuning Transformers**: GPT-2 fine-tuning is effective for Arabic text generation, even with a small dataset.
- **GPU Utilization**: Using Colab’s Tesla T4 GPU significantly speeds up training.

## 📚 References
- PyTorch Documentation: [pytorch.org](https://pytorch.org)
- Transformers Library: [huggingface.co/transformers](https://huggingface.co/transformers)
- Farasa Stemmer: [farasa.qcri.org](https://farasa.qcri.org)
- PyArabic Documentation: [pyarabic.readthedocs.io](https://pyarabic.readthedocs.io)
