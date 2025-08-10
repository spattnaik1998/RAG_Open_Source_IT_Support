# RAG Model Evaluation: Comparative Analysis of Language Models for IT Support Systems

## 🎯 Overview

This project presents a comprehensive evaluation of three state-of-the-art language models in Retrieval-Augmented Generation (RAG) applications specifically designed for IT support scenarios. The evaluation compares **DeepSeek-R1 Distill Qwen 7B**, **Gemma 2B**, and **Phi-3 Mini** across multiple metrics to provide actionable insights for enterprise deployment.

## 🔧 Technical Architecture

### RAG Pipeline Components
- **Document Processing**: RecursiveCharacterTextSplitter (1000 chars/chunk, 200 overlap)
- **Vector Store**: FAISS with sentence-transformers/all-MiniLM-L6-v2 embeddings  
- **Retrieval**: Top-k=3 relevant chunks per query
- **Models**: HuggingFace Transformers with LangChain integration

### Evaluation Framework
- **Dataset**: IT support documentation (26,737 characters, 36 chunks)
- **Test Queries**: 5 realistic IT support scenarios
- **Metrics**: BLEU, ROUGE-1/2/L, Semantic Similarity
- **Hardware**: CUDA-enabled GPU (14.74 GiB memory)

## 📊 Key Results

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic Sim |
|-------|------|---------|---------|---------|--------------|
| **Phi-3 Mini** | 0.0383 | **0.3559** ⭐ | 0.1339 | 0.2611 | **0.8039** ⭐ |
| **Gemma 2B** | **0.0802** ⭐ | 0.3050 | **0.1608** ⭐ | **0.2621** ⭐ | 0.6486 |
| **DeepSeek-R1** | 0.0589 | 0.3520 | 0.1334 | 0.2443 | 0.7635 |

### 🏆 Performance Summary
- **Phi-3 Mini**: Best semantic understanding and content coverage
- **Gemma 2B**: Superior precision and exact phrasal matching  
- **DeepSeek-R1**: Most balanced performance across all metrics

## 🚀 Quick Start

### Prerequisites
```bash
pip install transformers accelerate sentence-transformers faiss-cpu
pip install langchain_community nltk rouge-score sacrebleu
pip install scikit-learn pandas openpyxl
```

### Environment Setup
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### Basic Usage
```python
# Load your IT documentation
with open('source_document.txt', 'r', encoding='utf-8') as f:
    document_text = f.read()

# Initialize RAG pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Create vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Load and evaluate models
# See complete implementation in rag_evaluation_code.py

## 🔍 Evaluation Metrics Explained

### BLEU Score
Measures n-gram overlap between generated and reference answers. Higher scores indicate better lexical similarity.

### ROUGE Metrics
- **ROUGE-1**: Unigram recall and precision
- **ROUGE-2**: Bigram overlap (captures phrase accuracy)
- **ROUGE-L**: Longest common subsequence

### Semantic Similarity
Cosine similarity between sentence embeddings, capturing meaning beyond surface-level words.

## 🎯 Use Case Recommendations

### Choose **Phi-3 Mini** for:
- Customer-facing applications requiring natural language flexibility
- Scenarios where semantic understanding is paramount
- Applications with diverse query formulations

### Choose **Gemma 2B** for:
- Regulated environments requiring exact procedural language
- Safety-critical systems where precision matters
- Resource-constrained deployments (2B parameters)

### Choose **DeepSeek-R1** for:
- Mixed-use environments with varied requirements
- General-purpose IT support applications
- Deployments prioritizing consistent performance

## ⚡ Performance Optimizations

### Memory Management
```python
import torch
import gc

# Clear GPU memory between model loads
torch.cuda.empty_cache()
gc.collect()
```

### Model Configuration
```python
# Optimized settings for production
pipeline_config = {
    "torch_dtype": "auto",
    "device_map": "auto", 
    "max_new_tokens": 256,
    "temperature": 0.7,
    "do_sample": True
}
```

## 📈 Experimental Results

### Question-Specific Performance
1. **PIN Reset**: All models performed well (avg. 0.75+ semantic similarity)
2. **Email Configuration**: Phi-3 Mini excelled (0.82 semantic similarity)
3. **VPN Setup**: Balanced performance across models
4. **Printer Troubleshooting**: Most challenging scenario
5. **Office Issues**: DeepSeek-R1 showed consistency

### Key Insights
- Model size ≠ performance (Gemma 2B competitive despite 2B params)
- Semantic similarity diverges from n-gram metrics
- Context retrieval crucial for all models
- Balanced evaluation requires multiple metrics

## 🛠 Technical Requirements

### Hardware
- **GPU**: CUDA-compatible with 8GB+ memory recommended
- **CPU**: Multi-core processor for embedding computation
- **RAM**: 16GB+ for model loading and processing

### Software
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.35.0+
- **CUDA**: 11.8+ (for GPU acceleration)

## 📚 References

### Academic Papers
- [BLEU: A Method for Automatic Evaluation](https://aclanthology.org/P02-1040/)
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

### Model Documentation
- [DeepSeek-R1](https://huggingface.co/deepseek-ai/deepseek-r1-distill-qwen-7b)
- [Gemma 2B](https://huggingface.co/google/gemma-2b-it)
- [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
git clone https://github.com/yourusername/rag-model-evaluation.git
cd rag-model-evaluation
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace for model hosting and transformers library
- LangChain community for RAG framework components
- Google Colab for providing computational resources
- Open source contributors to evaluation metric implementations

---

⭐ **Star this repository if you found it helpful!** ⭐

*Last updated: August 2025*
