# Custom LLM Dataset Generation Template

A comprehensive pipeline for creating high-quality synthetic datasets for domain-specific LLM fine-tuning. Generate Q&A pairs from PDFs and web content with automated quality assessment and training-ready preprocessing.

## 🚀 Overview

This template enables you to create specialized datasets for any domain through:

- **Multi-source data extraction** from PDFs and web scraping
- **Synthetic Q&A generation** using advanced prompting techniques
- **Automated quality assessment** with LLM-based filtering
- **Training-ready preprocessing** with customizable chat templates
- **Flexible domain adaptation** for any subject matter

## 📋 Prerequisites

```bash
git clone https://github.com/BlazeWild/Custom_LLM_DataGen_Template.git
cd Custom_LLM_DataGen_Template
pip install -r requirements.txt
```

Set up environment variables in `.env`:

```
GOOGLE_API_KEY=your_google_api_key_here
SERPAPI_KEY=your_serpapi_key_here
```

## 📁 Setup Your Data

### 1. Add Your PDF Documents

Place your domain-specific PDF documents in the `data/` folder:

```
data/
├── document1.pdf
├── document2.pdf
└── domain_guide.pdf
```

### 2. Configure Domain Settings

Update the prompts in:

- `generated_prompt.py` - Modify for your domain
- `agent_webscraper/prompt.py` - Customize web scraping queries

## 🔄 Dataset Generation Pipeline

### Step 1: Extract Chunks from PDFs

Process your PDF documents to create content chunks:

```bash
python chunk_generation.py
```

**What it does:**

- Processes all PDFs in the `data/` directory
- Creates 2000-character chunks with 50-character overlap using Docling
- Saves contextualized chunks as individual JSON files in `chunks/` folder
- Generates comprehensive metadata for tracking

**Example chunk structure:**

```json
{
  "source_file": "data/domain_guide.pdf",
  "chunk_index": 15,
  "raw_text": "Your domain-specific content here...",
  "contextualized_text": "Enhanced context: Your domain-specific content here...",
  "metadata": {
    "chunk_size": 485,
    "contextualized_size": 525
  }
}
```

### Step 2: Web Scraping (Optional)

Generate additional chunks from web sources using LangGraph:

```bash
langgraph dev
```

This starts the LangGraph development server. In the UI prompt box, input requests like:

**Example prompts:**

```
Generate 50 chunks about machine learning fundamentals
Create chunks for sustainable agriculture practices
Make 30 chunks about cybersecurity best practices
```

**What it does:**

- Scrapes web content using intelligent LangGraph agent
- Performs quality inspection with LLM evaluation
- Saves high-quality chunks to `chunks/` directory
- Configurable chunk limits and topics for any domain

### Step 3: Generate Synthetic Q&A Pairs

Transform chunks into training-ready Q&A pairs:

```bash
python syntheticdatageneration.py
```

**What it does:**

- Generates 15 Q&A pairs per chunk using Google Gemini
- Strategic distribution: 85% positive examples, 10% boundary cases, 5% negative examples
- Progress tracking with checkpoint resume capability
- Outputs structured dataset in `dataset/raw.json`

**Example generated pairs:**

```json
[
  {
    "question": "What are the key principles of your domain?",
    "answer": "The **key principles** include:\n\n- **Principle 1**: Detailed explanation\n- **Principle 2**: Comprehensive coverage\n- **Principle 3**: Practical application"
  }
]
```

### Step 4: Quality Assessment

Filter and score generated Q&A pairs:

```bash
python dataquality_check.py
```

**What it does:**

- Dual scoring system (accuracy + style) on 1-10 scale
- LLM-based quality evaluation for domain relevance
- Filters examples scoring >6 on both metrics
- Provides detailed quality analytics and pass rates

### Step 5: Data Preprocessing

Format data for training:

```bash
python preprocess.py
```

**What it does:**

- Converts complex JSON to simple Q&A format
- Standardizes structure for training compatibility
- Final validation and cleanup
- Creates training-ready dataset in `final_dataset/filtered.json`

### Step 6: Model Training (Optional)

Fine-tune your model with the generated dataset:

```bash
python train.py
```

**Before training:**

- Customize the chat template in `train.py` according to your model
- Adjust system prompts for your domain
- Configure training parameters for your hardware

**What it does:**

- QLoRA training with 4-bit quantization
- Custom chat template for consistent behavior
- Configurable epochs and learning rate schedule
- Optimized for various GPU configurations

## 📁 Project Structure

```
├── data/                          # Place your PDF documents here
├── chunks/                        # Generated content chunks (auto-created)
├── dataset/                       # Intermediate datasets (auto-created)
│   ├── raw.json                  # Raw Q&A generation with chunks
│   ├── unfiltered.json           # Flattened Q&A pairs
│   └── quality_results.json      # Quality scored data
├── final_dataset/                 # Final training datasets (auto-created)
│   └── filtered.json             # Final training dataset
├── agent_webscraper/              # Web scraping components
│   ├── agent.py                   # LangGraph scraping agent
│   ├── tools.py                   # Scraping tools
│   └── prompt.py                  # Domain-agnostic prompt templates
├── chunk_generation.py            # PDF chunk extraction
├── syntheticdatageneration.py     # Q&A pair generation
├── dataquality_check.py           # Quality assessment
├── preprocess.py                  # Data formatting
├── generated_prompt.py            # Customizable prompt templates
└── train.py                       # Training script (customize for your model)
```

## ⚙️ Configuration

**Chunk Generation:**

- Chunk size: 2000 characters
- Overlap: 50 characters
- Format: Individual JSON files in `chunks/`

**Synthetic Data Generation:**

- 15 Q&A pairs per chunk (adjustable)
- 4-second rate limiting for API calls
- Google Gemini 2.0 Flash model

**Quality Assessment:**

- Accuracy threshold: >6/10
- Style threshold: >6/10
- Batch processing: 5 pairs per call

**Training (Customizable):**

- Base model: Any HuggingFace model
- LoRA parameters: Adjustable in train.py
- Chat template: Customize for your model
- Hardware: Configurable for your setup

## 🎯 Domain Customization

To adapt this template for your specific domain:

### 1. Update Data Sources

- Place domain-specific PDFs in `data/` folder
- Modify web scraping queries in `agent_webscraper/prompt.py`

### 2. Customize Prompts

- Edit `generated_prompt.py` to reflect your domain
- Update system prompts and example Q&A pairs
- Adjust the distribution of positive/negative examples

### 3. Quality Criteria

- Modify scoring criteria in `dataquality_check.py`
- Define domain-specific quality standards
- Adjust filtering thresholds

### 4. Training Configuration

- Customize chat template in `train.py`
- Update system prompts for your domain
- Configure model parameters for your use case

## 📊 Expected Results

- **Dataset size**: 1000-5000+ high-quality Q&A pairs (depends on input data)
- **Quality score**: 80-90% examples pass filtering
- **Domain coverage**: Comprehensive coverage of your input materials
- **Training readiness**: Properly formatted for immediate fine-tuning

## 📚 References

- [Complete methodology documentation](CREATING_SYNTHETIC_DATA_BLOG.md)
- [Training guide](TRAINING_SUMMARY.md)
- [LangGraph documentation](https://python.langchain.com/docs/langgraph)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)

## 🚀 Quick Start Example

```bash
# 1. Place your PDFs in data/ folder
cp your_domain_docs.pdf data/

# 2. Generate chunks from PDFs
python chunk_generation.py

# 3. Generate Q&A pairs
python syntheticdatageneration.py

# 4. Assess quality
python dataquality_check.py

# 5. Preprocess for training
python preprocess.py

# 6. Your training data is ready in final_dataset/filtered.json
```

Transform any domain expertise into a high-quality training dataset!
