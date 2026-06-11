# 🧬 Interpretable Drug Repurposing Platform Based on Graph Neural Networks (DRGNN)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18+-blue.svg?style=for-the-badge&logo=react)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16+-green.svg?style=for-the-badge&logo=node.js)](https://nodejs.org/)

**🎓 Graduation Project - Advanced Machine Learning for Drug Discovery**

*Leveraging Graph Neural Networks to identify new therapeutic uses for existing drugs through interpretable AI*

</div>

---

## 📋 Table of Contents

- [🔬 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [📊 Performance Metrics](#-performance-metrics)
- [🏗️ System Architecture](#️-system-architecture)
- [📈 Datasets and Resources](#-datasets-and-resources)
- [🚀 Installation and Setup](#-installation-and-setup)
- [💻 Usage](#-usage)
- [🧪 Model Performance](#-model-performance)
- [🎯 API Endpoints](#-api-endpoints)
- [🔧 Technical Stack](#-technical-stack)
- [📚 Research Papers & References](#-research-papers--references)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Contact](#-contact)

---

## 🔬 Project Overview

Drug repurposing is a crucial strategy in pharmaceutical research that aims to identify new therapeutic applications for existing approved drugs. This project presents a novel approach using **Graph Neural Networks (GNNs)** to predict drug-disease associations while providing interpretable insights into the decision-making process.

### 🎯 Problem Statement

Traditional drug discovery is time-consuming and expensive. Drug repurposing offers a faster, cost-effective alternative by finding new uses for existing drugs. However, current methods lack interpretability and comprehensive analysis of complex biological relationships.

### 💡 Our Solution

DRGNN leverages advanced Graph Neural Networks with interpretable AI to:
- Predict drug-disease associations with high accuracy
- Provide biological pathway explanations
- Enable real-time drug candidate ranking
- Offer a user-friendly platform for researchers

---

## ✨ Key Features

### 🧠 **Graph Neural Network Architecture**
- Advanced heterogeneous GNN models for drug-disease association prediction
- Multi-layered attention mechanisms for complex relationship modeling
- Scalable architecture supporting large biomedical knowledge graphs

### 🔍 **Interpretability & Explainability**
- **GraphMask Explainer**: Post-hoc explanations for model predictions
- **Meta-path Analysis**: Disease → Gene → Drug pathway interpretations
- **Biological Pathway Extraction**: Meaningful insights into drug mechanisms

### 💊 **Drug Repurposing Pipeline**
- Systematic identification of new therapeutic uses for existing drugs
- Confidence scoring and statistical validation
- Integration with comprehensive biomedical databases

### 🖥️ **Interactive Web Platform**
- User-friendly interface for exploring predictions and insights
- Real-time search and ranking of drug candidates
- Visual representations of biological pathways and relationships

### ⚡ **High-Performance API**
- RESTful services for drug predictions and explanations
- Fast inference with cached embeddings
- Comprehensive endpoint coverage for all platform features

### 📊 **Robust Evaluation**
- Multiple validation datasets and cross-validation techniques
- Comprehensive performance metrics (AUC-ROC, AUPRC, etc.)
- Statistical significance testing and confidence intervals

---

## 📊 Performance Metrics

<div align="center">

| Metric | Score | Description |
|--------|--------|-------------|
| **AUC-ROC** | **87.44%** | Area Under ROC Curve |
| **AUPRC** | **84.96%** | Area Under Precision-Recall Curve |
| **Drug Entities** | **7,900+** | Unique drugs in the system |
| **Disease Targets** | **5,000+** | Disease conditions covered |
| **Total Diseases** | **17,080** | Diseases in PrimeKG dataset |
| **Relationships** | **8M+** | Biomedical relationships processed |

</div>

---

## 🏗️ Project Structure

```
drgnn/
├── drgnn/                  # Core Python package (pip-installable)
│   ├── __init__.py         # Package exports (DRGNN, DataControl)
│   ├── model.py            # DRGNN model class + HeteroRGCN
│   ├── data.py             # DataControl for data loading/splitting
│   ├── train.py            # Training pipeline utilities
│   ├── predict.py          # Prediction utilities
│   └── utils/              # Utility modules
│       ├── data_utils.py   # Data processing, evaluation, graph utils
│       ├── model_utils.py  # Model save/load, optimizer helpers
│       ├── graph_utils.py  # Graph creation and visualization
│       └── config.py       # Path configuration
├── API/                    # Web interface (fork of Drug_Explorer)
│   ├── drug_server/        # Flask backend API
│   └── src/                # React/TypeScript frontend
├── Notebooks/              # Jupyter notebooks (EDA, training, analysis)
├── Documentation/          # Thesis PDF, presentation, project rollup
├── DRGNN utils/            # Original utility scripts
├── tests/                  # Unit tests
├── scripts/                # Utility scripts (data setup, etc.)
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation config
├── pyproject.toml           # Build system config
└── LICENSE                 # MIT License
```

![DRGNN API](https://github.com/user-attachments/assets/aa495c05-63b5-41d6-bb07-a26ea7643aed)

### 🔧 **DRGNN API (Fork)**
- **Interactive Drug Search**: Real-time search and ranking of drug candidates
- **RESTful Endpoints**: Comprehensive API for predictions and explanations
- **Confidence Metrics**: Statistical validation scores and pathway analysis
- **Performance**: 87.44% AUC-ROC, 7,900+ drug entities, 5,000+ disease targets
- *Note: The API frontend is adapted from [Drug_Explorer](https://github.com/wangqianwen0418/Drug_Explorer)*

### 🧠 **DRGNN Architecture**
- **Multi-layered Design**: User Interface, API & Business Logic, AI/ML Model, and Data layers
- **Heterogeneous GNN**: Advanced graph neural network with attention mechanisms
- **GraphMask Explainer**: Post-hoc explanations and biological pathway extraction
- **Real-time Processing**: Fast inference with cached embeddings and predictions

![DRGNN Architecture](https://github.com/user-attachments/assets/accb7cf3-6190-43b8-b56b-78b13d2329fb)

### ⚙️ **DRGNN Framework**
- **Complete Pipeline**: From PrimeKG dataset to API decision output
- **Two-Stage Training**: Pre-training on full graph → Fine-tuning for drug-disease predictions
- **Interpretable Pathways**: Disease → Gene → Drug meta-path explanations
- **Performance Metrics**: 87.44% AUROC, 84.96% AUPRC

![DRGNN Framework](https://github.com/user-attachments/assets/a2b0d5eb-95cf-47cc-9e25-f4fa04b762de)

---

## 📈 Datasets and Resources

### 🔗 **Core Dataset: PrimeKG**

[PrimeKG](https://www.nature.com/articles/s41597-023-01960-3) is a comprehensive biomedical knowledge graph that integrates **29 curated resources**:

**Integrated Resources:**
- **UniProt**: Protein sequences and functional information
- **DrugBank**: Comprehensive drug and drug target database
- **MONDO**: Disease ontology and classification
- **CTD**: Comparative Toxicogenomics Database
- **OMIM**: Online Mendelian Inheritance in Man
- **DisGeNET**: Gene-disease associations
- **Protein-Protein Interactions (PPIs)**: Molecular interaction networks

**Dataset Statistics:**
- **17,080 diseases** with comprehensive annotations
- **8M+ relationships** across multiple biological domains
- **Multiple entity types**: Drugs, genes, proteins, diseases, phenotypes
- **High-quality curation** from trusted biomedical sources

![Knowledge Graph](https://github.com/user-attachments/assets/b57e8107-c27d-4be7-a7d4-f88eb8724ab1)

### 📁 **Project Supplementary Data**

| Resource | Description | Access |
|----------|-------------|---------|
| **Complete Workspace** | Full DRGNN Web Application with UI | [Google Drive Link](https://drive.google.com/file/d/1pqnJZUa9__4aUHreNNgGL9auiI0A28Qc/view?usp=sharing) |
| **Model Results** | Predictions, validation results, performance analysis | [Google Drive Folder](https://drive.google.com/drive/folders/1NmfSmFIEqAgWfuOmyFeNK48iwE7C41fz?usp=sharing) |

---

## 🚀 Installation and Setup

### 📋 Prerequisites

Ensure you have the following installed on your system:

```bash
Python 3.8+
Node.js 16+
Git
```

### 🐍 Python Package Setup

1. **Clone the repository**
```bash
git clone https://github.com/salahabdelkhabir/drgnn.git
cd drgnn
```

2. **Create virtual environment**
```bash
python -m venv drgnn_env
source drgnn_env/bin/activate  # On Windows: drgnn_env\Scripts\activate
```

3. **Install the package and dependencies**
```bash
pip install -e .            # Install drgnn package in editable mode
pip install -r requirements.txt  # Ensure all dependencies
```

**Core Dependencies:**
```txt
torch>=1.9.0
dgl>=0.8.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
networkx>=2.6.0
flask>=2.0.0
flask-cors>=3.0.0
```

4. **Download and setup data**
```bash
python scripts/setup_data.py
```

### ⚛️ Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install Node.js dependencies**
```bash
npm install
```

3. **Start development server**
```bash
npm start
```

### 🏃‍♂️ Running the Application

1. **Using the DRGNN package**
```python
from drgnn import DRGNN
from drgnn.data import DataControl

# Load preprocessed data
data = DataControl(data_folder='path/to/data')
data.load_preprocessed('path/to/data')

# Initialize and train model
model = DRGNN(data=data, device='cuda:0')
model.model_initialize()
model.pretrain(n_epoch=2)
model.finetune(n_epoch=500)
model.save_model('./drgnn_model')
```

2. **Start the backend API**
```bash
cd API && python drug_server/application.py
```

3. **Start the frontend (in another terminal)**
```bash
cd API && npm start
```

3. **Access the application**
- Frontend: `http://localhost:3000`
- API: `http://localhost:5000`
- API Documentation: `http://localhost:5000/docs`

---

## 💻 Usage

### 🐍 **Using the DRGNN Package**

```python
from drgnn import DRGNN
from drgnn.data import DataControl

data = DataControl(data_folder='data/primekg')
data.load_preprocessed('data/primekg')

model = DRGNN(data=data, device='cuda:0')
model.load_pretrained('models/drgnn_finetuned')

predictions = model.predict(data.df_test)
```

### 🌐 **REST API (for web interface)**

```python
import requests

# Search for drug repurposing candidates
response = requests.get(
    'http://localhost:8002/api/drug_predictions',
    params={
        'disease_id': '1687.0',
        'top_n': 10
    }
)

predictions = response.json()
```

### 📊 **Getting Explanations**

```python
# Get interpretable explanations via the API
response = requests.get(
    'http://localhost:8002/api/attention_pair',
    params={
        'disease': '1687.0',
        'drug': '6809.0'
    }
)

explanation = response.json()
```

### 🌐 **Web Interface Usage**

1. **Search Interface**: Enter disease name or drug compound
2. **Results Dashboard**: View ranked predictions with confidence scores
3. **Explanation Panel**: Explore biological pathways and mechanisms
4. **Export Options**: Download results in various formats (CSV, JSON, PDF)

---

## 🧪 Model Performance

### 📈 **Evaluation Metrics**

| Dataset | AUC-ROC | AUPRC | Precision | Recall | F1-Score |
|---------|---------|-------|-----------|---------|----------|
| **Training** | 0.9012 | 0.8834 | 0.8756 | 0.8923 | 0.8839 |
| **Validation** | 0.8744 | 0.8496 | 0.8421 | 0.8567 | 0.8493 |
| **Test** | 0.8698 | 0.8445 | 0.8389 | 0.8521 | 0.8454 |

### 🎯 **Cross-Validation Results**

```
5-Fold Cross-Validation:
├── Fold 1: AUC-ROC = 0.8756 ± 0.0123
├── Fold 2: AUC-ROC = 0.8689 ± 0.0156
├── Fold 3: AUC-ROC = 0.8734 ± 0.0134
├── Fold 4: AUC-ROC = 0.8712 ± 0.0145
└── Fold 5: AUC-ROC = 0.8778 ± 0.0118

Average: 0.8744 ± 0.0135
```



---

## 🎯 API Endpoints

### 🔍 **API Endpoints**

#### `GET /api/diseases`
List all available disease IDs with treatment flags

#### `GET /api/drug_predictions`
Get predicted drugs for a disease

**Parameters:**
- `disease_id` (string): Disease ID
- `top_n` (int, optional): Number of top predictions (default: 200)

**Response:**
```json
{
  "predictions": [
    {"score": 0.8923, "id": "6809.0", "known": true},
    {"score": 0.8756, "id": "6810.0", "known": false}
  ],
  "metapath_summary": [...]
}
```

#### `GET /api/attention`
Get attention tree for a disease or drug node

**Parameters:**
- `disease` (string): Disease node ID
- `drug` (string): Drug node ID

#### `GET /api/attention_pair`
Get attention paths connecting disease and drug

**Parameters:**
- `disease` (string): Disease node ID
- `drug` (string): Drug node ID

---

## 🔧 Technical Stack

### 🐍 **Backend Technologies**
- **Python 3.8+**: Core programming language
- **PyTorch 1.9+**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities
- **NetworkX**: Graph processing and analysis
- **Flask**: Web framework for API development

### ⚛️ **Frontend Technologies**
- **React 18+**: User interface framework
- **Node.js 16+**: JavaScript runtime environment
- **Material-UI**: Component library
- **D3.js**: Data visualization
- **Axios**: HTTP client for API communication

### 🧠 **AI/ML Technologies**
- **Graph Neural Networks**: Core model architecture
- **GraphMask**: Explainability framework
- **Attention Mechanisms**: Enhanced model performance
- **Meta-path Analysis**: Interpretable pathway extraction

### 🗄️ **Data & Infrastructure**
- **PrimeKG**: Biomedical knowledge graph
- **JSON/CSV**: Data exchange formats
- **RESTful API**: Service architecture
---

## 📚 Research Papers & References

1. **PrimeKG Dataset**: [Nature Scientific Data](https://www.nature.com/articles/s41597-023-01960-3)
2. **Reference Approach**: [A foundation model for clinician-centered drug repurposing](https://www.nature.com/articles/s41591-024-03233-x)
3. **Graph Neural Networks for Drug Discovery**: [Relevant literature and methodologies](https://medium.com/@mulugetas/drug-discovery-and-graph-neural-networks-gnns-a-regression-example-fc738e0f11f3)
4. **Interpretable AI in Healthcare**: [XAI](https://ieeexplore.ieee.org/document/9916585)

---

## 🤝 Contributing

We welcome contributions to the DRGNN project! Here's how you can help:

### 🚀 **Getting Started**

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### 📋 **Contribution Areas**

- **Model Improvements**: Enhanced GNN architectures
- **New Features**: Additional API endpoints or UI components
- **Documentation**: Improved guides and tutorials
- **Testing**: Expanded test coverage
- **Performance**: Optimization and scalability improvements

### 🐛 **Bug Reports**

When reporting bugs, please include:
- Detailed description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version, etc.)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Salah Gamal Abdelkhabir

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

### 👩‍🏫 **Academic Supervision**
- **Prof. Dr. Noha Elattar** - Research Supervisor and Academic Guidance

### 🏛️ **Institution**
- **Delta University for Science and Technology**
- **Bioinformatics Department**
- **Faculty of Artificial Intelligence**

### 🔬 **Research Group**
- **DRGNN Research Team**
- Contributors and collaborators in the biomedical AI field
### 🌟 **Special Thanks**
- [PrimeKG](https://www.nature.com/articles/s41597-023-01960-3): For providing comprehensive biomedical knowledge
- **Open Source Community**: For essential libraries and frameworks
- **PyTorch Geometric Team**: For excellent graph neural network tools
- **Scientific Community**: For advancing interpretable AI in healthcare
- [Qianwen Wang](https://github.com/wangqianwen0418) — Original [Drug_Explorer](https://github.com/wangqianwen0418/Drug_Explorer) project used as the web interface frontend

---

## 📞 Contact

<div align="center">

### 👨‍💻 **Author: Salah Gamal Abdelkhabir**

[![Email](https://img.shields.io/badge/Email-salahabdelkhabir%40gmail.com-red?style=for-the-badge&logo=gmail)](mailto:salahabdelkhabir@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sala7abdelkhabir)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/salahabdelkhabir)

**🎓 Bioinformatics Student | 🧬 AI Researcher | 💊 Drug Discovery Enthusiast**

</div>

---

## ⚠️ Important Disclaimers

### 🔬 **Research Purpose**
This platform is designed for **research and educational purposes only**. All results and predictions should be interpreted within an academic context.

### 🏥 **Medical Applications**
**⚠️ This platform should NOT be used for actual medical decisions without proper clinical validation and regulatory oversight.**

### 📋 **Validation Requirements**
- Clinical trials and validation studies are required before any medical application
- Regulatory approval must be obtained for therapeutic use
- Professional medical consultation is essential for any health-related decisions

### 🔒 **Data Privacy**
- No personal health information is collected or processed
- All data used is from publicly available research datasets
- Platform designed with privacy-first principles

---

<div align="center">

### 🌟 **Star this repository if you found it helpful!**

**Made with ❤️ for advancing drug discovery through interpretable AI**

---

*© 2024 Salah Gamal Abdelkhabir. This is a graduation project demonstrating the application of Graph Neural Networks in drug repurposing.*

</div>
