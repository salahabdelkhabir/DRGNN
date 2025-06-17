
# Interpretable Drug Repurposing Platform Based on Graph Neural Networks (DRGNN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16+-green.svg)](https://nodejs.org/)

## 🎓 Graduation Project

This repository contains the implementation of an **Interpretable Drug Repurposing Platform Based on Graph Neural Networks (DRGNN)** - a graduation project focused on leveraging advanced machine learning techniques to identify new therapeutic uses for existing drugs through interpretable AI.

## 🔬 Project Overview

Drug repurposing is a crucial strategy in pharmaceutical research that aims to identify new therapeutic applications for existing approved drugs. This project presents a novel approach using Graph Neural Networks (GNNs) to predict drug-disease associations while providing interpretable insights into the decision-making process.

### Key Features

- **Graph Neural Network Architecture**: Advanced heterogeneous GNN models for drug-disease association prediction
- **Interpretability**: GraphMask explainer and meta-path analysis for understanding model predictions
- **Drug Repurposing**: Systematic identification of new therapeutic uses for existing drugs
- **Interactive Platform**: User-friendly web interface for exploring predictions and insights
- **Comprehensive Evaluation**: Robust validation using multiple datasets and performance metrics
- **Real-time API**: RESTful services for drug predictions and explanations

## 🏗️ System Architecture

The DRGNN platform consists of three main components:

### 1. DRGNN API
- **Interactive Drug Search**: Real-time search and ranking of drug candidates
- **RESTful Endpoints**: Comprehensive API for predictions and explanations
- **Confidence Metrics**: Statistical validation scores and pathway analysis
- **Performance**: 0.8744% AUC-ROC, 1,200+ drug entities, 5,000+ disease targets
  
![image](https://github.com/user-attachments/assets/aa495c05-63b5-41d6-bb07-a26ea7643aed)

### 2. DRGNN Architecture
- **Multi-layered Design**: User Interface, API & Business Logic, AI/ML Model, and Data layers
- **Heterogeneous GNN**: Advanced graph neural network with attention mechanisms
- **GraphMask Explainer**: Post-hoc explanations and biological pathway extraction
- **Real-time Processing**: Fast inference with cached embeddings and predictions
  
![DRGNN Architecture](https://github.com/user-attachments/assets/accb7cf3-6190-43b8-b56b-78b13d2329fb)

### 3. DRGNN Framework
- **Complete Pipeline**: From PrimeKG dataset to API decision output
- **Two-Stage Training**: Pre-training on full graph → Fine-tuning for drug-disease predictions
- **Interpretable Pathways**: Disease → Gene → Drug meta-path explanations
- **Performance Metrics**: 0.8744% AUROC, 0.8496% AUPRC
  
![DRGNN Framework](https://github.com/user-attachments/assets/c39baeba-24ac-4fb5-87c9-f255062d34fb)

## 📊 Datasets and Resources

### Core Dataset
- **PrimeKG**: Biomedical knowledge graph from 29 resources including:
  - UniProt (protein information)
  - DrugBank (comprehensive drug database)
  - MONDO (disease ontology)
  - CTD (Comparative Toxicogenomics Database)
  - OMIM (Online Mendelian Inheritance in Man)
  - DisGeNET (gene-disease associations)
  - PPI Networks (protein-protein interactions)

### Project Resources
- **Complete Workspace**: [DRGNN Web Application](https://drive.google.com/file/d/1pqnJZUa9__4aUHreNNgGL9auiI0A28Qc/view?usp=sharing)
- **DRGNN Results**: [Model Results and Predictions](https://drive.google.com/drive/folders/1NmfSmFIEqAgWfuOmyFeNK48iwE7C41fz?usp=sharing)

## 🚀 Installation and Setup

### Prerequisites

```bash
Python 3.8+
Node.js 16+
PyTorch 1.9+
PyTorch Geometric
NumPy
Pandas
Scikit-learn
NetworkX
Flask
React 18+
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/salahabdelkhabir/DRGNN.git
cd DRGNN
```

2. **Backend Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
# Navigate to frontend directory
cd drug_explorer

# Install Node.js dependencies
npm install
```

### Running the Application

#### 1. Start the Backend Server
```bash
python "Drug_Explorer\drug_server\application.py"
```

#### 2. Start the Frontend Application
```bash
cd drug_explorer
set NODE_OPTIONS=--openssl-legacy-provider && npm start
```

The application will be available at `http://localhost:3000` with the backend API running on the configured port.

## 📁 Project Structure

```
DRGNN/
├── API/                   # Flask backend application for serving the model via REST API
│   └── ...                # Contains application logic, routes, config, database access, etc.

├── DRGNN utils/           # Core utilities and model data
│   └── ...                # GNN models, preprocessing scripts, configuration, etc.

├── Notebooks/             # Jupyter notebooks for experiments, analysis, or demos
│   └── ...                # Interactive exploration and testing of model behavior

├── README.md              # Project overview, setup, and usage documentation

```

## 🧠 Model Architecture

### Dynamic Relational Graph Neural Network (DRGNN)

The core model architecture includes:

1. **Graph Encoder**: Embeds heterogeneous nodes and edges into latent representations
2. **Relational Layers**: Processes different types of biological relationships
3. **Attention Mechanism**: Highlights important connections and pathways
4. **DistMult Decoder**: Predicts drug-disease association probabilities
5. **GraphMask Explainer**: Provides interpretable explanations

### Training Process

1. **Pre-training**: Learn general graph representations on the full biomedical knowledge graph
2. **Fine-tuning**: Specialized training for drug-disease association prediction
3. **Validation**: Disease-centric data splits to ensure robust evaluation

## 📈 Performance Results

### Evaluation Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **AUC-ROC** | 92% | Area Under Receiver Operating Characteristic Curve |
| **AUC-PR** | 80% | Area Under Precision-Recall Curve |
| **Precision@10** | 85% | Precision at top 10 predictions |
| **Recall@10** | 79% | Recall at top 10 predictions |
| **F1-Score** | 84% | Harmonic mean of precision and recall |

### Validated Case Studies

The model successfully identified several known drug repurposing cases:
- **Aspirin** for cardiovascular disease prevention
- **Metformin** for anti-aging applications  
- **Sildenafil** for pulmonary hypertension
- **Allopregne tiparoxec** for metabolic disorders

## 🔍 Interpretability Features

- **Attention Visualization**: Interactive display of important graph connections
- **Meta-Path Analysis**: Biological pathway explanations (Disease → Gene → Drug)
- **Feature Importance**: Identification of key drug and disease characteristics
- **Subgraph Extraction**: Relevant biomedical network segments for each prediction
- **GraphMask Explanations**: Post-hoc interpretability with confidence scores

## 🌐 Web Platform Features

### Frontend (React)
- **Interactive Interface**: Modern, responsive design with real-time updates
- **Drug Search**: Advanced search and filtering capabilities
- **Visualization**: D3.js-powered interactive graph visualizations
- **Results Dashboard**: Comprehensive display of predictions and metrics

### Backend (Flask)
- **RESTful API**: Standardized endpoints for all functionalities
- **Model Serving**: Real-time inference with optimized performance
- **Data Management**: Efficient handling of large biomedical datasets
- **Caching System**: Fast retrieval of predictions and explanations

### API Endpoints

```
POST /api/predict/drug-disease        # Predict drug-disease associations
GET  /api/drugs/{drug_id}/predictions # Get predictions for specific drug
GET  /api/diseases/{disease_id}/treatments # Get potential treatments
POST /api/explain/prediction         # Get explanation for prediction
```

## 🤝 Contributing

We welcome contributions to improve the DRGNN platform:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Commit changes**: `git commit -am 'Add new feature'`
4. **Push to branch**: `git push origin feature/new-feature`
5. **Submit pull request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Supervision**: [Prof.Dr.Noha Elattar]
- **Institution**: [Delta University for Science and Technology]  
- **Department**: [Vioinformatics Department - Faculty of AII]
- **Research Group**: [DRGNN]

Special thanks to:
- The PrimeKG dataset creators for providing comprehensive biomedical knowledge
- The open-source community for essential libraries and frameworks

## 📞 Contact Information

**Author**: Salah Gamal Abdelkhabir  
**Email**: salahabdelkhabir@gmail.com  
**LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/sala7abdelkhabir)
**GitHub**: [@salahabdelkhabir](https://github.com/salahabdelkhabir)


```


---

**Note**: This is a graduation project demonstrating the application of Graph Neural Networks in drug repurposing. The code and results are for academic and research purposes. Medical applications require proper validation and regulatory approval.

**Disclaimer**: This platform is designed for research and educational purposes. It should not be used for actual medical decisions without proper clinical validation and regulatory oversight.
