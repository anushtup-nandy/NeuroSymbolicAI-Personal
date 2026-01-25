# Decision Intelligence Engine

**A mathematically-sound personal intelligence system** with clean, modular architecture for maintainability and extensibility.

```
modules/
├── core/           # Types, configuration, model loading
├── graph/          # Knowledge graph + analytics (learning)
├── ingestion/      # Multi-source data pipeline (training)
├── heuristics/     # Decision models (WDM, TOPSIS, etc.)
└── models/         # Mathematical models (Causal inference)
```

### 🏗️ Benefits

- **Maintainability**: Single responsibility per module
- **Testability**: Isolated unit testing
- **Extensibility**: Easy to add new features
- **Performance**: Selective imports, faster startup
- **Collaboration**: Clear ownership boundaries

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install streamlit networkx sentence-transformers spacy pandas numpy scikit-learn graphviz requests beautifulsoup4

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Run the Application

```bash
# New modular application
streamlit run main.py

# Legacy monolithic app (for comparison)
streamlit run decision_legacy.py
```

---

## 📦 Module Overview

### Core Module (`modules/core/`)

- **`types.py`**: NodeType, RelationType enums
- **`config.py`**: Configuration constants, model loaders

### Graph Module (`modules/graph/`)

- **`knowledge_graph.py`**: PersonalGraph class (storage + embeddings)
- **`analytics.py`**: Search, PageRank, communities, visualization

### Ingestion Module (`modules/ingestion/`)

- **`base.py`**: BaseIngestor with shared extraction logic
- **`obsidian.py`**: Obsidian vault ingestion
- **`google_takeout.py`**: Gmail, Search, Chrome, YouTube

### Heuristics Module (`modules/heuristics/`)

- **`base.py`**: DecisionHeuristic protocol
- **`wdm.py`**: Weighted Decision Matrix
- **`minimax_regret.py`**: Minimax Regret
- **`topsis.py`**: TOPSIS
- **`bayesian.py`**: Bayesian Decision Theory

### Models Module (`modules/models/`)

- **`causal.py`**: Causal inference with temporal DAG

---

## 🎯 Key Features

### 1. **Multi-Source Knowledge Ingestion**
- Obsidian vault parsing with [[wiki-links]]
- Google Takeout (Gmail, Search, Chrome, YouTube)
- Entity extraction (spaCy NER + regex)
- Relation discovery (causal patterns, co-occurrence)

### 2. **Intelligent Knowledge Graph**
- Multi-layer architecture (explicit, semantic, temporal)
- Sentence-BERT embeddings (384-dim)
- PageRank for influential concepts
- Community detection for thought clusters

### 3. **Rigorous Mathematical Models**

| Model | Module | Use Case |
|-------|--------|----------|
| **WDM** | `heuristics/wdm.py` | General decisions |
| **Minimax Regret** | `heuristics/minimax_regret.py` | Risk-averse |
| **TOPSIS** | `heuristics/topsis.py` | Compromise solutions |
| **Bayesian** | `heuristics/bayesian.py` | Evidence updating |
| **Causal** | `models/causal.py` | Cause-effect |

---

## 📖 Usage Examples

### Import and Use Modules

```python
# Import specific modules
from modules.core import load_sentence_model, NodeType, RelationType
from modules.graph import PersonalGraph, analytics
from modules.ingestion import ObsidianIngestor
from modules.heuristics import DecisionModels

# Load models
model = load_sentence_model()

# Create knowledge graph
graph = PersonalGraph(model)

# Ingest data
ingestor = ObsidianIngestor(graph, spacy_model)
ingestor.ingest("/path/to/vault")

# Search
results = analytics.semantic_search(graph, "machine learning", top_k=5)

# Make decisions
options = [
    {'name': 'Option A', 'scores': {'ROI': 8, 'Risk': 6}},
    {'name': 'Option B', 'scores': {'ROI': 6, 'Risk': 9}}
]
factors = [
    {'name': 'ROI', 'weight': 0.6},
    {'name': 'Risk', 'weight': 0.4}
]

decision = DecisionModels.weighted_decision_matrix(options, factors)
print(decision)
```

### Run Decision Analysis

1. Launch: `streamlit run main.py`
2. Build graph (sidebar)
3. Enter decision query
4. Define options & factors
5. Score and analyze

---

## 🧪 Testing

```bash
# Run tests (when available)
pytest tests/ -v

# Test specific module
pytest tests/graph/test_analytics.py

# With coverage
pytest tests/ --cov=modules --cov-report=html
```

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────┐
│         main.py (Streamlit UI)          │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌──────┐  ┌────────┐  ┌─────────┐
│ Core │  │ Graph  │  │Ingestion│
└──────┘  └────────┘  └─────────┘
    │          │          │
    ▼          ▼          ▼
┌──────────┐  ┌────────────┐
│Heuristics│  │   Models   │
└──────────┘  └────────────┘
```

---

## 🔧 Configuration

Edit `modules/core/config.py`:

```python
# Performance tuning
SEMANTIC_THRESHOLD = 0.7  # Similarity threshold
EMBEDDING_BATCH_SIZE = 32  # Batch size for embeddings

# Model selection
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # Default
    # return SentenceTransformer('all-mpnet-base-v2')  # Higher quality
```

## 🗺️ Roadmap

### Completed ✅
- Modular architecture
- All decision models extracted
- Graph analytics separated
- Ingestion pipeline modularized

### Next Steps
- [ ] Comprehensive test suite
- [ ] Performance benchmarks
- [ ] Documentation for each module
- [ ] Plugin system for custom heuristics
- [ ] CLI interface

---

## 📚 Documentation

- **[Quickstart Guide](quickstart.md)**: Get started in 5 minutes
- **[Implementation Plan](implementation_plan.md)**: Architecture details
- **Module Docs**: See docstrings in each module

---

## 🤝 Contributing

```bash
# Development setup
git clone <repo>
cd decision_engine
pip install -r requirements.txt

# Create feature branch
git checkout -b feature/new-heuristic

# Add your module
# modules/heuristics/my_model.py

# Test
pytest tests/

# Submit PR
```

---

## 📄 License

MIT License - See LICENSE file

