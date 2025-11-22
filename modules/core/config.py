"""
Configuration and Model Loading for Decision Intelligence Engine
=================================================================
Centralized configuration constants and cached model loaders.
"""

import os
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
import spacy
import torch


# ========================================
# CONFIGURATION CONSTANTS
# ========================================

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:4b"
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# Performance tuning
SEMANTIC_THRESHOLD = 0.7
EMBEDDING_BATCH_SIZE = 32
MAX_SEARCH_RESULTS = 10

# Device detection for GPU acceleration
def get_device():
    """
    Detect best available device for model inference.
    
    Priority:
    1. MPS (Apple Silicon GPU) - Mac M1/M2/M3
    2. CUDA (NVIDIA GPU) - Linux/Windows
    3. CPU (fallback)
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()


# ========================================
# CACHED MODEL LOADERS
# ========================================

@st.cache_resource
def load_sentence_model():
    """
    Load Sentence-BERT model (384-dim, 22MB) with GPU acceleration.
    
    This model is used for semantic embeddings throughout the system.
    Automatically uses MPS on Apple Silicon for ~5-10x speedup.
    Cached to avoid reloading on every Streamlit rerun.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    print(f"✅ Sentence-BERT loaded on: {DEVICE.upper()}")
    return model


@st.cache_resource
def load_spacy_model():
    """
    Load spaCy NER model.
    
    Used for entity extraction from text.
    Downloads automatically if not present.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy model (first run)...")
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")


# ========================================
# SESSION STATE INITIALIZATION
# ========================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'graph' not in st.session_state:
        st.session_state['graph'] = None
    if 'ingestion_log' not in st.session_state:
        st.session_state['ingestion_log'] = []
    if 'embeddings_cache' not in st.session_state:
        st.session_state['embeddings_cache'] = {}
