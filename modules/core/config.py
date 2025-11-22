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


# ========================================
# CACHED MODEL LOADERS
# ========================================

@st.cache_resource
def load_sentence_model():
    """
    Load Sentence-BERT model (384-dim, 22MB).
    
    This model is used for semantic embeddings throughout the system.
    Cached to avoid reloading on every Streamlit rerun.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')


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
