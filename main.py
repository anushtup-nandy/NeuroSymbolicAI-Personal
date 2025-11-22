"""
Decision Intelligence Engine v2.0 - Main Application
====================================================
Streamlit application that orchestrates all modules.

This is the new modular entry point that replaces the monolithic decision.py.
"""

import streamlit as st
import os
import pandas as pd

# Core modules
from modules.core.config import (
    load_sentence_model, 
    load_spacy_model, 
    init_session_state,
    DEFAULT_MODEL
)

# Graph modules
from modules.graph.knowledge_graph import PersonalGraph
from modules.graph import analytics

# Ingestion modules
from modules.ingestion.obsidian import ObsidianIngestor
from modules.ingestion.google_takeout import GoogleTakeoutIngestor
from modules.ingestion.llm_extractor import LLMExtractor  # Phase 1
from modules.ingestion.triplet_extractor import TripletExtractor  # Phase 1

# Heuristics
from modules.heuristics.base import DecisionModels

# Models
from modules.models.causal import CausalInference

# Agents (Phase 2)
from modules.agents.decision_parser import DecisionParser
from modules.agents.pattern_matcher import HistoricalPatternMatcher
from modules.agents.bias_detector import BiasDetector


# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="Decision Intelligence Engine v2",
    layout="wide",
    page_icon="🧬"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stats-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()


# ========================================
# SIDEBAR: SETUP & INGESTION
# ========================================

with st.sidebar:
    st.header("⚙️ Engine Configuration")
    
    # Paths
    obsidian_path = st.text_input(
        "Obsidian Vault Path",
        value="/Users/anushtupnandy/OneDrive/Documents/BITS/obsidian_home/Home base",
        help="Path to your Obsidian vault directory"
    )
    
    takeout_path = st.text_input(
        "Google Takeout Path",
        value="",
        help="Path to extracted Google Takeout folder (optional)"
    )
    
    # Options
    use_llm = st.checkbox(
        "🤖 Enable LLM Extraction",
        value=False,
        help="Use Gemma 3 for advanced entity extraction (slower)"
    )
    
    semantic_threshold = st.slider(
        "Semantic Similarity Threshold",
        0.0, 1.0, 0.7,
        help="Minimum similarity to connect concepts (higher = stricter)"
    )
    
    # Ingestion
    if st.button("🚀 Build Knowledge Graph", type="primary"):
        with st.spinner("Initializing models..."):
            sentence_model = load_sentence_model()
            spacy_model = load_spacy_model()
        
        st.session_state['graph'] = PersonalGraph(sentence_model)
        
        # Initialize Phase 1 LLM components if enabled
        llm_extractor = None
        triplet_extractor = None
        if use_llm:
            with st.spinner("Initializing LLM components..."):
                llm_extractor = LLMExtractor()
                triplet_extractor = TripletExtractor(llm_extractor)
                st.session_state['llm_extractor'] = llm_extractor
                st.session_state['triplet_extractor'] = triplet_extractor
        
        # Ingest Obsidian
        obsidian_ingestor = ObsidianIngestor(
            st.session_state['graph'], 
            spacy_model,
            llm_extractor=llm_extractor,
            triplet_extractor=triplet_extractor
        )
        with st.spinner("Analyzing Obsidian vault..."):
            obsidian_ingestor.ingest(obsidian_path, use_llm=use_llm)
        
        # Store Phase 1 outputs
        if use_llm and llm_extractor:
            st.session_state['interest_profile'] = obsidian_ingestor.get_interest_profile()
            st.session_state['decision_patterns'] = obsidian_ingestor.get_decision_patterns()
            st.info(f"✨ LLM Analysis: Found {len(st.session_state['decision_patterns'])} decision patterns")
        
        # Ingest Google Takeout (if path provided)
        if takeout_path and os.path.exists(takeout_path):
            takeout_ingestor = GoogleTakeoutIngestor(
                st.session_state['graph'],
                spacy_model
            )
            with st.spinner("Parsing Google Takeout..."):
                takeout_ingestor.ingest(takeout_path)
        
        # Build semantic layer
        with st.spinner("Building semantic connections..."):
            edges_added = st.session_state['graph'].build_semantic_layer(semantic_threshold)
            st.session_state['ingestion_log'].append(f"🔗 Added {edges_added} semantic edges")
        
        st.success("✅ Knowledge graph built successfully!")
    
    # Stats
    if st.session_state['graph']:
        st.divider()
        st.subheader("📊 Graph Statistics")
        stats = st.session_state['graph'].export_stats()
        st.metric("Nodes", stats['num_nodes'])
        st.metric("Edges", stats['num_edges'])
        st.metric("Density", f"{stats['density']:.4f}")
        st.metric("Avg Degree", f"{stats['avg_degree']:.2f}")
    
    # Logs
    with st.expander("📜 System Logs"):
        for log in st.session_state['ingestion_log'][-20:]:
            st.text(log)


# ========================================
# MAIN TABS
# ========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Decision Cockpit",
    "🕸️ Knowledge Explorer",
    "🧠 Mental Models",
    "🔬 Causal Analysis"
])


# ========================================
# TAB 1: DECISION COCKPIT
# ========================================

with tab1:
    st.markdown("<div class='main-header'>Strategic Decision Interface</div>", unsafe_allow_html=True)
    
    if not st.session_state['graph']:
        st.warning("⚠️ Please build the knowledge graph first (sidebar)")
    else:
        # === PHASE 2: NATURAL LANGUAGE INPUT ===
        st.subheader("🗣️ Natural Language Decision Input (v3.0)")
        st.caption("Describe your decision in plain English - the AI will auto-generate options, factors, and scores!")
        
        nl_input = st.text_area(
            "Describe Your Decision",
            placeholder="Example: Should I take the startup job ($140k + equity) or stay at BigCo ($180k)? I care about growth and learning, but also financial security.",
            height=120,
            key="nl_decision_input"
        )
        
        col_auto1, col_auto2 = st.columns([1, 1])
        
        with col_auto1:
            auto_gen_button = st.button("🤖 Auto-Generate From Text", type="primary")
        
        with col_auto2:
            if 'llm_extractor' in st.session_state:
                st.success("✅ LLM Ready")
            else:
                st.warning("⚠️ Enable LLM in sidebar for auto-generation")
        
        # Auto-generation logic
        if auto_gen_button:
            if 'llm_extractor' not in st.session_state:
                st.error("❌ Please enable 'LLM Extraction' in sidebar and rebuild the graph first!")
            elif not nl_input or len(nl_input.strip()) < 20:
                st.error("❌ Please provide a more detailed decision description (at least 20 characters)")
            else:
                with st.spinner("🧠 Analyzing your decision with AI..."):
                    # Phase 2: Parse with AI
                    parser = DecisionParser(
                        st.session_state['llm_extractor'],
                        st.session_state['graph']
                    )
                    
                    try:
                        parsed = parser.parse(nl_input)
                        
                        # Update session state with parsed decision
                        st.session_state['options'] = [
                            {
                                'name': opt.name,
                                'scores': parsed.initial_scores.get(opt.name, {})
                            }
                            for opt in parsed.options
                        ]
                        
                        st.session_state['factors'] = [
                            {
                                'name': f.name,
                                'weight': f.weight
                            }
                            for f in parsed.factors
                        ]
                        
                        # Get historical suggestions if available
                        suggested_weights = {}
                        if 'decision_patterns' in st.session_state:
                            matcher = HistoricalPatternMatcher(
                                st.session_state['graph'],
                                st.session_state['decision_patterns']
                            )
                            suggested_weights = matcher.suggest_weights(
                                [f.name for f in parsed.factors],
                                parsed.question
                            )
                            
                            # Update weights with historical suggestions
                            for factor in st.session_state['factors']:
                                if factor['name'] in suggested_weights:
                                    factor['weight'] = suggested_weights[factor['name']]
                        
                        # Detect biases
                        detector = BiasDetector()
                        warnings = detector.detect_biases(
                            parsed,
                            st.session_state.get('decision_patterns', []),
                            st.session_state.get('interest_profile', {}),
                            suggested_weights
                        )
                        
                        # Show success and warnings
                        st.success(f"✅ Auto-generated: {len(parsed.options)} options, {len(parsed.factors)} factors")
                        
                        if warnings:
                            st.warning(f"⚠️ Detected {len(warnings)} potential biases:")
                            for warning in warnings:
                                severity_icon = {
                                    'high': '🔴',
                                    'medium': '🟡',
                                    'low': 'ℹ️'
                                }.get(warning.severity, '⚠️')
                                
                                with st.expander(f"{severity_icon} {warning.bias_type.replace('_', ' ').title()}"):
                                    st.write(f"**{warning.message}**")
                                    if warning.evidence:
                                        st.caption(f"Evidence: {warning.evidence}")
                                    if warning.suggestion:
                                        st.info(warning.suggestion)
                        else:
                            st.success("✅ No cognitive biases detected")
                        
                        # Show what was extracted
                        with st.expander("📊 See What Was Extracted"):
                            st.write("**Options:**")
                            for opt in parsed.options:
                                st.write(f"- {opt.name}")
                                if opt.mentioned_pros:
                                    st.write(f"  ✅ Pros: {', '.join(opt.mentioned_pros)}")
                                if opt.mentioned_cons:
                                    st.write(f"  ❌ Cons: {', '.join(opt.mentioned_cons)}")
                            
                            st.write("\n**Factors:**")
                            for f in parsed.factors:
                                st.write(f"- {f.name} (weight: {f.weight:.2f})")
                        
                        st.info("👇 Review and adjust the auto-generated decision below")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to parse decision: {str(e)}")
                        st.write("Please try rephrasing your decision or check that Ollama is running.")
        
        st.divider()
        st.subheader("📝 Manual Entry (or Review Auto-Generated)")
        st.caption("You can manually enter/edit your decision here, or review what was auto-generated above")
        
        # Options & Factors Setup
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Options")
            if 'options' not in st.session_state:
                st.session_state['options'] = [
                    {'name': 'Option A', 'scores': {}},
                    {'name': 'Option B', 'scores': {}}
                ]
            
            if st.button("➕ Add Option"):
                st.session_state['options'].append({
                    'name': f"Option {len(st.session_state['options'])+1}",
                    'scores': {}
                })
            
            for i, opt in enumerate(st.session_state['options']):
                opt['name'] = st.text_input(
                    f"Option {i+1}",
                    opt['name'],
                    key=f"opt_{i}"
                )
        
        with col2:
            st.subheader("Decision Factors")
            if 'factors' not in st.session_state:
                st.session_state['factors'] = [
                    {'name': 'ROI', 'weight': 0.3},
                    {'name': 'Risk', 'weight': 0.3},
                    {'name': 'Time', 'weight': 0.4}
                ]
            
            if st.button("➕ Add Factor"):
                st.session_state['factors'].append({
                    'name': "New Factor",
                    'weight': 0.5
                })
            
            for i, factor in enumerate(st.session_state['factors']):
                c1, c2 = st.columns([3, 2])
                factor['name'] = c1.text_input(
                    f"Factor {i+1}",
                    factor['name'],
                    key=f"factor_{i}"
                )
                factor['weight'] = c2.slider(
                    "Weight",
                    0.0, 1.0, factor['weight'],
                    key=f"weight_{i}"
                )
        
        # Weight validation
        st.divider()
        weight_sum = sum(f['weight'] for f in st.session_state['factors'])

        if abs(weight_sum - 1.0) > 0.01:
            st.error(f"⚠️ Weights sum to {weight_sum:.2f} (should be 1.0)")
        else:
            st.success(f"✅ Weights sum to {weight_sum:.2f}")

        # Show breakdown
        with st.expander("📊 Weight Breakdown"):
            for f in st.session_state['factors']:
                pct = f['weight'] / weight_sum * 100 if weight_sum > 0 else 0
                st.progress(f['weight'] / max(weight_sum, 1.0))
                st.caption(f"{f['name']}: {f['weight']:.3f} ({pct:.1f}%)")
        
        # Scoring Matrix
        st.divider()
        st.subheader("📊 Evaluation Matrix")
        st.caption("Rate each option on each factor (0-10 scale)")
        
        for opt in st.session_state['options']:
            with st.expander(f"Score: {opt['name']}", expanded=False):
                cols = st.columns(len(st.session_state['factors']))
                for idx, factor in enumerate(st.session_state['factors']):
                    default = opt['scores'].get(factor['name'], 5)
                    opt['scores'][factor['name']] = cols[idx].number_input(
                        factor['name'],
                        0, 10, default,
                        key=f"score_{opt['name']}_{factor['name']}"
                    )
        
        # Run Analysis
        if st.button("🚀 Run Decision Analysis", type="primary"):
            weights = [f['weight'] for f in st.session_state['factors']]
            total = sum(weights)

            if abs(total - 1.0) > 0.01:
                st.warning(f"⚠️ Weights sum to {total:.2f}. Auto-normalizing to 1.0...")
                for f in st.session_state['factors']:
                    f['weight'] = f['weight'] / total

            st.divider()
            
            # 1. Mathematical Models
            st.header("1. Mathematical Decision Models")
            
            model_tabs = st.tabs(["WDM", "Minimax Regret", "TOPSIS", "Bayesian"])
            
            with model_tabs[0]:
                st.subheader("Weighted Decision Matrix")
                st.caption("Optimistic model: Maximizes weighted sum of scores")
                df_wdm = DecisionModels.weighted_decision_matrix(
                    st.session_state['options'],
                    st.session_state['factors']
                )
                st.dataframe(
                    df_wdm.style.highlight_max(axis=0, subset=['WDM_Score']),
                    use_container_width=True
                )
            
            with model_tabs[1]:
                st.subheader("Minimax Regret")
                st.caption("Risk-averse model: Minimizes maximum regret")
                df_regret = DecisionModels.minimax_regret(
                    st.session_state['options'],
                    st.session_state['factors']
                )
                st.dataframe(
                    df_regret.style.highlight_min(axis=0, subset=['Max_Regret']),
                    use_container_width=True
                )
            
            with model_tabs[2]:
                st.subheader("TOPSIS Analysis")
                st.caption("Finds option closest to ideal solution")
                df_topsis = DecisionModels.topsis(
                    st.session_state['options'],
                    st.session_state['factors']
                )
                st.dataframe(
                    df_topsis.style.highlight_max(axis=0, subset=['TOPSIS_Score']),
                    use_container_width=True
                )
            
            with model_tabs[3]:
                st.subheader("Bayesian Decision Theory")
                st.caption("Updates probabilities based on evidence")
                df_bayes = DecisionModels.bayesian_decision(
                    st.session_state['options'],
                    st.session_state['factors']
                )
                st.dataframe(
                    df_bayes.style.highlight_max(axis=0, subset=['Posterior_Normalized']),
                    use_container_width=True
                )
            
            # 2. Semantic Context
            st.divider()
            st.header("2. Knowledge Graph Insights")
            
            # Use nl_input as query if available, otherwise use a default
            query = nl_input if nl_input else "decision analysis"
            
            related_nodes = analytics.semantic_search(st.session_state['graph'], query, top_k=10)
            
            if related_nodes:
                st.success(f"Found {len(related_nodes)} relevant memories")
                
                for node_id, similarity, data in related_nodes:
                    node_type = data.get('type', 'concept')
                    icon = {
                        'heuristic': '📜',
                        'person': '👤',
                        'organization': '🏢',
                        'resource': '🔗',
                        'event': '📅'
                    }.get(node_type, '📄')
                    
                    with st.expander(f"{icon} {node_id} (Relevance: {similarity:.3f})"):
                        st.write(data['content'][:300] + "...")
            else:
                st.info("No highly relevant nodes found. Try adjusting the query.")
            
            # 3. Contradiction Detection
            st.divider()
            st.header("3. Contradiction Analysis")
            contradictions = analytics.detect_contradictions(st.session_state['graph'], query)
            
            if contradictions:
                st.warning(f"⚠️ Found {len(contradictions)} potential contradictions")
                for i, contra in enumerate(contradictions):
                    st.error(f"""
                    **Contradiction {i+1}** (Similarity: {contra['similarity']:.3f})
                    - Node 1: {contra['node_1']}
                    - Node 2: {contra['node_2']}
                    """)
            else:
                st.success("✅ No contradictions detected")
            
            # 4. Influential Concepts
            st.divider()
            st.header("4. Most Influential Concepts (PageRank)")
            influential = analytics.get_influential_nodes(st.session_state['graph'], top_k=10)
            
            df_influence = pd.DataFrame(influential, columns=['Concept', 'PageRank'])
            st.dataframe(df_influence, use_container_width=True)


# ========================================
# TAB 2: KNOWLEDGE EXPLORER
# ========================================

with tab2:
    st.markdown("<div class='main-header'>Knowledge Graph Explorer</div>", unsafe_allow_html=True)
    
    if not st.session_state['graph']:
        st.warning("Build the knowledge graph first")
    else:
        # Search Interface
        search_query = st.text_input("🔍 Search Concepts", "")
        
        if search_query:
            results = analytics.semantic_search(st.session_state['graph'], search_query, top_k=5)
            
            if results:
                st.success(f"Found {len(results)} results")
                
                central_nodes = [r[0] for r in results]
                
                # Visualize
                st.subheader("📊 Concept Network")
                depth = st.slider("Visualization Depth", 1, 3, 2)
                
                viz_graph = analytics.visualize_subgraph(
                    st.session_state['graph'],
                    central_nodes,
                    depth=depth
                )
                st.graphviz_chart(viz_graph)
                
                # Path Finding
                st.divider()
                st.subheader("🛤️ Path Finding")
                col1, col2 = st.columns(2)
                source_node = col1.selectbox("Source", central_nodes)
                target_node = col2.selectbox("Target", central_nodes)
                
                if source_node != target_node:
                    path = analytics.find_path(st.session_state['graph'], source_node, target_node)
                    if path:
                        st.success(f"Path found: {' → '.join(path)}")
                    else:
                        st.info("No direct path found")
        
        # Community Detection
        st.divider()
        st.subheader("🌐 Thought Clusters (Community Detection)")
        communities = analytics.detect_communities(st.session_state['graph'])
        
        if communities:
            st.write(f"Found {len(communities)} communities")
            for i, community in enumerate(communities[:5]):
                with st.expander(f"Community {i+1} ({len(community)} nodes)"):
                    st.write(list(community)[:20])


# ========================================
# TAB 3: MENTAL MODELS
# ========================================

with tab3:
    st.markdown("<div class='main-header'>Active Mental Frameworks</div>", unsafe_allow_html=True)
    
    models = [
        {
            "Model": "Weighted Decision Matrix",
            "Type": "Optimization",
            "Use Case": "General purpose, multi-criteria decisions",
            "Status": "✅ Active"
        },
        {
            "Model": "Minimax Regret",
            "Type": "Risk Management",
            "Use Case": "High-stakes, uncertainty, risk aversion",
            "Status": "✅ Active"
        },
        {
            "Model": "TOPSIS",
            "Type": "Distance-based",
            "Use Case": "Finding compromise solutions",
            "Status": "✅ Active"
        },
        {
            "Model": "Bayesian Decision Theory",
            "Type": "Probabilistic",
            "Use Case": "Updating beliefs with new evidence",
            "Status": "✅ Active"
        },
        {
            "Model": "PageRank",
            "Type": "Graph Analytics",
            "Use Case": "Identifying core concepts",
            "Status": "✅ Active"
        },
        {
            "Model": "Community Detection",
            "Type": "Clustering",
            "Use Case": "Discovering thought patterns",
            "Status": "✅ Active"
        },
        {
            "Model": "Causal Inference",
            "Type": "Temporal",
            "Use Case": "Understanding cause-effect relationships",
            "Status": "⚠️ Requires temporal data"
        }
    ]
    
    st.table(pd.DataFrame(models))
    
    st.divider()
    st.subheader("📚 Framework Details")
    
    with st.expander("Weighted Decision Matrix (WDM)"):
        st.markdown("""
        **Formula:** `Score = Σ(weight_i × score_i)`
        
        **Strengths:**
        - Intuitive and transparent
        - Fast computation
        - Good for stable preferences
        
        **Weaknesses:**
        - Assumes linear utility
        - Sensitive to weight calibration
        """)
    
    with st.expander("Minimax Regret"):
        st.markdown("""
        **Formula:** `Regret = max_score - option_score`
        
        **Strengths:**
        - Risk-averse
        - Minimizes opportunity cost
        - Good for irreversible decisions
        
        **Weaknesses:**
        - Conservative
        - May miss high-reward options
        """)
    
    with st.expander("TOPSIS"):
        st.markdown("""
        **Concept:** Find option closest to ideal and farthest from worst
        
        **Strengths:**
        - Considers both positive and negative ideals
        - Robust to outliers
        
        **Weaknesses:**
        - Computationally heavier
        - Requires normalization
        """)


# ========================================
# TAB 4: CAUSAL ANALYSIS
# ========================================

with tab4:
    st.markdown("<div class='main-header'>Causal Inference Engine</div>", unsafe_allow_html=True)
    
    if not st.session_state['graph']:
        st.warning("Build the knowledge graph first")
    else:
        st.info("🧪 Experimental: Requires temporal metadata in your data")
        
        causal_engine = CausalInference(st.session_state['graph'])
        
        # Build DAG
        if st.button("Build Temporal DAG"):
            with st.spinner("Analyzing temporal relationships..."):
                dag = causal_engine.build_temporal_dag()
                st.success(f"Built DAG with {len(dag.nodes())} temporal nodes and {len(dag.edges())} edges")
                
                if len(dag.nodes()) > 0:
                    st.session_state['causal_dag'] = dag
        
        # Causal Effect Estimation
        if 'causal_dag' in st.session_state:
            st.divider()
            st.subheader("🔬 Causal Effect Estimation")
            
            nodes_list = list(st.session_state['causal_dag'].nodes())
            
            if len(nodes_list) >= 2:
                col1, col2 = st.columns(2)
                treatment = col1.selectbox("Treatment (Cause)", nodes_list)
                outcome = col2.selectbox("Outcome (Effect)", nodes_list)
                
                if st.button("Estimate Causal Effect"):
                    effect = causal_engine.estimate_causal_effect(
                        treatment, outcome, 
                        st.session_state['causal_dag']
                    )
                    
                    st.metric("Estimated Causal Effect", f"{effect:.4f}")
                    
                    if effect > 0.5:
                        st.success("Strong positive causal relationship detected")
                    elif effect > 0.2:
                        st.info("Moderate causal relationship detected")
                    else:
                        st.warning("Weak or no causal relationship detected")
            else:
                st.warning("Not enough temporal nodes for causal analysis")


# ========================================
# FOOTER
# ========================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <b>Decision Intelligence Engine v2.0</b><br>
    Powered by: Sentence-BERT • spaCy • NetworkX • scikit-learn<br>
    Mathematical Models: WDM • Regret Minimization • TOPSIS • Bayesian • Causal Inference
</div>
""", unsafe_allow_html=True)
