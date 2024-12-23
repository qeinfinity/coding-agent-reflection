# metrics/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
import json

def load_data(db_path: str):
    """Load metrics data from SQLite database."""
    conn = sqlite3.connect(db_path)
    
    # Load interactions
    interactions_df = pd.read_sql_query(
        """
        SELECT *
        FROM interactions
        ORDER BY timestamp DESC
        LIMIT 100
        """,
        conn
    )
    
    # Load completions
    completions_df = pd.read_sql_query(
        """
        SELECT *
        FROM completions
        ORDER BY id DESC
        LIMIT 100
        """,
        conn
    )
    
    conn.close()
    return interactions_df, completions_df

def create_dashboard(db_path: str = "metrics.db"):
    """Create Streamlit dashboard for metrics visualization."""
    st.title("Coding Agent Metrics Dashboard")
    
    try:
        interactions_df, completions_df = load_data(db_path)
        
        if len(interactions_df) == 0:
            st.warning("No metrics data available yet. Run some queries through the agent first.")
            return
        
        # Summary metrics
        st.header("Summary Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_tokens = interactions_df['total_tokens'].mean()
            st.metric("Avg Tokens/Request", f"{int(avg_tokens):,}")
            
        with col2:
            avg_time = interactions_df['response_time'].mean()
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
        with col3:
            total_interactions = len(interactions_df)
            st.metric("Total Interactions", total_interactions)
        
        # Token usage over time
        st.header("Token Usage Trend")
        fig_tokens = px.line(
            interactions_df,
            x='timestamp',
            y='total_tokens',
            title='Token Usage Over Time'
        )
        st.plotly_chart(fig_tokens)
        
        # Layer access patterns
        st.header("Layer Access Patterns")
        layer_patterns = interactions_df['layers_accessed'].apply(json.loads)
        layer_matrix = pd.DataFrame([
            [1 if l in pattern else 0 for l in range(1, 5)]
            for pattern in layer_patterns
        ])
        
        fig_layers = px.imshow(
            layer_matrix.T,
            labels=dict(x="Interaction", y="Layer"),
            y=['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'],
            title="Layer Access Heatmap",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_layers)
        
        # Response time distribution
        st.header("Response Time Distribution")
        fig_time = px.histogram(
            interactions_df,
            x='response_time',
            title='Response Time Distribution',
            nbins=20
        )
        st.plotly_chart(fig_time)
        
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        st.info("If this is your first run, try running some queries through the agent first.")

if __name__ == "__main__":
    create_dashboard()