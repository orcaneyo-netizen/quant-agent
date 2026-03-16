import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.orchestrator import run_quantagent

st.set_page_config(page_title="QuantAgent", page_icon="📈", layout="wide")

st.markdown("""
# 📈 QuantAgent — Quant Portfolio Analysis
**Markowitz MPT · Fama-French · Behavioral Finance**
""")

# Sidebar
st.sidebar.header("Configuration")
default_tickers = "NVDA, AAPL, MSFT, TSLA, AMZN"
ticker_input = st.sidebar.text_input("Enter Tickers (comma separated)", value=default_tickers)
analyze_button = st.sidebar.button("Analyze Portfolio")

if analyze_button or 'quant_results' in st.session_state:
    if analyze_button:
        st.session_state['quant_results'] = None
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        if not tickers:
            st.error("Please enter valid tickers.")
        else:
            with st.spinner(f"Orchestrating agents and optimizing frontier based on core theories for {tickers}..."):
                try:
                    state = run_quantagent(tickers)
                    st.session_state['quant_results'] = state
                except Exception as e:
                    st.error(f"Error running QuantAgent: {e}")

    state = st.session_state.get('quant_results')
    if state:
        report = state.get('final_report', {})
        weights = state.get('portfolio_weights', {})
        frontier_data = state.get('frontier_data', [])
        shap_insights = state.get('shap_insights', {})
        news = state.get('news_outputs', {})
        factors = state.get('ff_factors', {})

        # TAB VIEW
        tab1, tab2, tab3 = st.tabs(["📊 Portfolio Allocation", "🕵️ Per Ticker analysis", "📁 Raw Outputs"])

        with tab1:
            st.subheader("Markowitz Efficient Frontier")
            
            w_col1, w_col2, w_col3 = st.columns(3)
            w_col1.metric("Optimal Sharpe Ratio", f"{state.get('sharpe_ratio', 0.0):.2f}")
            w_col2.metric("Expected Return", f"{state.get('expected_return', 0.0)*100:.2f}%")
            w_col3.metric("Volatility (Risk)", f"{state.get('volatility', 0.0)*100:.2f}%")

            left_col, right_col = st.columns(2)

            with left_col:
                 st.write("#### Allocation Breakdown")
                 alloc = {k: v for k,v in weights.items() if v > 0.01}
                 if alloc:
                      fig_pie = px.pie(values=list(alloc.values()), names=list(alloc.keys()), title="Optimal Weights")
                      st.plotly_chart(fig_pie, use_container_width=True)
                 else:
                      st.warning("No allocations assigned above 1% threshold.")

            with right_col:
                 st.write("#### Efficient Frontier visualizer (Hero Chart)")
                 if frontier_data:
                      df_front = pd.DataFrame(frontier_data)
                      fig = go.Figure()
                      fig.add_trace(go.Scatter(
                          x=df_front['volatility'], y=df_front['return'],
                          mode='markers',
                          marker=dict(size=5, color=df_front['sharpe'], colorscale='Viridis', colorbar=dict(title="Sharpe"), opacity=0.6),
                          name="Random Portfolios"
                      ))
                      # Optimal star
                      fig.add_trace(go.Scatter(
                          x=[state.get('volatility')], y=[state.get('expected_return')],
                          mode='markers', marker=dict(color='red', size=15, symbol='star'),
                          name="Optimal Max Sharpe"
                      ))
                      fig.update_layout(xaxis_title="Risk (Volatility)", yaxis_title="Expected Return")
                      st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Fama-French & Kahneman Bias Analytics")
            
            for ticker in state['tickers']:
                 ticker_report = report.get('per_ticker', {}).get(ticker, {})
                 ticker_news = news.get(ticker, {})
                 ticker_factors = factors.get(ticker, {})
                 
                 with st.expander(f"**{ticker}** Parameters Insights"):
                      
                      # Line 1: Recommendation Badge + Bias flags
                      rec = ticker_report.get('recommendation', 'HOLD')
                      # Manual badge color
                      color = "green" if rec == 'BUY' else "red" if rec == 'SELL' else "blue"
                      st.markdown(f"### <span style='color:{color}'>{rec}</span> (Confidence: {ticker_report.get('confidence', 0.0)})", unsafe_allowed_html=True)
                      
                      # Behavioral Flags
                      b_flags = ticker_news.get('bias_flags', {})
                      flag_str = ""
                      if b_flags.get('overreaction'): flag_str += "🚨 OVERREACTION "
                      if b_flags.get('herding'): flag_str += "🐑 HERDING "
                      if b_flags.get('anchoring'): flag_str += "⚓ ANCHORING "
                      
                      if flag_str:
                           st.error(f"Kahneman Bias Triggered: {flag_str}")
                      else:
                           st.success("No critical behavioral bias triggers.")
                           
                      # Breakdown columns
                      u_col1, u_col2 = st.columns(2)
                      
                      with u_col1:
                           st.write("**Fama-French Factor Attribute Loadings**")
                           # Create Factor visual bar list
                           factor_bars = {
                                'Market Beta': ticker_factors.get('market_beta', 1.0),
                                'SMB (Size)': ticker_factors.get('smb_loading', 0.0),
                                'HML (Value)': ticker_factors.get('hml_loading', 0.0),
                                'Momentum': ticker_factors.get('momentum', 0.0)
                           }
                           fig_ff = px.bar(
                                x=list(factor_bars.values()), y=list(factor_bars.keys()), orientation='h',
                                title="Fama-French factor exposures", labels={'x': 'Loading Value', 'y': 'Factor'}
                           )
                           st.plotly_chart(fig_ff, use_container_width=True)

                      with u_col2:
                           st.write("**SHAP Explainability Impact (Target = Allocation %)**")
                           t_shap = shap_insights.get(ticker, [])
                           if t_shap:
                                feat_names = [x['feature'] for x in t_shap]
                                feat_impacts = [x['shap_impact'] for x in t_shap]
                                fig_shap = px.bar(
                                    x=feat_impacts, y=feat_names, orientation='h',
                                    labels={'x': 'Impact on Allocation', 'y': 'Feature'},
                                    color=feat_impacts, color_continuous_scale='RdBu_r'
                                )
                                st.plotly_chart(fig_shap, use_container_width=True)
                           
                      st.write(f"**Sentiment Profile:** Raw: `{ticker_news.get('raw_sentiment_score')}` | Adjusted: `{ticker_news.get('adjusted_sentiment_score')}`")

        with tab3:
            st.subheader("Workflow Execution JSON")
            st.json(state)
            if report.get('summary'):
                 st.info(f"Report Summary: {report.get('summary')}")
