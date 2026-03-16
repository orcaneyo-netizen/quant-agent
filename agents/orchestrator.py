import sys
import os
import concurrent.futures
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.news_agent import analyze_news
from agents.financials_agent import analyze_fundamentals
from agents.technical_agent import analyze_technicals
from synthesis.portfolio_optimizer import optimize_portfolio
from synthesis.shap_explainer import explain_decisions
from synthesis.report_generator import generate_final_report
from rag.chroma_store import store

from synthesis.factor_loader import load_fama_french_factors

# 1. Define AgentState
class AgentState(TypedDict):
    tickers: List[str]
    news_outputs: Dict[str, Any]
    financials_outputs: Dict[str, Any]
    technical_outputs: Dict[str, Any]
    ff_factors: Dict[str, Any]  # Fama-French loadings
    rag_context: str
    portfolio_weights: Dict[str, float]
    sharpe_ratio: float
    expected_return: float
    volatility: float
    frontier_data: List[dict]
    shap_insights: Dict[str, Any]
    final_report: Dict[str, Any]

# 2. Define Nodes
def news_node(state: AgentState) -> dict:
    tickers = state['tickers']
    results = {}
    
    # Run parallel for all tickers
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(analyze_news, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                results[t] = future.result()
            except Exception as e:
                print(f"News node failed for {t}: {e}")
                
    return {"news_outputs": results}

def financials_node(state: AgentState) -> dict:
    tickers = state['tickers']
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(analyze_fundamentals, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                results[t] = future.result()
            except Exception as e:
                print(f"Financials node failed for {t}: {e}")
                
    return {"financials_outputs": results}

def technical_node(state: AgentState) -> dict:
    tickers = state['tickers']
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(analyze_technicals, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                # Add slight delay so yfinance isn't thrashed
                results[t] = future.result()
            except Exception as e:
                 print(f"Technical node failed for {t}: {e}")
                 
    return {"technical_outputs": results}

def factor_node(state: AgentState) -> dict:
    """
    Fetches Fama-French 3 Factors and calculates loading regressions.
    """
    tickers = state['tickers']
    technicals = state.get('technical_outputs', {})
    
    # Extract momentum dictionary from technical agent
    momentum_dict = {}
    for t in tickers:
         momentum_dict[t] = technicals.get(t, {}).get('momentum', 0.0)
         
    factors = load_fama_french_factors(tickers, momentum_dict=momentum_dict)
    return {"ff_factors": factors}

def optimizer_node(state: AgentState) -> dict:
    """
    Combines agent outputs, querying RAG, and runs MPT optimizer with Kahneman weights.
    """
    tickers = state['tickers']
    news = state.get('news_outputs', {})
    
    # 1. Prepare signal scores map supporting adjusted_sentiment_score
    ticker_signals = {}
    for t in tickers:
        n_res = news.get(t, {})
        # Use Kahneman adjusted score
        ticker_signals[t] = {
            'sentiment': n_res.get('adjusted_sentiment_score', 0.0)
        }
        
    print(f"Running portfolio optimizer for {tickers}")
    opt_results = optimize_portfolio(ticker_signals)
    
    # 2. Query RAG context
    combined_context_text = ""
    try:
        for t in tickers:
            ctx = store.query(t)
            combined_context_text += f"\n--- Context for {t} ---\n{ctx}\n"
    except Exception as e:
         combined_context_text = "RAG context query crashed."

    return {
        "portfolio_weights": opt_results.get('weights', {}),
        "sharpe_ratio": opt_results.get('sharpe_ratio', 0.0),
        "expected_return": opt_results.get('expected_return', 0.0),
        "volatility": opt_results.get('volatility', 0.0),
        "frontier_data": opt_results.get('frontier_data', []),
        "rag_context": combined_context_text
    }

def shap_node(state: AgentState) -> dict:
    """
    Explains allocation outputs using Fama-French factors via SHAP values.
    """
    tickers = state['tickers']
    news = state.get('news_outputs', {})
    financials = state.get('financials_outputs', {})
    factors = state.get('ff_factors', {})
    weights = state.get('portfolio_weights', {})
    
    ticker_features = {}
    for t in tickers:
         ticker_features[t] = {
             'adjusted_sentiment_score': news.get(t, {}).get('adjusted_sentiment_score', 0.0),
             'fundamentals': financials.get(t, {}).get('fundamentals', {}),
             'ff_factors': factors.get(t, {})
         }
         
    insights = explain_decisions(ticker_features, weights)
    return {"shap_insights": insights}

def generator_node(state: AgentState) -> dict:
    """
    Synthesizes final comprehensive JSON report.
    """
    tickers = state['tickers']
    news = state.get('news_outputs', {})
    financials = state.get('financials_outputs', {})
    technicals = state.get('technical_outputs', {})
    
    # Combine agents output for full ticker view
    combined_agents = {}
    for t in tickers:
        combined_agents[t] = {
            'news': news.get(t, {}),
            'fundamentals': financials.get(t, {}),
            'technicals': technicals.get(t, {})
        }
        
    weights_res = {
        'weights': state.get('portfolio_weights', {}),
        'sharpe_ratio': state.get('sharpe_ratio', 0.0),
        'expected_return': state.get('expected_return', 0.0),
        'volatility': state.get('volatility', 0.0)
    }
    
    shap_insights = state.get('shap_insights', {})
    rag_ctx = state.get('rag_context', "")
    
    final_report = generate_final_report(combined_agents, weights_res, shap_insights, rag_ctx)
    return {"final_report": final_report}

# 3. Build StateGraph Workflow
workflow = StateGraph(AgentState)

def parallel_agents_node(state: AgentState) -> dict:
    """Runs news, financials, and technical node logic in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_news = executor.submit(news_node, state)
        future_fin = executor.submit(financials_node, state)
        future_tech = executor.submit(technical_node, state)
        
        # Gather Results
        res_news = future_news.result()
        res_fin = future_fin.result()
        res_tech = future_tech.result()
        
    return {
        "news_outputs": res_news["news_outputs"],
        "financials_outputs": res_fin["financials_outputs"],
        "technical_outputs": res_tech["technical_outputs"]
    }

# Add Nodes
workflow.add_node("agents", parallel_agents_node)
workflow.add_node("factor", factor_node)
workflow.add_node("optimizer", optimizer_node)
workflow.add_node("shap", shap_node)
workflow.add_node("generator", generator_node)

# Define Connections
workflow.add_edge(START, "agents")
workflow.add_edge("agents", "factor")
workflow.add_edge("factor", "optimizer")
workflow.add_edge("optimizer", "shap")
workflow.add_edge("shap", "generator")
workflow.add_edge("generator", END)

# Compile
app = workflow.compile()

def run_quantagent(tickers: List[str]) -> dict:
    """Entrypoint to run Graph orchestrator."""
    initial_state = {
        'tickers': [t.upper() for t in tickers],
        'news_outputs': {},
        'financials_outputs': {},
        'technical_outputs': {},
        'ff_factors': {},
        'rag_context': "",
        'portfolio_weights': {},
        'sharpe_ratio': 0.0,
        'final_report': {}
    }
    print(f"--- Starting QuantAgent graph for {tickers} ---")
    output = app.invoke(initial_state)
    return output

if __name__ == "__main__":
    res = run_quantagent(["NVDA", "AAPL"])
    import json
    print(json.dumps(res.get('final_report', {}), indent=2))
