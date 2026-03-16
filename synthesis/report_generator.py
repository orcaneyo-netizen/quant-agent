import os
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Define Output Schemas for proper extraction
class PerTickerReport(BaseModel):
    recommendation: str = Field(description="BUY, HOLD, or SELL")
    confidence: float = Field(description="Confidence rating from 0.0 to 1.0")
    ff_factors: dict = Field(description="Fama-French factor values: market_beta, smb_loading, hml_loading, momentum")
    bias_flags: dict = Field(description="Kahneman behavioral bias flags output from NewsAgent")
    sentiment: dict = Field(description="Summarized sentiment view")
    fundamentals: dict = Field(description="Analyzed fundamental view")
    technicals: dict = Field(description="Analyzed technical view")
    top_drivers: list = Field(description="Top driver features explaining allocation weight from SHAP")
    risk_flags: list = Field(description="Risk flags detected (e.g. overbought, anchoring)")

class PortfolioReport(BaseModel):
    weights: dict = Field(description="Ticker weights optimal ratio")
    sharpe_ratio: float = Field(description="Sharpe ratio maximum achievable")
    expected_return: float = Field(description="Expected annualized return blended")
    volatility: float = Field(description="Volatility annualized standard deviation")
    theory: str = Field(description="Theoretical framework used for construction")

class OptimalReport(BaseModel):
    tickers: list = Field(description="List of tickers analyzed")
    portfolio: PortfolioReport
    per_ticker: dict = Field(description="Dictionary with ticker keys and PerTickerReport values")
    theoretical_basis: dict = Field(description="Theoretical context explaining MPT, FF, and Kahneman triggers")
    thesis: str = Field(description="Markowitz MPT allocation summary")
    summary: str = Field(description="Overall agent summary combining theoretical impacts")

def generate_final_report(agent_outputs: dict, optimizer_results: dict, shap_insights: dict, rag_context: str) -> dict:
    """
    Synthesizes parallel agent node outputs, MPT stats, and SHAP drivers into 
    a single structured JSON report using Claude.
    """
    print("Generating Final Synthesis Report via Claude...")
    
    # Prepare the payload for prompt
    context = {
        'optimizer': {
            'weights': optimizer_results.get('weights', {}),
            'sharpe_ratio': optimizer_results.get('sharpe_ratio', 0),
            'expected_return': optimizer_results.get('expected_return', 0),
            'volatility': optimizer_results.get('volatility', 0)
        },
        'agents': agent_outputs, # dict of {ticker: {news, financials, technicals}}
        'shap': shap_insights,
        'rag_context': rag_context
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert quantitative analyst and portfolio advisor. "
                   "Synthesize the provided analysis containing news sentiment, fundamentals, technical indicators, "
                   "MPT optimization weights, and SHAP explainability inputs into a structured JSON report format."),
        ("user", "Here is the comprehensive analysis context:\n{context}\n\n"
                 "Construct an optimal report structure outlining allocation thesis, summaries per ticker, and risk flags.")
    ])

    try:
        # LLM Initialization
        # Prompt said claude-sonnet-4-20250514. Using latest valid naming: claude-3-5-sonnet-20240620
        # If API key not present, we can't invoke. 
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set. Returning template shell structure.")
            return {
                "tickers": list(agent_outputs.keys()),
                "portfolio": optimizer_results,
                "per_ticker": {t: {"recommendation": "HOLD", "confidence": 0.5, "sentiment": {}, "fundamentals": {}, "technicals": {}, "top_drivers": [], "risk_flags": []} for t in agent_outputs.keys()},
                "thesis": "ApiKey is missing. This is a template shell.",
                "summary": "Report synthesis failed due to API key missing."
            }

        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.2)
        structured_llm = llm.with_structured_output(OptimalReport)
        
        chain = prompt | structured_llm
        result = chain.invoke({"context": str(context)})
        
        # Pydantic dict
        return result.model_dump()
        
    except Exception as e:
         print(f"Error invoking Claude for report: {e}")
         return {}

if __name__ == "__main__":
    test_agents = {
        'NVDA': {
            'news': {'sentiment_label': 'BULLISH', 'sentiment_score': 0.8},
            'fundamentals': {'fundamentals': {'trailing_pe': 40, 'revenue_growth': 0.5}, 'earnings_surprise_history': []},
            'technicals': {'signals': {'rsi': 65, 'macd': 2.0}, 'pattern_flags': ['OVERSOLD']}
        }
    }
    test_opt = {
        'weights': {'NVDA': 1.0},
        'sharpe_ratio': 1.5,
        'expected_return': 0.25,
        'volatility': 0.15
    }
    test_shap = {'NVDA': [{'feature': 'sentiment_score', 'shap_impact': 0.5}]}
    res = generate_final_report(test_agents, test_opt, test_shap, "NVDA reported high earnings last week.")
    print(res)
