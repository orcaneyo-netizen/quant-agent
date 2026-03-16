import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.orchestrator import run_quantagent

app = FastAPI(title="QuantAgent API", description="LLM-Powered Autonomous MPT Portfolio Analyst")

class AnalyzeRequest(BaseModel):
    tickers: List[str]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Runs the autonomous orchestrator on the given list of tickers and 
    returns a granular optimized portfolio synthesis report.
    """
    if not request.tickers:
         raise HTTPException(status_code=400, detail="Ticker list cannot be empty.")
         
    try:
         graph_output = run_quantagent(request.tickers)
         report = graph_output.get('final_report', {})
         
         if not report:
              return {
                  "status": "partial_success",
                  "message": "Orchestrator finished but final report was unavailable, returning state details.",
                  "graph_state_metadata": {
                      "sharpe_ratio": graph_output.get('sharpe_ratio'),
                      "weights": graph_output.get('portfolio_weights')
                  }
              }
              
         return report
         
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Graph orchestration failed: {str(e)}")

# To run: uvicorn api.main:app --reload
