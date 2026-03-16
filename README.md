# 📈 QuantAgent — LLM-Powered Portfolio Analysis Agent

**QuantAgent** combines Markowitz Modern Portfolio Theory (1952) for optimal asset allocation, the Fama-French Three-Factor Model (1993) for SHAP-based explainability, and Kahneman & Tversky's Behavioral Finance for bias-adjusted sentiment signals. Orchestrated by LangGraph, backed by ChromaDB RAG, and synthesized by Claude LLM.

---

## 🔬 Theoretical Framework

### 1. Markowitz Modern Portfolio Theory (1952)
- **Concept:** Maximize the Sharpe ratio on the efficient frontier.
- **Implementation:** Blends historical returns with LLM sentiment-adjusted expected returns (`historical * (1 + 0.3 * sentiment_score)`). Covariance calculation utilizes Ledoit-Wolf shrinkage matrix.

### 2. Fama-French Three-Factor Model (1993)
- **Concept:** Explains asset returns based on Market beta risk premium, SMB (Small Minus Big - size), and HML (High Minus Low - value style). 
- **Implementation:** Included Carhart Momentum as a 4th factor. OLS regression fit per ticker explains weights inside the SHAP explainer framework.

### 3. Kahneman & Tversky Behavioral Finance
- **Concept:** Account for psychological biases like anchoring, overreaction, and herding.
- **Implementation:** `news_agent.py` scans trigger thresholds adjusting aggregated FinBERT scores appropriately (e.g. reduction for overreaction/herding spikes).

---

## 🛠️ Tech Stack & Badges

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-black.svg)
![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-Optimization-blue)
![pandas_datareader](https://img.shields.io/badge/datareader-FF%20Factors-lightblue)

---

## 🚀 Installation & Usage

### 1. Clone & Setup Directory
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Copy `.env.example` to `.env` and provide your API keys:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 3. Running individual Layers

**A. Deploy Streamlit Dashboard:**
```bash
streamlit run ui/app.py
```

**B. Deploy FastAPI Backend:**
```bash
uvicorn api.main:app --reload
```

---

## 📚 References
1. **Markowitz, H.** (1952). *Portfolio Selection*, The Journal of Finance.
2. **Fama, E. F., & French, K. R.** (1993). *Common risk factors on the returns of stocks and bonds*, JFE.
3. **Kahneman, D., & Tversky, A.** (1979). *Prospect Theory: An Analysis of Decision under Risk*, Econometrica.
