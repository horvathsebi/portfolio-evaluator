
## Overview
This project applies **Modern Portfolio Theory (MPT)** and **Monte Carlo simulation** to analyze and construct efficient stock portfolios.  
By combining financial theory with Python programming, it explores the trade-off between risk and return, and identifies optimal asset allocations based on the **Sharpe ratio**.

---

## Project Structure
- `portfolio_evaluator.py` — contains all the main functions for portfolio analysis, risk assessment, Monte Carlo simulations and plotting results
- `tester.ipynb` — Jupyter Notebook used to test and run the functions from `portfolio_functions.py`
- `README.md` — project description

---

## Features
- Pulls historical stock data via **Yahoo Finance API**
- Calculates:
  - Daily returns
  - Expected portfolio return
  - Portfolio volatility (standard deviation)
  - Covariance and correlation matrices
- Runs **Monte Carlo simulations** across thousands of random weight combinations
- Identifies:
  - Maximum Sharpe ratio portfolio
  - Minimum volatility portfolio
- Visualizes the **efficient frontier**


---

## Tech
- Python, Jupyter Notebook  
- Pandas, NumPy, Matplotlib, yFinance

---

## Expected Output
- **Efficient Frontier Plot**: risk vs expected return  
- **Portfolio Statistics Table**: expected return, volatility, Sharpe ratio  
- **Optimal Portfolio Allocation**: weight distribution across assets  

---

## How to run

### 1. Clone the repository
```bash
git clone https://github.com/horvathsebi/portfolio-evaluator.git
cd portfolio-evaluator
```

### 2. Install required packages
```bash
pip install pandas numpy matplotlib yfinance fredapi curl-cffi
```

### 3. Obtain a FRED API key
- Go to https://fred.stlouisfed.org/, sign up, and get your API key

### 4. Insert your API key
- Open `portfolio_evaluator.py` and replace ''freadapikey.FREDAPI'' in the 'risk_free_rate' function with your FRED API key

### 5. Run the analysis
- Open `tester.ipynb` to run the pre-built examples  
- Or import and experiment with the functions in `portfolio_evaluator.py` in your own notebook/script




