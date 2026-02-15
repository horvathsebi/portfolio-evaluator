# Portfolio Evaluator: Asset Allocation Engine

> Available at https://portfolio-eval.streamlit.app
## Project Overview
This repository contains a quantitative framework for stochastic portfolio optimization and risk modeling. Built with **Python** and **Streamlit**.

This tool utilizes **Vectorized Monte Carlo simulations** (currently up to 100,000 runs) to construct the Efficient Frontier, identifying optimal asset allocations (Max Sharpe, Min Volatility).

## Core Capabilities & Methodology

### 1. Simulation
To handle the computational load of $N \times 100,000$ matrix operations, the Monte Carlo uses **NumPy linear algebra**.
* **Einstein Summation:** Portfolio volatility is computed via tensor contraction (`np.einsum('ij,jk,ik->i', ...)`) to perform triple matrix products across all vectors simultaneously.

### 2. Robust Data Architecture
* **Direct API Interfacing:** Implements a custom `curl_cffi` client with TLS fingerprinting (impersonating Chrome) to query Yahoo Finance directly, mitigating rate-limiting frictions common in public libraries.
* **Total Return Logic:** Reconstructs "Adjusted Close" prices by forward-adjusting for dividend events to capture true economic return.

### 3. Macro-Adjusted Risk Metrics
* **Dynamic Benchmarking:** Integrates with the **FRED API** to adjust the Risk-Free Rate ($R_f$) in real-time based on the US 10-Year Treasury Yield.
* **Tail Risk:** Calculates **Parametric Value at Risk (VaR)** at the 95% confidence interval assuming a Gaussian distribution of log-returns.

## Technology
* **Python:** Core logic and framework.
* **NumPy:** Vectorized covariance and matrix calculations.
* **Pandas:** Time-series manipulation.
* **Streamlit:** Interactive web dashboard.
* **Matplotlib:** Visualizations.
* **FredAPI:** Macroeconomic data integration.

## Usage

To replicate the analysis locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/horvathsebi/portfolio-evaluator.git](https://github.com/horvathsebi/portfolio-evaluator.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib streamlit scipy fredapi curl_cffi openpyxl
    ```
3.  **Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```


**Sebestyén Horváth**

*BSc, Corporate Finance*

*University of Buckingham & IBS Budapest*

www.linkedin.com/in/sebestyén-horváth
