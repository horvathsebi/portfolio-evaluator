import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred 
import matplotlib.pyplot as plt
from portfolio_eval import PortfolioEvaluator
import io


st.set_page_config(page_title="Portfolio Evaluator", layout="wide", initial_sidebar_state="auto")

def apply_custom_plt_theme():
    brand_bg = (91/255, 109/255, 144/255)       
    brand_dark = (37/255, 37/255, 37/255)       
    brand_text = (206/255, 206/255, 206/255)   
    brand_primary = (74/255, 74/255, 74/255)    

    plt.rcParams.update({
        "font.family": "monospace",
        "font.size": 10,
        "text.color": brand_text,
        "figure.facecolor": brand_bg,    
        "axes.facecolor": brand_dark,    
        "axes.edgecolor": brand_primary,
        "axes.labelcolor": brand_text,
        "xtick.color": brand_text,
        "ytick.color": brand_text,
        "grid.color": brand_primary,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.color": brand_primary,
        "patch.edgecolor": brand_text,
    })
apply_custom_plt_theme()




def get_risk_free_rate(api_key=None):
    """Fetches 10Y Treasury Rate from FRED if key exists, else returns 4.0%"""
    default_rate = 0.04
    if not api_key:
        return default_rate
    
    try:
        fred = Fred(api_key=api_key)
        series = fred.get_series('DGS10')
        latest_rate = series.iloc[-1]
        return float(latest_rate) / 100.0
    except Exception as e:
        st.sidebar.warning(f"Error fetching FRED data: {e}. Using default 4%.")
        return default_rate


st.title("Portfolio Evaluator - Sebestyén Horváth")


st.sidebar.header("Settings")


fred_api_key = st.sidebar.text_input("FRED API Key (Optional)", type="password", help="If empty uses 4% as default")
rf_rate = get_risk_free_rate(fred_api_key)
st.sidebar.caption(f"Using Risk-Free Rate: {rf_rate:.2%}")

tickers_input = st.sidebar.text_input("Tickers", "AAPL, MSFT, PLTR, GUV.F, LMT")

weights_input = st.sidebar.text_input("Weights (Optional)", "", help="If empty assigns equal weights.")

period_years = st.sidebar.number_input("Years of Data", min_value=1, max_value=10, value=6)
num_sims = st.sidebar.slider("Simulations", 10000, 100000, 60000)

@st.cache_resource
def get_portfolio_data(tickers_list, weights_list, period, rf):
    pe = PortfolioEvaluator(tickers_list, weights_list, risk_free_rate=rf)
    pe.fetch_data(period=f"{period}y") 
    return pe

if 'pe_data' not in st.session_state:
    st.session_state['pe_data'] = None

if st.sidebar.button("Analyze"):

    tickers = [t.strip() for t in tickers_input.split(',')]
    
    if not weights_input.strip():
        weights = None
    else:
        try:
            weights = [float(w) for w in weights_input.split(',')]
        except:
            st.error("Invalid weights format.")
            st.stop()

    with st.spinner("Fetching data..."):
        pe = get_portfolio_data(tickers, weights, period_years, rf_rate)
        
        if pe.returns_df is None or pe.returns_df.empty:
            st.error("No data fetched.")
            st.stop()
            
    st.session_state['pe_data'] = pe


#Github
st.sidebar.markdown("<br>" * 10, unsafe_allow_html=True)
st.sidebar.link_button("Other Projects", "https://github.com/horvathsebi", type="tertiary")

if st.session_state['pe_data'] is not None: 
    pe = st.session_state['pe_data']


    st.subheader("Portfolio Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hist. Mean Return", f"{pe.hist_mean_return:.2%}")
    col2.metric("Volatility (Ann.)", f"{pe.std_dev:.2%}")
    col3.metric(f"Sharpe (Rf={rf_rate:.2%})", f"{pe.sharpe_ratio:.2f}")
    col4.metric("VaR (95%)", f"{pe.var_95:.2%}", help="Value at Risk: The maximum loss (or sometimes min gain) expected 95% of the time over a year.")


    st.subheader("Correlation Matrix")
    st.dataframe(pe.get_correlation_matrix())

    st.subheader("Portfolio Simulations")
    
    mc_results = pe.run_monte_carlo(num_sims)
    
    if not mc_results.empty:

        st.markdown("---")
        st.subheader("Optimal Portfolios")


        max_sharpe_idx = mc_results['Sharpe'].idxmax()
        min_vol_idx = mc_results['Volatility'].idxmin()

        max_sharpe_port = mc_results.loc[max_sharpe_idx]
        min_vol_port = mc_results.loc[min_vol_idx]


        def draw_pie(weights_row, title):
            filtered_weights = weights_row[pe.tickers]
            filtered_weights = filtered_weights[filtered_weights > 0.01]
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            wedges, texts, autotexts = ax.pie(
                filtered_weights, 
                labels=filtered_weights.index, 
                autopct='%1.1f%%', 
                startangle=90,
                textprops={'color':"white", 'fontsize': 12},
                colors=plt.cm.cool(np.linspace(0, 1, len(filtered_weights)))
            )
            plt.setp(autotexts, size=10, weight="bold", color="black")
            ax.set_title(title, color="white", fontsize=14, pad=20)
            fig.patch.set_alpha(0) 
            return fig



        col_best, col_safe = st.columns(2)

        with col_best:
            st.markdown("### Maximum Sharpe Portfolio")
            st.metric("Historical Mean Return", f"{max_sharpe_port['Return']:.2%}")
            st.metric("Volatility", f"{max_sharpe_port['Volatility']:.2%}")
            st.metric("Sharpe Ratio", f"{max_sharpe_port['Sharpe']:.2f}")
            
            st.write("**Allocation:**")
            fig_ms = draw_pie(max_sharpe_port, "Max Sharpe Allocation")
            st.pyplot(fig_ms)

        with col_safe:
            st.markdown("### Minimum Volatility Portfolio")
            st.metric("Historical Mean Return", f"{min_vol_port['Return']:.2%}")
            st.metric("Volatility", f"{min_vol_port['Volatility']:.2%}")
            st.metric("Sharpe Ratio", f"{min_vol_port['Sharpe']:.2f}")
            
            st.write("**Allocation:**")
            fig_mv = draw_pie(min_vol_port, "Min Volatility Allocation")
            st.pyplot(fig_mv)
            fig, ax = plt.subplots(figsize=(10, 6))
            years_label = "Year" if period_years == 1 else "Years"
            
            sc = ax.scatter(mc_results['Volatility'], mc_results['Return'], 
                            c=mc_results['Sharpe'], cmap='cool', s=10)
            plt.colorbar(sc, label='Sharpe Ratio')
            
        ax.scatter(pe.std_dev, pe.hist_mean_return, c='red', s=30, marker='*', label='Your Portfolio')
        
        ax.set_xlabel('Standard Deviation')
        ax.set_ylabel(f'Mean Returns Over {period_years} {years_label}')
        ax.set_title(f'Portfolio Simulations ({num_sims} runs)')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough data for simulation.")

    

    
    
    # Excel
    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine='openpyxl') as writer:
        pe.returns_df.to_excel(writer, sheet_name='Returns')
        pe.get_correlation_matrix().to_excel(writer, sheet_name='Correlation')
    xlsx_buffer.seek(0)

    # CSV
    csv_buffer = io.BytesIO()
    pe.returns_df.to_csv(csv_buffer, index=True)
    csv_buffer.seek(0)

    st.download_button("Export Data to Excel", xlsx_buffer, "portfolio_data.xlsx", "application/vnd.ms-excel", type="tertiary")
    st.download_button("Export Data to CSV", csv_buffer, "portfolio_data.csv", "text/csv", type="tertiary")




else:
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    st.markdown("""
    1. Enter tickers separated by commas.
       
    2. Enter the corresponding portfolio weights (if applicable).
    
    3. Select how many years of data you need, and the number of simulations to run.
    
    4. Enter your Fred API Key (for risk-free data).          
    
    4. Click Analyze.
    
    ---
    *This tool uses Monte Carlo simulations to estimate portfolio performance based on historical closing prcies.*

    *Data is sourced using Yahoo Finance API*             
    """)