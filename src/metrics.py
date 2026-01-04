"""
Metrics calculation module for quantitative energy analysis.

This module handles calculation of valuation, quality, and risk metrics.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# Constants for debt-to-equity ratio conversion
D_E_RATIO_MIN_THRESHOLD = 1  # Minimum value to consider as percentage
D_E_RATIO_MAX_THRESHOLD = 100  # Maximum value to consider as percentage


def calculate_returns(prices):
    """Calculate logarithmic daily returns."""
    return np.log(prices / prices.shift(1)).dropna()


def calculate_valuation_metrics(fund_df):
    """
    Calculate valuation metrics.
    
    Args:
        fund_df: DataFrame with fundamental data
        
    Returns:
        DataFrame with valuation metrics
    """
    print("\n[3] Calculando métricas de valuation...")
    
    val = pd.DataFrame(index=fund_df.index)
    
    # Earnings Yield = 1 / P/E
    val['earnings_yield'] = 1 / fund_df['trailingPE']
    val['earnings_yield'] = val['earnings_yield'].replace([np.inf, -np.inf], np.nan)
    
    # FCF Yield = FCF / Market Cap
    val['fcf_yield'] = fund_df['freeCashflow'] / fund_df['marketCap']
    
    # EV/EBITDA (inverse for score - lower is better)
    val['ev_ebitda'] = fund_df['enterpriseToEbitda']
    val['ev_ebitda_inv'] = 1 / val['ev_ebitda']
    val['ev_ebitda_inv'] = val['ev_ebitda_inv'].replace([np.inf, -np.inf], np.nan)
    
    # P/E
    val['pe_ratio'] = fund_df['trailingPE']
    
    # P/B
    val['pb_ratio'] = fund_df['priceToBook']
    
    # P/FCF = Market Cap / FCF
    val['p_fcf'] = fund_df['marketCap'] / fund_df['freeCashflow']
    val['p_fcf'] = val['p_fcf'].replace([np.inf, -np.inf], np.nan)
    
    # Dividend Yield
    val['div_yield'] = fund_df['dividendYield']
    
    return val


def calculate_valuation_score(val_df):
    """
    Create composite valuation score (normalized Z-score).
    
    Args:
        val_df: DataFrame with valuation metrics
        
    Returns:
        DataFrame with valuation scores
    """
    # Metrics where HIGHER is better
    higher_better = ['earnings_yield', 'fcf_yield', 'ev_ebitda_inv', 'div_yield']
    
    scores = pd.DataFrame(index=val_df.index)
    
    for col in higher_better:
        if col in val_df.columns:
            data = val_df[col].dropna()
            if len(data) > 1 and data.std() > 0:
                z = (val_df[col] - data.mean()) / data.std()
                scores[col + '_z'] = z
    
    # Final score = average of Z-scores
    scores['valuation_score'] = scores.mean(axis=1, skipna=True)
    
    return scores


def calculate_quality_metrics(fund_df):
    """
    Calculate quality metrics.
    
    Args:
        fund_df: DataFrame with fundamental data
        
    Returns:
        DataFrame with quality metrics
    """
    print("\n[4] Calculando métricas de qualidade...")
    
    qual = pd.DataFrame(index=fund_df.index)
    
    # Margins
    qual['profit_margin'] = fund_df['profitMargins']
    qual['operating_margin'] = fund_df['operatingMargins']
    
    # Returns
    qual['roe'] = fund_df['returnOnEquity']
    qual['roa'] = fund_df['returnOnAssets']
    
    # FCF/CFO (capital discipline)
    qual['fcf_cfo_ratio'] = fund_df['freeCashflow'] / fund_df['operatingCashflow']
    qual['fcf_cfo_ratio'] = qual['fcf_cfo_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # FCF Margin = FCF / Revenue
    qual['fcf_margin'] = fund_df['freeCashflow'] / fund_df['totalRevenue']
    
    # Leverage
    # Debt/Equity: yfinance often returns percentage rather than ratio
    # Convert values > D_E_RATIO_MIN_THRESHOLD and <= D_E_RATIO_MAX_THRESHOLD to ratios (divide by 100)
    d_to_e_raw = fund_df['debtToEquity']
    d_to_e_adj = d_to_e_raw.apply(
        lambda x: x / 100 if pd.notna(x) and D_E_RATIO_MIN_THRESHOLD < x <= D_E_RATIO_MAX_THRESHOLD else x
    )
    qual['debt_to_equity'] = d_to_e_adj
    qual['net_debt'] = fund_df['totalDebt'] - fund_df['totalCash']
    qual['net_debt_ebitda'] = qual['net_debt'] / fund_df['ebitda']
    qual['net_debt_ebitda'] = qual['net_debt_ebitda'].replace([np.inf, -np.inf], np.nan)
    
    # Liquidity
    qual['current_ratio'] = fund_df['currentRatio']
    qual['quick_ratio'] = fund_df['quickRatio']
    
    return qual


def calculate_quality_score(qual_df):
    """
    Create composite quality score.
    
    Args:
        qual_df: DataFrame with quality metrics
        
    Returns:
        DataFrame with quality scores
    """
    # Metrics where HIGHER is better
    higher_better = ['profit_margin', 'operating_margin', 'roe', 'roa', 
                     'fcf_cfo_ratio', 'fcf_margin', 'current_ratio', 'quick_ratio']
    
    # Metrics where LOWER is better
    lower_better = ['debt_to_equity', 'net_debt_ebitda']
    
    scores = pd.DataFrame(index=qual_df.index)
    
    for col in higher_better:
        if col in qual_df.columns:
            data = qual_df[col].dropna()
            if len(data) > 1 and data.std() > 0:
                z = (qual_df[col] - data.mean()) / data.std()
                scores[col + '_z'] = z
    
    for col in lower_better:
        if col in qual_df.columns:
            data = qual_df[col].dropna()
            if len(data) > 1 and data.std() > 0:
                z = -1 * (qual_df[col] - data.mean()) / data.std()  # Inverted
                scores[col + '_z'] = z
    
    scores['quality_score'] = scores.mean(axis=1, skipna=True)
    
    return scores


def calculate_risk_metrics(returns, periods_year=252):
    """
    Calculate risk metrics for each asset.
    
    Args:
        returns: DataFrame with returns
        periods_year: Number of periods per year (default 252 for daily)
        
    Returns:
        DataFrame with risk metrics
    """
    print("\n[5] Calculando métricas de risco...")
    
    risk = pd.DataFrame(index=returns.columns)
    
    # Annualized return
    risk['ret_annual'] = returns.mean() * periods_year
    
    # Annualized volatility
    risk['vol_annual'] = returns.std() * np.sqrt(periods_year)
    
    # Sharpe (assuming rf = 4% for USD)
    rf = 0.04
    risk['sharpe'] = (risk['ret_annual'] - rf) / risk['vol_annual']
    
    # Max Drawdown
    # When returns are log returns, convert to price index via exponential of cumulative sum
    for col in returns.columns:
        prices_norm = np.exp(returns[col].cumsum())
        rolling_max = prices_norm.expanding().max()
        drawdown = (prices_norm - rolling_max) / rolling_max
        risk.loc[col, 'max_drawdown'] = drawdown.min()
    
    # VaR and CVaR (95% and 99%)
    for col in returns.columns:
        ret = returns[col].dropna()
        risk.loc[col, 'var_95'] = np.percentile(ret, 5)
        risk.loc[col, 'var_99'] = np.percentile(ret, 1)
        risk.loc[col, 'cvar_95'] = ret[ret <= np.percentile(ret, 5)].mean()
        risk.loc[col, 'cvar_99'] = ret[ret <= np.percentile(ret, 1)].mean()
    
    # Skewness and Kurtosis
    risk['skewness'] = returns.skew()
    risk['kurtosis'] = returns.kurtosis()
    
    return risk


def calculate_betas(returns, benchmark_col='SPY'):
    """
    Calculate betas relative to benchmark.
    
    Args:
        returns: DataFrame with returns
        benchmark_col: Benchmark column name
        
    Returns:
        Series with beta values
    """
    print(f"\n[6] Calculando betas vs {benchmark_col}...")
    
    if benchmark_col not in returns.columns:
        print(f"  ✗ Benchmark {benchmark_col} não disponível")
        return pd.Series(dtype=float)
    
    betas = {}
    bench = returns[benchmark_col].dropna()
    
    for col in returns.columns:
        if col == benchmark_col:
            betas[col] = 1.0
            continue
        
        # Align dates
        common = returns[[col, benchmark_col]].dropna()
        if len(common) < 60:
            betas[col] = np.nan
            continue
        
        # OLS regression
        X = sm.add_constant(common[benchmark_col])
        model = sm.OLS(common[col], X).fit()
        betas[col] = model.params[benchmark_col]
    
    return pd.Series(betas, name=f'beta_{benchmark_col}')


def multifactor_regression(returns, stock, factors=['SPY', 'CL=F', 'XLE', 'OIH']):
    """
    Multifactor regression: ret_stock ~ alpha + b1*SPY + b2*WTI + ...
    
    Args:
        returns: DataFrame with returns
        stock: Stock ticker
        factors: List of factor tickers
        
    Returns:
        Dictionary with regression results
    """
    available = [f for f in factors if f in returns.columns]
    if not available or stock not in returns.columns:
        return None
    
    data = returns[[stock] + available].dropna()
    if len(data) < 60:
        return None
    
    y = data[stock]
    X = sm.add_constant(data[available])
    model = sm.OLS(y, X).fit()
    
    return {
        'alpha': model.params['const'],
        'betas': {f: model.params[f] for f in available},
        'r_squared': model.rsquared,
        'pvalues': {f: model.pvalues[f] for f in available}
    }


def rolling_beta(returns, stock, benchmark, window=252):
    """
    Calculate rolling beta.
    
    Args:
        returns: DataFrame with returns
        stock: Stock ticker
        benchmark: Benchmark ticker
        window: Rolling window size
        
    Returns:
        Series with rolling beta values
    """
    if stock not in returns.columns or benchmark not in returns.columns:
        return None
    
    data = returns[[stock, benchmark]].dropna()
    
    rolling_cov = data[stock].rolling(window).cov(data[benchmark])
    rolling_var = data[benchmark].rolling(window).var()
    
    return rolling_cov / rolling_var
