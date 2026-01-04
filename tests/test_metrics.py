"""
Unit tests for metrics module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import (
    calculate_risk_metrics,
    calculate_returns,
    calculate_valuation_metrics,
    calculate_quality_metrics
)


# Test configuration
TEST_RANDOM_SEED = 42


@pytest.fixture
def random_seed():
    """Fixture to provide consistent random seed for tests."""
    return TEST_RANDOM_SEED


def test_max_drawdown():
    """Test if max drawdown is being calculated correctly."""
    # Create simple return series
    returns = pd.DataFrame({
        'TEST': [0.01, -0.05, 0.02, -0.10, 0.05]
    })
    
    risk = calculate_risk_metrics(returns)
    
    assert risk.loc['TEST', 'max_drawdown'] < 0, "Max drawdown should be negative"
    assert risk.loc['TEST', 'vol_annual'] > 0, "Volatility should be positive"


def test_sharpe_ratio(random_seed):
    """Test Sharpe Ratio calculation."""
    # Create random returns
    np.random.seed(random_seed)
    returns = pd.DataFrame({
        'TEST': np.random.normal(0.0005, 0.01, 252)
    })
    
    risk = calculate_risk_metrics(returns)
    
    assert 'sharpe' in risk.columns, "Sharpe ratio should be in columns"
    assert pd.notna(risk.loc['TEST', 'sharpe']), "Sharpe ratio should not be NaN"


def test_returns_calculation():
    """Test logarithmic returns calculation."""
    prices = pd.DataFrame({
        'TEST': [100, 105, 103, 108, 110]
    })
    
    returns = calculate_returns(prices)
    
    assert len(returns) == 4, "Returns should have n-1 rows"
    assert returns.columns[0] == 'TEST', "Column name should be preserved"
    assert not returns.isnull().any().any(), "Returns should not contain NaN"


def test_risk_metrics_shape(random_seed):
    """Test that risk metrics have correct shape."""
    np.random.seed(random_seed)
    returns = pd.DataFrame({
        'STOCK1': np.random.normal(0.0005, 0.01, 252),
        'STOCK2': np.random.normal(0.0003, 0.015, 252)
    })
    
    risk = calculate_risk_metrics(returns)
    
    assert risk.shape[0] == 2, "Should have 2 rows (one per stock)"
    assert 'ret_annual' in risk.columns, "Should have annualized return"
    assert 'vol_annual' in risk.columns, "Should have annualized volatility"
    assert 'sharpe' in risk.columns, "Should have Sharpe ratio"
    assert 'max_drawdown' in risk.columns, "Should have max drawdown"


def test_valuation_metrics():
    """Test valuation metrics calculation."""
    fund_df = pd.DataFrame({
        'trailingPE': [15.0, 20.0, 12.0],
        'freeCashflow': [1e9, 2e9, 1.5e9],
        'marketCap': [50e9, 100e9, 60e9],
        'enterpriseToEbitda': [8.0, 10.0, 7.0],
        'dividendYield': [0.03, 0.02, 0.04],
        'priceToBook': [2.0, 3.0, 1.5]
    }, index=['STOCK1', 'STOCK2', 'STOCK3'])
    
    val = calculate_valuation_metrics(fund_df)
    
    assert 'earnings_yield' in val.columns, "Should have earnings yield"
    assert 'fcf_yield' in val.columns, "Should have FCF yield"
    assert 'ev_ebitda' in val.columns, "Should have EV/EBITDA"
    assert val.shape[0] == 3, "Should have 3 rows"


def test_quality_metrics():
    """Test quality metrics calculation."""
    fund_df = pd.DataFrame({
        'profitMargins': [0.15, 0.20, 0.18],
        'returnOnEquity': [0.18, 0.22, 0.20],
        'debtToEquity': [50.0, 30.0, 40.0],  # In percentage form
        'currentRatio': [1.5, 2.0, 1.8],
        'freeCashflow': [1e9, 2e9, 1.5e9],
        'operatingCashflow': [1.5e9, 2.5e9, 2e9],
        'totalRevenue': [10e9, 15e9, 12e9],
        'totalDebt': [5e9, 3e9, 4e9],
        'totalCash': [2e9, 1e9, 1.5e9],
        'ebitda': [2e9, 3e9, 2.5e9],
        'operatingMargins': [0.20, 0.25, 0.22],
        'returnOnAssets': [0.10, 0.12, 0.11],
        'quickRatio': [1.2, 1.5, 1.3]
    }, index=['STOCK1', 'STOCK2', 'STOCK3'])
    
    qual = calculate_quality_metrics(fund_df)
    
    assert 'profit_margin' in qual.columns, "Should have profit margin"
    assert 'roe' in qual.columns, "Should have ROE"
    assert 'debt_to_equity' in qual.columns, "Should have D/E ratio"
    assert qual.shape[0] == 3, "Should have 3 rows"
    
    # Test D/E conversion
    assert qual.loc['STOCK1', 'debt_to_equity'] == 0.5, "D/E should be converted from percentage"


def test_var_and_cvar(random_seed):
    """Test VaR and CVaR calculations."""
    np.random.seed(random_seed)
    returns = pd.DataFrame({
        'TEST': np.random.normal(0, 0.02, 1000)
    })
    
    risk = calculate_risk_metrics(returns)
    
    assert 'var_95' in risk.columns, "Should have VaR 95%"
    assert 'cvar_95' in risk.columns, "Should have CVaR 95%"
    assert risk.loc['TEST', 'var_95'] < 0, "VaR 95% should be negative"
    assert risk.loc['TEST', 'cvar_95'] < risk.loc['TEST', 'var_95'], "CVaR should be more extreme than VaR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
