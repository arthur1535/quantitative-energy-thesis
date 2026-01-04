"""
Data fetching module for quantitative energy analysis.

This module handles downloading price and fundamental data via yfinance.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yfinance as yf


def fetch_price_data(tickers, start, end):
    """
    Download price data via yfinance with error handling.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        
    Returns:
        DataFrame with price data
    """
    print("\n[1] Baixando dados de preços...")
    
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty and len(df) > 100:
                # Extract Close column (may be MultiIndex or simple)
                if isinstance(df.columns, pd.MultiIndex):
                    close_data = df['Close'][ticker] if ticker in df['Close'].columns else df['Close'].iloc[:, 0]
                else:
                    close_data = df['Close']
                data[ticker] = close_data
                print(f"  ✓ {ticker}: {len(df)} registros")
            else:
                print(f"  ✗ {ticker}: dados insuficientes")
        except Exception as e:
            print(f"  ✗ {ticker}: erro - {str(e)[:50]}")
    
    if not data:
        raise ValueError("Nenhum dado de preço obtido!")
    
    prices = pd.DataFrame(data)
    prices = prices.dropna(how='all').ffill().bfill()
    return prices


def fetch_fundamental_data(tickers):
    """
    Extract fundamental data via yfinance.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        DataFrame with fundamental data
    """
    print("\n[2] Baixando dados fundamentalistas...")
    
    fundamentals = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # Extract available metrics
            fund = {
                'marketCap': info.get('marketCap', None),
                'enterpriseValue': info.get('enterpriseValue', None),
                'trailingPE': info.get('trailingPE', None),
                'forwardPE': info.get('forwardPE', None),
                'priceToBook': info.get('priceToBook', None),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', None),
                'profitMargins': info.get('profitMargins', None),
                'operatingMargins': info.get('operatingMargins', None),
                'returnOnEquity': info.get('returnOnEquity', None),
                'returnOnAssets': info.get('returnOnAssets', None),
                'debtToEquity': info.get('debtToEquity', None),
                'currentRatio': info.get('currentRatio', None),
                'quickRatio': info.get('quickRatio', None),
                'freeCashflow': info.get('freeCashflow', None),
                'operatingCashflow': info.get('operatingCashflow', None),
                'totalDebt': info.get('totalDebt', None),
                'totalCash': info.get('totalCash', None),
                'ebitda': info.get('ebitda', None),
                'totalRevenue': info.get('totalRevenue', None),
                'dividendYield': info.get('dividendYield', None),
                'payoutRatio': info.get('payoutRatio', None),
                'beta': info.get('beta', None),
            }
            
            fundamentals[ticker] = fund
            print(f"  ✓ {ticker}: dados obtidos")
            
        except Exception as e:
            print(f"  ✗ {ticker}: erro - {str(e)[:50]}")
            fundamentals[ticker] = {}
    
    return pd.DataFrame(fundamentals).T
