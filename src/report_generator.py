"""
Report generation module for quantitative energy analysis.

This module handles result saving, visualization, and report generation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def save_results(val_metrics, qual_metrics, risk_metrics, combined_scores, output_dir='output/results'):
    """
    Save DataFrames to CSV for later analysis.
    
    Args:
        val_metrics: DataFrame with valuation metrics
        qual_metrics: DataFrame with quality metrics
        risk_metrics: DataFrame with risk metrics
        combined_scores: DataFrame with combined scores
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    val_metrics.to_csv(f'{output_dir}/valuation_{timestamp}.csv')
    qual_metrics.to_csv(f'{output_dir}/quality_{timestamp}.csv')
    risk_metrics.to_csv(f'{output_dir}/risk_{timestamp}.csv')
    combined_scores.to_csv(f'{output_dir}/scores_{timestamp}.csv')
    
    print(f"\n✓ Resultados salvos em {output_dir}/")


def plot_efficient_frontier(returns, risk, output_dir='output'):
    """
    Plot efficient frontier.
    
    Args:
        returns: Series with annualized returns
        risk: Series with annualized volatility
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(risk, returns, s=100, alpha=0.7)
    
    for i, ticker in enumerate(returns.index):
        plt.annotate(ticker, (risk.iloc[i], returns.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Volatilidade Anualizada')
    plt.ylabel('Retorno Anualizado')
    plt.title('Fronteira Eficiente - Setor de Energia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficient_frontier.png', dpi=300)
    plt.close()
    
    print(f"✓ Gráfico salvo: {output_dir}/efficient_frontier.png")


def plot_rolling_beta(returns, stock, benchmark='SPY', window=252, output_dir='output'):
    """
    Plot rolling beta.
    
    Args:
        returns: DataFrame with returns
        stock: Stock ticker
        benchmark: Benchmark ticker
        window: Rolling window size
        output_dir: Output directory path
    """
    from src.metrics import rolling_beta
    
    os.makedirs(output_dir, exist_ok=True)
    
    beta_roll = rolling_beta(returns, stock, benchmark, window)
    
    if beta_roll is not None:
        plt.figure(figsize=(12, 6))
        beta_roll.plot()
        plt.title(f'Beta Rolling {stock} vs {benchmark} ({window}d)')
        plt.ylabel('Beta')
        plt.xlabel('Data')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/rolling_beta_{stock}.png', dpi=300)
        plt.close()
        
        print(f"✓ Gráfico salvo: {output_dir}/rolling_beta_{stock}.png")


def plot_correlation_matrix(returns, output_dir='output'):
    """
    Plot correlation matrix heatmap.
    
    Args:
        returns: DataFrame with returns
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    corr = returns.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Matriz de Correlação - Retornos')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300)
    plt.close()
    
    print(f"✓ Gráfico salvo: {output_dir}/correlation_matrix.png")


def plot_drawdown_chart(returns, stock, output_dir='output'):
    """
    Plot drawdown chart for a specific stock.
    
    Args:
        returns: DataFrame with returns
        stock: Stock ticker
        output_dir: Output directory path
    """
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    if stock not in returns.columns:
        return
    
    # Calculate cumulative returns and drawdown
    prices_norm = np.exp(returns[stock].cumsum())
    rolling_max = prices_norm.expanding().max()
    drawdown = (prices_norm - rolling_max) / rolling_max
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price chart
    ax1.plot(prices_norm.index, prices_norm.values)
    ax1.set_ylabel('Preço Normalizado')
    ax1.set_title(f'{stock} - Evolução de Preço e Drawdown')
    ax1.grid(True)
    
    # Drawdown chart
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.plot(drawdown.index, drawdown.values, color='red')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Data')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drawdown_{stock}.png', dpi=300)
    plt.close()
    
    print(f"✓ Gráfico salvo: {output_dir}/drawdown_{stock}.png")


def format_table(df, title, float_format='.2f'):
    """
    Format DataFrame for display.
    
    Args:
        df: DataFrame to format
        title: Table title
        float_format: Float format string
        
    Returns:
        Formatted DataFrame
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)
    
    # Convert to formatted string
    formatted = df.copy()
    for col in formatted.columns:
        if formatted[col].dtype in ['float64', 'float32']:
            formatted[col] = formatted[col].apply(
                lambda x: f'{x:{float_format}}' if pd.notna(x) else 'N/A'
            )
    
    print(formatted.to_string())
    return formatted


def generate_report(val_df, qual_df, risk_df, beta_spy, beta_wti, mc_results, 
                    combined_scores, selected_ticker, mf_results):
    """
    Generate final report to console.
    
    Args:
        val_df: DataFrame with valuation metrics
        qual_df: DataFrame with quality metrics
        risk_df: DataFrame with risk metrics
        beta_spy: Series with SPY betas
        beta_wti: Series with WTI betas
        mc_results: DataFrame with Monte Carlo results
        combined_scores: DataFrame with combined scores
        selected_ticker: List with selected ticker(s)
        mf_results: Dictionary with multifactor regression results
    """
    print("\n")
    print("="*70)
    print("                    RELATÓRIO FINAL - ANÁLISE QUANTITATIVA")
    print("                    SETOR DE PETRÓLEO/ENERGIA (EUA)")
    print("="*70)
    
    # Executive Summary
    print("\n" + "-"*70)
    print("EXECUTIVE SUMMARY")
    print("-"*70)
    
    if selected_ticker:
        print(f"• ATIVO SELECIONADO (QUBO/SA): {selected_ticker[0]}")
    
    if combined_scores is not None and 'final_score' in combined_scores.columns:
        ranking = combined_scores['final_score'].sort_values(ascending=False)
        print(f"• RANKING COMPLETO: {' > '.join(ranking.index.tolist())}")
    
    print("• Análise baseada em: Valuation, Qualidade, Risco, Monte Carlo")
    print("• Método de otimização: Simulated Annealing (quantum-inspired)")
    
    # Valuation table
    if val_df is not None:
        cols_show = ['earnings_yield', 'fcf_yield', 'ev_ebitda', 'pe_ratio', 'div_yield']
        cols_avail = [c for c in cols_show if c in val_df.columns]
        if cols_avail:
            format_table(val_df[cols_avail].round(4), "MÉTRICAS DE VALUATION")
    
    # Quality table
    if qual_df is not None:
        cols_show = ['profit_margin', 'roe', 'fcf_margin', 'debt_to_equity', 'net_debt_ebitda', 'current_ratio']
        cols_avail = [c for c in cols_show if c in qual_df.columns]
        if cols_avail:
            format_table(qual_df[cols_avail].round(4), "MÉTRICAS DE QUALIDADE")
    
    # Risk table
    if risk_df is not None:
        cols_show = ['ret_annual', 'vol_annual', 'sharpe', 'max_drawdown', 'var_95', 'cvar_95']
        cols_avail = [c for c in cols_show if c in risk_df.columns]
        if cols_avail:
            format_table(risk_df[cols_avail].round(4), "MÉTRICAS DE RISCO")
    
    # Beta table
    if beta_spy is not None and beta_wti is not None:
        beta_df = pd.DataFrame({'beta_SPY': beta_spy, 'beta_WTI': beta_wti})
        format_table(beta_df.round(3), "BETAS (SENSIBILIDADES)")
    
    # Monte Carlo table
    if mc_results is not None:
        cols_show = ['mean', 'median', 'std', 'var_95', 'prob_positive']
        cols_avail = [c for c in cols_show if c in mc_results.columns]
        if cols_avail:
            format_table(mc_results[cols_avail].round(4), "MONTE CARLO (10k simulações, t-Student)")
    
    # Combined scores table
    if combined_scores is not None:
        cols_show = ['return_z', 'valuation_z', 'quality_z', 'risk_z', 'final_score']
        cols_avail = [c for c in cols_show if c in combined_scores.columns]
        if cols_avail:
            format_table(combined_scores[cols_avail].round(3), "SCORES COMBINADOS (Z-scores)")
    
    print("\n" + "="*70)
    print("FIM DO RELATÓRIO")
    print("="*70)
