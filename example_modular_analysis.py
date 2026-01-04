"""
Example script demonstrating usage of the modular structure.

This script shows how to use the refactored modules for energy sector analysis.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime, timedelta

# Import our modules
from src.data_fetcher import fetch_price_data, fetch_fundamental_data
from src.metrics import (
    calculate_returns, 
    calculate_valuation_metrics,
    calculate_valuation_score,
    calculate_quality_metrics,
    calculate_quality_score,
    calculate_risk_metrics,
    calculate_betas,
    multifactor_regression
)
from src.optimization import (
    monte_carlo_simulation,
    create_score_matrix,
    simulated_annealing_selection,
    scipy_optimization
)
from src.report_generator import (
    save_results,
    plot_efficient_frontier,
    plot_rolling_beta,
    plot_correlation_matrix,
    plot_drawdown_chart,
    generate_report
)


def main():
    """Main analysis function."""
    
    # Configuration
    TICKERS_ACOES = ['CVX', 'XOM', 'COP', 'SLB', 'HAL']
    TICKERS_ETFS = ['XLE', 'OIH']
    BENCHMARKS = ['SPY', 'CL=F']
    ALL_TICKERS = TICKERS_ACOES + TICKERS_ETFS + BENCHMARKS
    
    # Time period
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=5*365)  # 5 years
    
    # Weights for final scoring
    WEIGHTS = {
        'return': 0.25,
        'valuation': 0.25,
        'quality': 0.25,
        'risk_penalty': 0.25
    }
    
    print("="*70)
    print("ANÁLISE QUANTITATIVA - DEMONSTRAÇÃO MODULAR")
    print("="*70)
    print(f"Data da análise: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Período: {START_DATE.strftime('%Y-%m-%d')} a {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Ativos analisados: {TICKERS_ACOES}")
    print("="*70)
    
    # Step 1: Fetch data
    prices = fetch_price_data(ALL_TICKERS, START_DATE, END_DATE)
    fundamentals = fetch_fundamental_data(TICKERS_ACOES)
    
    # Step 2: Calculate returns
    returns = calculate_returns(prices)
    
    # Step 3: Calculate metrics
    val_metrics = calculate_valuation_metrics(fundamentals)
    val_scores = calculate_valuation_score(val_metrics)
    
    qual_metrics = calculate_quality_metrics(fundamentals)
    qual_scores = calculate_quality_score(qual_metrics)
    
    # Filter returns to stocks only for risk metrics
    stock_returns = returns[[col for col in TICKERS_ACOES if col in returns.columns]]
    risk_metrics = calculate_risk_metrics(stock_returns)
    
    # Step 4: Calculate betas
    beta_spy = calculate_betas(returns, 'SPY')
    beta_wti = calculate_betas(returns, 'CL=F')
    
    # Step 5: Monte Carlo simulation
    mc_results = monte_carlo_simulation(stock_returns, n_simulations=10000, horizon_days=252)
    
    # Step 6: Create combined scores
    combined_scores = create_score_matrix(val_scores, qual_scores, risk_metrics, mc_results, WEIGHTS)
    
    # Step 7: Optimization
    selected_ticker, best_score = simulated_annealing_selection(combined_scores, n_select=1)
    
    # Optional: continuous optimization
    optimal_weights = scipy_optimization(combined_scores, max_weight=0.40)
    
    # Step 8: Generate visualizations
    print("\n[10] Gerando visualizações...")
    
    # Efficient frontier
    plot_efficient_frontier(
        risk_metrics['ret_annual'], 
        risk_metrics['vol_annual']
    )
    
    # Rolling beta for top stocks
    if selected_ticker:
        for ticker in selected_ticker:
            plot_rolling_beta(returns, ticker, 'SPY', window=252)
    
    # Correlation matrix
    plot_correlation_matrix(stock_returns)
    
    # Drawdown chart for selected stock
    if selected_ticker:
        plot_drawdown_chart(returns, selected_ticker[0])
    
    # Step 9: Save results
    print("\n[11] Salvando resultados...")
    save_results(val_metrics, qual_metrics, risk_metrics, combined_scores)
    
    # Step 10: Generate report
    # Perform multifactor regression for report
    mf_results = {}
    for ticker in TICKERS_ACOES:
        if ticker in returns.columns:
            result = multifactor_regression(returns, ticker, factors=['SPY', 'CL=F', 'XLE', 'OIH'])
            if result:
                mf_results[ticker] = result
    
    generate_report(
        val_metrics, 
        qual_metrics, 
        risk_metrics, 
        beta_spy, 
        beta_wti, 
        mc_results,
        combined_scores, 
        selected_ticker,
        mf_results
    )
    
    print("\n✅ Análise completa! Verifique o diretório 'output/' para resultados e gráficos.")


if __name__ == "__main__":
    main()
