"""
Optimization module for quantitative energy analysis.

This module handles Monte Carlo simulations and portfolio optimization
using Simulated Annealing and scipy methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t
from scipy.optimize import minimize


def scenario_returns(regression_results, scenarios):
    """
    Estimate expected returns by scenario using regression coefficients.
    
    Args:
        regression_results: Dictionary with regression results
        scenarios: Dictionary with scenario parameters
        
    Returns:
        Dictionary with expected returns by scenario
    """
    scenario_ret = {}
    
    for name, params in scenarios.items():
        # Return = alpha + b_SPY * shock_SPY + b_WTI * shock_WTI
        expected = regression_results['alpha'] * 252  # Annualize alpha
        
        if 'SPY' in regression_results['betas']:
            expected += regression_results['betas']['SPY'] * params['spy_shock']
        
        if 'CL=F' in regression_results['betas']:
            expected += regression_results['betas']['CL=F'] * params['wti_shock']
        
        scenario_ret[name] = expected
    
    return scenario_ret


def monte_carlo_simulation(returns, n_simulations=10000, horizon_days=252):
    """
    Monte Carlo simulation for future return distribution.
    
    Args:
        returns: DataFrame with returns
        n_simulations: Number of simulations
        horizon_days: Forecast horizon in days
        
    Returns:
        DataFrame with simulation results
    """
    print("\n[7] Executando Monte Carlo (10k simulações com t-Student)...")
    
    results = {}
    # Seed once for reproducibility across assets
    np.random.seed(42)
    # Use Student's t distribution for heavy tails
    df_t = 5  # degrees of freedom; lower df => heavier tails
    
    for col in returns.columns:
        ret = returns[col].dropna()
        if len(ret) < 100:
            continue
        
        mu = ret.mean()
        sigma = ret.std()
        
        # Simulate daily returns with Student's t distribution (heavy tails)
        simulated = t.rvs(df_t, loc=mu, scale=sigma, size=(n_simulations, horizon_days))
        
        # Total return at horizon
        total_returns = np.exp(simulated.sum(axis=1)) - 1
        
        results[col] = {
            'mean': total_returns.mean(),
            'median': np.median(total_returns),
            'std': total_returns.std(),
            'var_95': np.percentile(total_returns, 5),
            'var_99': np.percentile(total_returns, 1),
            'cvar_95': total_returns[total_returns <= np.percentile(total_returns, 5)].mean(),
            'prob_positive': (total_returns > 0).mean(),
            'prob_gt_10pct': (total_returns > 0.10).mean(),
            'prob_lt_minus_20pct': (total_returns < -0.20).mean()
        }
        
        print(f"  ✓ {col}: E[ret]={results[col]['mean']:.2%}, VaR95={results[col]['var_95']:.2%}")
    
    return pd.DataFrame(results).T


def create_score_matrix(val_scores, qual_scores, risk_metrics, mc_results, weights):
    """
    Create combined score matrix for QUBO.
    
    Args:
        val_scores: DataFrame with valuation scores
        qual_scores: DataFrame with quality scores
        risk_metrics: DataFrame with risk metrics
        mc_results: DataFrame with Monte Carlo results
        weights: Dictionary with component weights
        
    Returns:
        DataFrame with combined scores
    """
    tickers = val_scores.index.intersection(qual_scores.index)
    tickers = tickers.intersection(risk_metrics.index)
    
    combined = pd.DataFrame(index=tickers)
    
    # Normalize each component to Z-score
    
    # 1. Expected return (Monte Carlo mean)
    if mc_results is not None and 'mean' in mc_results.columns:
        ret_data = mc_results.loc[mc_results.index.isin(tickers), 'mean']
        if len(ret_data) > 1 and ret_data.std() > 0:
            combined['return_z'] = (ret_data - ret_data.mean()) / ret_data.std()
    
    # 2. Valuation score
    if 'valuation_score' in val_scores.columns:
        val_data = val_scores.loc[val_scores.index.isin(tickers), 'valuation_score']
        if len(val_data) > 1 and val_data.std() > 0:
            combined['valuation_z'] = (val_data - val_data.mean()) / val_data.std()
    
    # 3. Quality score
    if 'quality_score' in qual_scores.columns:
        qual_data = qual_scores.loc[qual_scores.index.isin(tickers), 'quality_score']
        if len(qual_data) > 1 and qual_data.std() > 0:
            combined['quality_z'] = (qual_data - qual_data.mean()) / qual_data.std()
    
    # 4. Risk penalty (inverted: higher risk = lower score)
    if 'max_drawdown' in risk_metrics.columns:
        # Max drawdown is negative, so we already have the correct sign
        dd_data = risk_metrics.loc[risk_metrics.index.isin(tickers), 'max_drawdown']
        if len(dd_data) > 1 and dd_data.std() > 0:
            combined['risk_z'] = (dd_data - dd_data.mean()) / dd_data.std()  # More negative = worse
    
    # Weighted final score
    combined['final_score'] = (
        weights['return'] * combined.get('return_z', 0) +
        weights['valuation'] * combined.get('valuation_z', 0) +
        weights['quality'] * combined.get('quality_z', 0) +
        weights['risk_penalty'] * combined.get('risk_z', 0)  # risk_z already has correct sign
    )
    
    return combined


def simulated_annealing_selection(scores_df, n_select=1, T_init=1.0, T_min=0.001, 
                                   alpha=0.995, max_iter=10000):
    """
    Simulated Annealing for 0/1 asset selection.
    Objective: maximize total score with constraint to select exactly n_select assets.
    
    Args:
        scores_df: DataFrame with scores
        n_select: Number of assets to select
        T_init: Initial temperature
        T_min: Minimum temperature
        alpha: Cooling rate
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (selected_tickers, best_objective)
    """
    print(f"\n[8] Executando Simulated Annealing (selecionar {n_select} ativo)...")
    
    tickers = list(scores_df.index)
    n = len(tickers)
    
    if n == 0:
        print("  ✗ Sem ativos para otimizar")
        return None, None
    
    scores = scores_df['final_score'].values
    
    # Initial state: randomly select n_select assets
    np.random.seed(42)
    current_state = np.zeros(n, dtype=int)
    initial_idx = np.random.choice(n, n_select, replace=False)
    current_state[initial_idx] = 1
    
    def objective(state):
        """Objective function: sum of scores of selected assets."""
        return np.dot(state, scores)
    
    def neighbor(state):
        """Generate neighbor by swapping a selected asset with an unselected one."""
        new_state = state.copy()
        selected = np.where(state == 1)[0]
        not_selected = np.where(state == 0)[0]
        
        if len(selected) > 0 and len(not_selected) > 0:
            # Swap a selected with an unselected
            to_remove = np.random.choice(selected)
            to_add = np.random.choice(not_selected)
            new_state[to_remove] = 0
            new_state[to_add] = 1
        
        return new_state
    
    current_obj = objective(current_state)
    best_state = current_state.copy()
    best_obj = current_obj
    
    T = T_init
    
    for iteration in range(max_iter):
        # Generate neighbor
        new_state = neighbor(current_state)
        new_obj = objective(new_state)
        
        # Energy difference
        delta = new_obj - current_obj
        
        # Accept or reject
        if delta > 0:
            current_state = new_state
            current_obj = new_obj
        else:
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                current_state = new_state
                current_obj = new_obj
        
        # Update best
        if current_obj > best_obj:
            best_state = current_state.copy()
            best_obj = current_obj
        
        # Cool down
        T = T * alpha
        if T < T_min:
            break
    
    # Result
    selected_indices = np.where(best_state == 1)[0]
    selected_tickers = [tickers[i] for i in selected_indices]
    
    print(f"  ✓ Ativo selecionado: {selected_tickers}")
    print(f"  ✓ Score final: {best_obj:.4f}")
    
    return selected_tickers, best_obj


def scipy_optimization(scores_df, max_weight=0.40):
    """
    Continuous optimization with scipy for comparison.
    
    Args:
        scores_df: DataFrame with scores
        max_weight: Maximum weight per asset
        
    Returns:
        Series with optimal weights
    """
    print("\n[9] Otimização contínua (scipy) para comparação...")
    
    tickers = list(scores_df.index)
    n = len(tickers)
    scores = scores_df['final_score'].values
    
    def neg_objective(w):
        return -np.dot(w, scores)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum = 1
    ]
    
    # Bounds
    bounds = [(0, max_weight) for _ in range(n)]
    
    # Initial point
    w0 = np.ones(n) / n
    
    result = minimize(neg_objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = pd.Series(result.x, index=tickers)
        print("  ✓ Pesos ótimos:")
        for t, w in weights.items():
            if w > 0.01:
                print(f"    {t}: {w:.2%}")
        return weights
    else:
        print("  ✗ Otimização falhou")
        return None
