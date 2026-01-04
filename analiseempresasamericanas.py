"""
================================================================================
ANÁLISE QUANTITATIVA - SETOR DE PETRÓLEO E ENERGIA (US)
================================================================================
Ativos: CVX, XOM, COP, SLB, HAL (+ ETFs XLE, OIH)
Benchmarks: SPY, CL=F (WTI), BZ=F (Brent)

Autor: Quant Analysis
Data: Janeiro 2026
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import t  # Student's t distribution for heavy-tailed Monte Carlo

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

TICKERS_ACOES = ['CVX', 'XOM', 'COP', 'SLB', 'HAL']
TICKERS_ETFS = ['XLE', 'OIH']
BENCHMARKS = ['SPY', 'CL=F', 'BZ=F']
ALL_TICKERS = TICKERS_ACOES + TICKERS_ETFS + BENCHMARKS

# BDRs na B3 (informativo)
BDR_MAP = {
    'CVX': 'CHVX34', 'XOM': 'EXXO34', 'COP': 'COPH34',
    'SLB': 'SLBG34', 'HAL': 'HALI34'
}

# Período de análise
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=10*365)  # 10 anos

# Pesos para scoring final
WEIGHTS = {
    'return': 0.25,
    'valuation': 0.25,
    'quality': 0.25,
    'risk_penalty': 0.25
}

# Cenários (probabilidades editáveis)
SCENARIOS = {
    'base': {'prob': 0.50, 'wti_shock': 0.0, 'spy_shock': 0.0},
    'bull': {'prob': 0.30, 'wti_shock': 0.30, 'spy_shock': 0.15},
    'bear': {'prob': 0.20, 'wti_shock': -0.30, 'spy_shock': -0.20}
}

print("="*70)
print("ANÁLISE QUANTITATIVA - SETOR PETRÓLEO/ENERGIA (US)")
print("="*70)
print(f"Data da análise: {datetime.now().strftime('%Y-%m-%d')}")
print(f"Período: {START_DATE.strftime('%Y-%m-%d')} a {END_DATE.strftime('%Y-%m-%d')}")
print(f"Ativos analisados: {TICKERS_ACOES}")
print("="*70)


# ==============================================================================
# SEÇÃO 1: COLETA DE DADOS
# ==============================================================================

def fetch_price_data(tickers, start, end):
    """Baixa dados de preços via yfinance com tratamento de erros."""
    print("\n[1] Baixando dados de preços...")
    
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty and len(df) > 100:
                # Extrair coluna Close (pode ser MultiIndex ou simples)
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
    """Extrai dados fundamentalistas via yfinance."""
    print("\n[2] Baixando dados fundamentalistas...")
    
    fundamentals = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # Extrair métricas disponíveis
            fund = {
                'marketCap': info.get('marketCap', np.nan),
                'enterpriseValue': info.get('enterpriseValue', np.nan),
                'trailingPE': info.get('trailingPE', np.nan),
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'operatingMargins': info.get('operatingMargins', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'returnOnAssets': info.get('returnOnAssets', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'currentRatio': info.get('currentRatio', np.nan),
                'quickRatio': info.get('quickRatio', np.nan),
                'freeCashflow': info.get('freeCashflow', np.nan),
                'operatingCashflow': info.get('operatingCashflow', np.nan),
                'totalDebt': info.get('totalDebt', np.nan),
                'totalCash': info.get('totalCash', np.nan),
                'ebitda': info.get('ebitda', np.nan),
                'totalRevenue': info.get('totalRevenue', np.nan),
                'dividendYield': info.get('dividendYield', np.nan),
                'payoutRatio': info.get('payoutRatio', np.nan),
                'beta': info.get('beta', np.nan),
            }
            
            fundamentals[ticker] = fund
            print(f"  ✓ {ticker}: dados obtidos")
            
        except Exception as e:
            print(f"  ✗ {ticker}: erro - {str(e)[:50]}")
            fundamentals[ticker] = {}
    
    return pd.DataFrame(fundamentals).T


# ==============================================================================
# SEÇÃO 2: MÉTRICAS DE VALUATION
# ==============================================================================

def calculate_valuation_metrics(fund_df):
    """Calcula métricas de valuation."""
    print("\n[3] Calculando métricas de valuation...")
    
    val = pd.DataFrame(index=fund_df.index)
    
    # Earnings Yield = 1 / P/E
    val['earnings_yield'] = 1 / fund_df['trailingPE']
    val['earnings_yield'] = val['earnings_yield'].replace([np.inf, -np.inf], np.nan)
    
    # FCF Yield = FCF / Market Cap
    val['fcf_yield'] = fund_df['freeCashflow'] / fund_df['marketCap']
    
    # EV/EBITDA (inverso para score - menor é melhor)
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
    """Cria score composto de valuation (Z-score normalizado)."""
    
    # Métricas onde MAIOR é melhor
    higher_better = ['earnings_yield', 'fcf_yield', 'ev_ebitda_inv', 'div_yield']
    
    scores = pd.DataFrame(index=val_df.index)
    
    for col in higher_better:
        if col in val_df.columns:
            data = val_df[col].dropna()
            if len(data) > 1 and data.std() > 0:
                z = (val_df[col] - data.mean()) / data.std()
                scores[col + '_z'] = z
    
    # Score final = média dos Z-scores
    scores['valuation_score'] = scores.mean(axis=1, skipna=True)
    
    return scores


# ==============================================================================
# SEÇÃO 3: MÉTRICAS DE QUALIDADE
# ==============================================================================

def calculate_quality_metrics(fund_df):
    """Calcula métricas de qualidade de gestão."""
    print("\n[4] Calculando métricas de qualidade...")
    
    qual = pd.DataFrame(index=fund_df.index)
    
    # Margens
    qual['profit_margin'] = fund_df['profitMargins']
    qual['operating_margin'] = fund_df['operatingMargins']
    
    # Retornos
    qual['roe'] = fund_df['returnOnEquity']
    qual['roa'] = fund_df['returnOnAssets']
    
    # FCF/CFO (disciplina de capital)
    qual['fcf_cfo_ratio'] = fund_df['freeCashflow'] / fund_df['operatingCashflow']
    qual['fcf_cfo_ratio'] = qual['fcf_cfo_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # FCF Margin = FCF / Revenue
    qual['fcf_margin'] = fund_df['freeCashflow'] / fund_df['totalRevenue']
    
    # Alavancagem
    # Debt/Equity: yfinance often returns percentage rather than ratio (e.g., 21.24 means 21.24%).
    # Convert values > 1 and <= 100 to ratios (divide by 100).
    d_to_e_raw = fund_df['debtToEquity']
    d_to_e_adj = d_to_e_raw.apply(
        lambda x: x / 100 if pd.notna(x) and 1 < x <= 100 else x
    )
    qual['debt_to_equity'] = d_to_e_adj
    qual['net_debt'] = fund_df['totalDebt'] - fund_df['totalCash']
    qual['net_debt_ebitda'] = qual['net_debt'] / fund_df['ebitda']
    qual['net_debt_ebitda'] = qual['net_debt_ebitda'].replace([np.inf, -np.inf], np.nan)
    
    # Liquidez
    qual['current_ratio'] = fund_df['currentRatio']
    qual['quick_ratio'] = fund_df['quickRatio']
    
    return qual


def calculate_quality_score(qual_df):
    """Cria score composto de qualidade."""
    
    # Métricas onde MAIOR é melhor
    higher_better = ['profit_margin', 'operating_margin', 'roe', 'roa', 
                     'fcf_cfo_ratio', 'fcf_margin', 'current_ratio', 'quick_ratio']
    
    # Métricas onde MENOR é melhor
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
                z = -1 * (qual_df[col] - data.mean()) / data.std()  # Invertido
                scores[col + '_z'] = z
    
    scores['quality_score'] = scores.mean(axis=1, skipna=True)
    
    return scores


# ==============================================================================
# SEÇÃO 4: MÉTRICAS DE RISCO
# ==============================================================================

def calculate_returns(prices):
    """Calcula retornos logarítmicos diários."""
    return np.log(prices / prices.shift(1)).dropna()


def calculate_risk_metrics(returns, periods_year=252):
    """Calcula métricas de risco para cada ativo."""
    print("\n[5] Calculando métricas de risco...")
    
    risk = pd.DataFrame(index=returns.columns)
    
    # Retorno anualizado
    risk['ret_annual'] = returns.mean() * periods_year
    
    # Volatilidade anualizada
    risk['vol_annual'] = returns.std() * np.sqrt(periods_year)
    
    # Sharpe (assumindo rf = 4% para USD)
    rf = 0.04
    risk['sharpe'] = (risk['ret_annual'] - rf) / risk['vol_annual']
    
    # Max Drawdown
    # When returns are log returns, convert to price index via exponential of cumulative sum.
    for col in returns.columns:
        prices_norm = np.exp(returns[col].cumsum())
        rolling_max = prices_norm.expanding().max()
        drawdown = (prices_norm - rolling_max) / rolling_max
        risk.loc[col, 'max_drawdown'] = drawdown.min()
    
    # VaR e CVaR (95% e 99%)
    for col in returns.columns:
        ret = returns[col].dropna()
        risk.loc[col, 'var_95'] = np.percentile(ret, 5)
        risk.loc[col, 'var_99'] = np.percentile(ret, 1)
        risk.loc[col, 'cvar_95'] = ret[ret <= np.percentile(ret, 5)].mean()
        risk.loc[col, 'cvar_99'] = ret[ret <= np.percentile(ret, 1)].mean()
    
    # Skewness e Kurtosis
    risk['skewness'] = returns.skew()
    risk['kurtosis'] = returns.kurtosis()
    
    return risk


def calculate_betas(returns, benchmark_col='SPY'):
    """Calcula betas em relação ao benchmark."""
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
        
        # Alinhar datas
        common = returns[[col, benchmark_col]].dropna()
        if len(common) < 60:
            betas[col] = np.nan
            continue
        
        # Regressão OLS
        X = sm.add_constant(common[benchmark_col])
        model = sm.OLS(common[col], X).fit()
        betas[col] = model.params[benchmark_col]
    
    return pd.Series(betas, name=f'beta_{benchmark_col}')


def multifactor_regression(returns, stock, factors=['SPY', 'CL=F', 'XLE', 'OIH']):
    """Regressão multifatorial: ret_stock ~ alpha + b1*SPY + b2*WTI + ..."""
    
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
    """Calcula beta rolling."""
    
    if stock not in returns.columns or benchmark not in returns.columns:
        return None
    
    data = returns[[stock, benchmark]].dropna()
    
    rolling_cov = data[stock].rolling(window).cov(data[benchmark])
    rolling_var = data[benchmark].rolling(window).var()
    
    return rolling_cov / rolling_var


# ==============================================================================
# SEÇÃO 5: CENÁRIOS E MONTE CARLO
# ==============================================================================

def scenario_returns(regression_results, scenarios):
    """Estima retornos esperados por cenário usando coeficientes de regressão."""
    
    scenario_ret = {}
    
    for name, params in scenarios.items():
        # Retorno = alpha + b_SPY * shock_SPY + b_WTI * shock_WTI
        expected = regression_results['alpha'] * 252  # Anualizando alpha
        
        if 'SPY' in regression_results['betas']:
            expected += regression_results['betas']['SPY'] * params['spy_shock']
        
        if 'CL=F' in regression_results['betas']:
            expected += regression_results['betas']['CL=F'] * params['wti_shock']
        
        scenario_ret[name] = expected
    
    return scenario_ret


def monte_carlo_simulation(returns, n_simulations=10000, horizon_days=252):
    """Simulação Monte Carlo para distribuição de retornos futuros."""
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
        
        # Simular retornos diários com distribuição t de Student (caudas gordas)
        simulated = t.rvs(df_t, loc=mu, scale=sigma, size=(n_simulations, horizon_days))
        
        # Retorno total no horizonte
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


# ==============================================================================
# SEÇÃO 6: QUBO / SIMULATED ANNEALING
# ==============================================================================

def create_score_matrix(val_scores, qual_scores, risk_metrics, mc_results, weights):
    """Cria matriz de scores combinados para QUBO."""
    
    tickers = val_scores.index.intersection(qual_scores.index)
    tickers = tickers.intersection(risk_metrics.index)
    
    combined = pd.DataFrame(index=tickers)
    
    # Normalizar cada componente para Z-score
    
    # 1. Retorno esperado (Monte Carlo mean)
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
    
    # 4. Risk penalty (invertido: maior risco = menor score)
    if 'max_drawdown' in risk_metrics.columns:
        # Max drawdown é negativo, então multiplicamos por -1 para penalizar
        dd_data = risk_metrics.loc[risk_metrics.index.isin(tickers), 'max_drawdown']
        if len(dd_data) > 1 and dd_data.std() > 0:
            combined['risk_z'] = (dd_data - dd_data.mean()) / dd_data.std()  # Mais negativo = pior
    
    # Score final ponderado
    combined['final_score'] = (
        weights['return'] * combined.get('return_z', 0) +
        weights['valuation'] * combined.get('valuation_z', 0) +
        weights['quality'] * combined.get('quality_z', 0) +
        weights['risk_penalty'] * combined.get('risk_z', 0)  # risk_z já está com sinal correto
    )
    
    return combined


def simulated_annealing_selection(scores_df, n_select=1, T_init=1.0, T_min=0.001, 
                                   alpha=0.995, max_iter=10000):
    """
    Simulated Annealing para seleção 0/1 de ativos.
    Objetivo: maximizar score total com restrição de selecionar exatamente n_select ativos.
    """
    print(f"\n[8] Executando Simulated Annealing (selecionar {n_select} ativo)...")
    
    tickers = list(scores_df.index)
    n = len(tickers)
    
    if n == 0:
        print("  ✗ Sem ativos para otimizar")
        return None, None
    
    scores = scores_df['final_score'].values
    
    # Estado inicial: selecionar aleatoriamente n_select ativos
    np.random.seed(42)
    current_state = np.zeros(n, dtype=int)
    initial_idx = np.random.choice(n, n_select, replace=False)
    current_state[initial_idx] = 1
    
    def objective(state):
        """Função objetivo: soma dos scores dos ativos selecionados."""
        return np.dot(state, scores)
    
    def neighbor(state):
        """Gera vizinho trocando um ativo selecionado por um não selecionado."""
        new_state = state.copy()
        selected = np.where(state == 1)[0]
        not_selected = np.where(state == 0)[0]
        
        if len(selected) > 0 and len(not_selected) > 0:
            # Trocar um selecionado por um não selecionado
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
        # Gerar vizinho
        new_state = neighbor(current_state)
        new_obj = objective(new_state)
        
        # Diferença de energia
        delta = new_obj - current_obj
        
        # Aceitar ou rejeitar
        if delta > 0:
            current_state = new_state
            current_obj = new_obj
        else:
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                current_state = new_state
                current_obj = new_obj
        
        # Atualizar melhor
        if current_obj > best_obj:
            best_state = current_state.copy()
            best_obj = current_obj
        
        # Resfriar
        T = T * alpha
        if T < T_min:
            break
    
    # Resultado
    selected_indices = np.where(best_state == 1)[0]
    selected_tickers = [tickers[i] for i in selected_indices]
    
    print(f"  ✓ Ativo selecionado: {selected_tickers}")
    print(f"  ✓ Score final: {best_obj:.4f}")
    
    return selected_tickers, best_obj


def scipy_optimization(scores_df, max_weight=0.40):
    """Otimização contínua com scipy para comparação."""
    print("\n[9] Otimização contínua (scipy) para comparação...")
    
    tickers = list(scores_df.index)
    n = len(tickers)
    scores = scores_df['final_score'].values
    
    def neg_objective(w):
        return -np.dot(w, scores)
    
    # Restrições
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Soma = 1
    ]
    
    # Limites
    bounds = [(0, max_weight) for _ in range(n)]
    
    # Ponto inicial
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


# ==============================================================================
# SEÇÃO 7: GERAÇÃO DE RELATÓRIO
# ==============================================================================

def format_table(df, title, float_format='.2f'):
    """Formata DataFrame para exibição."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)
    
    # Converter para string formatado
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
    """Gera relatório final no console."""
    
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
    
    # Tabela de Valuation
    if val_df is not None:
        cols_show = ['earnings_yield', 'fcf_yield', 'ev_ebitda', 'pe_ratio', 'div_yield']
        cols_avail = [c for c in cols_show if c in val_df.columns]
        if cols_avail:
            format_table(val_df[cols_avail].round(4), "MÉTRICAS DE VALUATION")
    
    # Tabela de Qualidade
    if qual_df is not None:
        cols_show = ['profit_margin', 'roe', 'fcf_margin', 'debt_to_equity', 'net_debt_ebitda', 'current_ratio']
        cols_avail = [c for c in cols_show if c in qual_df.columns]
        if cols_avail:
            format_table(qual_df[cols_avail].round(4), "MÉTRICAS DE QUALIDADE")
    
    # Tabela de Risco
    if risk_df is not None:
        cols_show = ['ret_annual', 'vol_annual', 'sharpe', 'max_drawdown', 'var_95', 'cvar_95']
        cols_avail = [c for c in cols_show if c in risk_df.columns]
        if cols_avail:
            format_table(risk_df[cols_avail].round(4), "MÉTRICAS DE RISCO (5 anos)")
    
    # Betas
    print("\n" + "-"*70)
    print("BETAS (SENSIBILIDADE)")
    print("-"*70)
    if beta_spy is not None:
        print("\nBeta vs SPY (mercado):")
        for t, b in beta_spy.items():
            if t in TICKERS_ACOES:
                print(f"  {t}: {b:.3f}" if pd.notna(b) else f"  {t}: N/A")
    
    if beta_wti is not None:
        print("\nBeta vs CL=F (WTI):")
        for t, b in beta_wti.items():
            if t in TICKERS_ACOES:
                print(f"  {t}: {b:.3f}" if pd.notna(b) else f"  {t}: N/A")
    
    # Teste de Hipóteses
    print("\n" + "-"*70)
    print("TESTE DAS HIPÓTESES")
    print("-"*70)
    
    print("\nH1: SLB tem maior 'torque' ao petróleo (beta WTI)?")
    if beta_wti is not None and 'SLB' in beta_wti:
        slb_beta = beta_wti.get('SLB', np.nan)
        cvx_beta = beta_wti.get('CVX', np.nan)
        xom_beta = beta_wti.get('XOM', np.nan)
        cop_beta = beta_wti.get('COP', np.nan)
        
        if pd.notna(slb_beta):
            majors_avg = np.nanmean([cvx_beta, xom_beta])
            print(f"  SLB beta WTI: {slb_beta:.3f}")
            print(f"  Média Majors (CVX/XOM): {majors_avg:.3f}")
            if slb_beta > majors_avg:
                print("  → CONFIRMADO: SLB tem maior sensibilidade ao petróleo")
            else:
                print("  → NÃO CONFIRMADO: Majors têm sensibilidade similar ou maior")
    
    print("\nH2: COP tem opcionalidade Venezuela?")
    print("  → QUALITATIVO: Não há dados públicos de claims/recebíveis via API.")
    print("  → Tratar como upside potencial em cenário de distensão geopolítica.")
    
    print("\nH3: Majors (CVX/XOM) vencem em robustez de balanço?")
    if qual_df is not None and 'debt_to_equity' in qual_df.columns:
        print("  Debt/Equity:")
        for t in ['CVX', 'XOM', 'COP', 'SLB', 'HAL']:
            if t in qual_df.index:
                val = qual_df.loc[t, 'debt_to_equity']
                print(f"    {t}: {val:.2f}" if pd.notna(val) else f"    {t}: N/A")
    
    # Monte Carlo
    if mc_results is not None:
        format_table(mc_results.round(4), "MONTE CARLO - DISTRIBUIÇÃO DE RETORNOS 12M")
    
    # Scores Combinados
    if combined_scores is not None:
        format_table(combined_scores.round(4), "SCORES COMBINADOS (Z-SCORE)")
    
    # Regressão Multifatorial
    print("\n" + "-"*70)
    print("REGRESSÃO MULTIFATORIAL")
    print("-"*70)
    if mf_results:
        for ticker, res in mf_results.items():
            if res:
                print(f"\n{ticker}:")
                print(f"  Alpha (anualizado): {res['alpha']*252:.4f}")
                for factor, beta in res['betas'].items():
                    pval = res['pvalues'][factor]
                    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    print(f"  Beta {factor}: {beta:.4f} {sig}")
                print(f"  R²: {res['r_squared']:.4f}")
    
    # Score Breakdown do Vencedor
    print("\n" + "-"*70)
    print("SCORE BREAKDOWN - ATIVO SELECIONADO")
    print("-"*70)
    if selected_ticker and combined_scores is not None:
        winner = selected_ticker[0]
        if winner in combined_scores.index:
            print(f"\n{winner}:")
            for col in combined_scores.columns:
                val = combined_scores.loc[winner, col]
                print(f"  {col}: {val:.4f}" if pd.notna(val) else f"  {col}: N/A")
    
    # Conclusão
    print("\n" + "-"*70)
    print("CONCLUSÃO E RECOMENDAÇÃO CONDICIONAL")
    print("-"*70)
    print("""
NOTA: Esta análise NÃO é recomendação de compra. É um framework quantitativo
para apoio à decisão, com limitações importantes listadas abaixo.

QUANDO ESCOLHER CADA ATIVO:

• SLB: Cenário de BULL em CAPEX do setor. Se você acredita que o ciclo de
  investimento em E&P vai se acelerar, SLB oferece maior "torque" operacional.
  RISCO: Alta volatilidade e sensibilidade a ciclo de capex.

• COP: Cenário de EVENT-DRIVEN ou normalização Venezuela. Empresa com boa
  execução operacional e potencial opcionalidade em recebíveis.
  RISCO: Menor diversificação que majors.

• CVX/XOM: Cenário BASE ou BEAR. Robustez de balanço, dividendos, menor
  volatilidade. Preferir em ambiente de incerteza macroeconômica.
  RISCO: Menor convexidade em cenário de alta do petróleo.

• HAL: Similar a SLB mas com perfil diferente de serviços. Avaliar em
  conjunto com análise de backlog e contratos.
""")
    
    # Limitações
    print("\n" + "-"*70)
    print("LIMITAÇÕES")
    print("-"*70)
    print("""
• Dados fundamentalistas via yfinance podem estar desatualizados ou incompletos.
• FCF e EBITDA dependem de disponibilidade na API.
• Não incorpora análise de management, ESG, ou fatores qualitativos.
• Claims da Venezuela (COP) tratados apenas como qualitativo.
• Monte Carlo assume distribuição normal (limitação conhecida).
• Betas históricos podem não refletir regime atual.
• Sem ajuste para dividendos extraordinários ou eventos corporativos.
""")


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Coleta de dados
    prices = fetch_price_data(ALL_TICKERS, START_DATE, END_DATE)
    fundamentals = fetch_fundamental_data(TICKERS_ACOES)
    
    # 2. Filtrar apenas ações para análise principal
    prices_stocks = prices[[c for c in TICKERS_ACOES if c in prices.columns]]
    prices_all = prices  # Manter todos para regressões
    
    # 3. Calcular retornos
    returns_all = calculate_returns(prices_all)
    returns_stocks = returns_all[[c for c in TICKERS_ACOES if c in returns_all.columns]]
    
    # 4. Métricas de Valuation
    val_metrics = calculate_valuation_metrics(fundamentals)
    val_scores = calculate_valuation_score(val_metrics)
    
    # 5. Métricas de Qualidade
    qual_metrics = calculate_quality_metrics(fundamentals)
    qual_scores = calculate_quality_score(qual_metrics)
    
    # 6. Métricas de Risco
    risk_metrics = calculate_risk_metrics(returns_stocks)
    
    # 7. Betas
    beta_spy = calculate_betas(returns_all, 'SPY')
    beta_wti = calculate_betas(returns_all, 'CL=F')
    
    # 8. Regressão Multifatorial
    mf_results = {}
    for ticker in TICKERS_ACOES:
        if ticker in returns_all.columns:
            mf_results[ticker] = multifactor_regression(returns_all, ticker, ['SPY', 'CL=F'])
    
    # 9. Monte Carlo
    mc_results = monte_carlo_simulation(returns_stocks)
    
    # 10. Criar scores combinados
    combined_scores = create_score_matrix(val_scores, qual_scores, risk_metrics, mc_results, WEIGHTS)
    
    # 11. Simulated Annealing
    selected_ticker, best_score = simulated_annealing_selection(combined_scores, n_select=1)
    
    # 12. Otimização contínua para comparação
    optimal_weights = scipy_optimization(combined_scores)
    
    # 13. Gerar relatório
    generate_report(
        val_df=val_metrics,
        qual_df=qual_metrics,
        risk_df=risk_metrics,
        beta_spy=beta_spy,
        beta_wti=beta_wti,
        mc_results=mc_results,
        combined_scores=combined_scores,
        selected_ticker=selected_ticker,
        mf_results=mf_results
    )
    
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA")
    print("="*70)
