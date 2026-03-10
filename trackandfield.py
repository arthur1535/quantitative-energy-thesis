# pyright: reportUnknownMemberType=none, reportUnknownArgumentType=none, reportUnknownVariableType=none, reportMissingTypeArgument=none

"""
================================================================================
ANÁLISE QUANTITATIVA - TRACK&FIELD (TFCO4.SA)
================================================================================
Ativo: TFCO4.SA (B3 - Brasil)
Setor: Varejo / Moda Esportiva & Lifestyle
Benchmarks: ^BVSP (Ibovespa), IFIX.SA, USDBRL=X (Dólar)

Autor: Quant Analysis
Data: Março 2026

CONTEXTO:
- Track&Field é uma das principais marcas brasileiras de moda fitness/lifestyle
- Modelo asset-light: franquias + e-commerce + lojas próprias
- Crescimento consistente via abertura de lojas e expansão de marca
- Setor de wellness/fitness em expansão pós-pandemia
- Ticker: TFCO4.SA (Unit na B3)

METODOLOGIA (mesma do setor de petróleo EUA):
1. Coleta de dados (preços + fundamentalistas)
2. Métricas de Valuation (P/E, P/B, EV/EBITDA, FCF Yield, Dividend Yield)
3. Métricas de Qualidade (margens, ROE, ROA, alavancagem, liquidez)
4. Métricas de Risco (volatilidade, VaR, CVaR, max drawdown, Sharpe)
5. Betas e Correlações (vs Ibovespa, setor de consumo)
6. Monte Carlo com t-Student (distribuição de retornos futuros)
7. Análise Técnica (SMA, RSI)
8. Cenários de Preço (bull/base/bear)
9. Scoring Final e Relatório
================================================================================
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Mapping, Optional, cast
import sys
import os

# Forçar encoding UTF-8 no Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
import yfinance as yf  # type: ignore
from scipy import stats  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.stats import t  # type: ignore

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

TICKER = 'TFCO4.SA'
TICKER_NOME = 'Track&Field'

# Peers / Comparáveis no setor de varejo/consumo/moda (B3)
PEERS = ['ALPA4.SA', 'SBFG3.SA', 'LREN3.SA', 'GRND3.SA', 'VIVA3.SA']
PEER_NAMES = {
    'TFCO4.SA': 'Track&Field',
    'ALPA4.SA': 'Alpargatas',
    'SBFG3.SA': 'Grupo SBF/Centauro',
    'LREN3.SA': 'Lojas Renner',
    'GRND3.SA': 'Grendene',
    'VIVA3.SA': 'Vivara',
}

# Benchmarks
BENCHMARKS = ['^BVSP', 'USDBRL=X']
ALL_TICKERS = [TICKER] + PEERS + BENCHMARKS

# Período de análise
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5 * 365)  # 5 anos (TFCO4 IPO em 2020)

# Taxa livre de risco (Selic ~ 14.25% a.a. em 2026)
TAXA_SELIC = 0.1425
TAXA_CDI_DIARIA = (1 + TAXA_SELIC) ** (1 / 252) - 1

# Pesos para scoring final (mesma estrutura do setor de petróleo)
WEIGHTS = {
    'return': 0.25,
    'valuation': 0.25,
    'quality': 0.25,
    'risk_penalty': 0.25,
}

# Cenários
SCENARIOS = {
    'base': {'prob': 0.50, 'ibov_shock': 0.0, 'selic_impact': 0.0},
    'bull': {'prob': 0.25, 'ibov_shock': 0.20, 'selic_impact': -0.03},
    'bear': {'prob': 0.25, 'ibov_shock': -0.20, 'selic_impact': 0.03},
}

print("=" * 70)
print("ANÁLISE QUANTITATIVA - TRACK&FIELD (TFCO4.SA)")
print("=" * 70)
print(f"📊 Data da análise: {datetime.now().strftime('%Y-%m-%d')}")
print(f"📅 Período: {START_DATE.strftime('%Y-%m-%d')} a {END_DATE.strftime('%Y-%m-%d')}")
print(f"🎯 Ativo principal: {TICKER} ({TICKER_NOME})")
print(f"📈 Peers: {[PEER_NAMES.get(p, p) for p in PEERS]}")
print(f"💰 Selic: {TAXA_SELIC:.2%} a.a.")
print("=" * 70)


# ==============================================================================
# SEÇÃO 1: COLETA DE DADOS
# ==============================================================================

def fetch_price_data(tickers, start, end):
    """Baixa dados de preços via yfinance com tratamento de erros."""
    print("\n[1] 📥 Baixando dados de preços...")

    data = {}
    for ticker in tickers:
        try:
            df_raw = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )

            if df_raw is not None and not df_raw.empty and len(df_raw) > 50:
                df = cast(pd.DataFrame, df_raw)
                if isinstance(df.columns, pd.MultiIndex):
                    close_data = (
                        df['Close'][ticker]
                        if ticker in df['Close'].columns
                        else df['Close'].iloc[:, 0]
                    )
                else:
                    close_data = df['Close']
                data[ticker] = close_data
                nome = PEER_NAMES.get(ticker, ticker)
                print(f"  ✅ {nome} ({ticker}): {len(df)} registros")
            else:
                print(f"  ❌ {ticker}: dados insuficientes")
        except Exception as e:
            print(f"  ❌ {ticker}: erro - {str(e)[:50]}")

    if not data:
        raise ValueError("Nenhum dado de preço obtido!")

    prices_df = pd.DataFrame(data)
    prices_df = prices_df.dropna(how='all').ffill().bfill()
    return prices_df


def fetch_fundamental_data(tickers):
    """Extrai dados fundamentalistas via yfinance."""
    print("\n[2] 📥 Baixando dados fundamentalistas...")

    fundamentals = {}
    for ticker in tickers:
        try:
            t_obj = yf.Ticker(ticker)
            info = t_obj.info

            fund = {
                'marketCap': info.get('marketCap', np.nan),
                'enterpriseValue': info.get('enterpriseValue', np.nan),
                'trailingPE': info.get('trailingPE', np.nan),
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'enterpriseToRevenue': info.get('enterpriseToRevenue', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'operatingMargins': info.get('operatingMargins', np.nan),
                'grossMargins': info.get('grossMargins', np.nan),
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
                'revenueGrowth': info.get('revenueGrowth', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
            }

            fundamentals[ticker] = fund
            nome = PEER_NAMES.get(ticker, ticker)
            print(f"  ✅ {nome} ({ticker}): dados obtidos")

        except Exception as e:
            print(f"  ❌ {ticker}: erro - {str(e)[:50]}")
            fundamentals[ticker] = {}

    return pd.DataFrame(fundamentals).T


# ==============================================================================
# SEÇÃO 2: MÉTRICAS DE VALUATION
# ==============================================================================

def calculate_valuation_metrics(fund_df):
    """Calcula métricas de valuation."""
    print("\n[3] 📊 Calculando métricas de valuation...")

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

    # EV/Revenue
    val['ev_revenue'] = fund_df['enterpriseToRevenue']

    # P/E
    val['pe_ratio'] = fund_df['trailingPE']
    val['forward_pe'] = fund_df['forwardPE']

    # P/B
    val['pb_ratio'] = fund_df['priceToBook']

    # P/FCF = Market Cap / FCF
    val['p_fcf'] = fund_df['marketCap'] / fund_df['freeCashflow']
    val['p_fcf'] = val['p_fcf'].replace([np.inf, -np.inf], np.nan)

    # Dividend Yield
    val['div_yield'] = fund_df['dividendYield']

    # Crescimento
    val['revenue_growth'] = fund_df['revenueGrowth']
    val['earnings_growth'] = fund_df['earningsGrowth']

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
    print("\n[4] 📊 Calculando métricas de qualidade...")

    qual = pd.DataFrame(index=fund_df.index)

    # Margens
    qual['gross_margin'] = fund_df['grossMargins']
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

    # Crescimento
    qual['revenue_growth'] = fund_df['revenueGrowth']
    qual['earnings_growth'] = fund_df['earningsGrowth']

    return qual


def calculate_quality_score(qual_df):
    """Cria score composto de qualidade."""

    # Métricas onde MAIOR é melhor
    higher_better = [
        'gross_margin', 'profit_margin', 'operating_margin',
        'roe', 'roa', 'fcf_cfo_ratio', 'fcf_margin',
        'current_ratio', 'quick_ratio',
        'revenue_growth', 'earnings_growth',
    ]

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
    print("\n[5] 📊 Calculando métricas de risco...")

    risk = pd.DataFrame(index=returns.columns)

    # Retorno anualizado
    risk['ret_annual'] = returns.mean() * periods_year

    # Volatilidade anualizada
    risk['vol_annual'] = returns.std() * np.sqrt(periods_year)

    # Sharpe (usando Selic como rf)
    rf = TAXA_SELIC
    risk['sharpe'] = (risk['ret_annual'] - rf) / risk['vol_annual']

    # Max Drawdown
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


def calculate_betas(returns, benchmark_col='^BVSP'):
    """Calcula betas em relação ao benchmark (Ibovespa)."""
    print(f"\n[6] 📊 Calculando betas vs {benchmark_col}...")

    if benchmark_col not in returns.columns:
        print(f"  ❌ Benchmark {benchmark_col} não disponível")
        return pd.Series(dtype=float)

    betas = {}

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


def multifactor_regression(returns, stock, factors=['^BVSP', 'USDBRL=X']):
    """Regressão multifatorial: ret_stock ~ alpha + b1*IBOV + b2*USDBRL + ..."""

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
        'pvalues': {f: model.pvalues[f] for f in available},
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
# SEÇÃO 5: MONTE CARLO COM t-STUDENT
# ==============================================================================

def monte_carlo_simulation(returns, n_simulations=10000, horizon_days=252):
    """Simulação Monte Carlo para distribuição de retornos futuros."""
    print("\n[7] 🎲 Executando Monte Carlo (10k simulações com t-Student)...")

    results = {}
    np.random.seed(42)
    df_t = 5  # graus de liberdade (caudas gordas)

    for col in returns.columns:
        ret = returns[col].dropna()
        if len(ret) < 100:
            continue

        mu = ret.mean()
        sigma = ret.std()

        # Simular retornos diários com distribuição t de Student
        simulated = t.rvs(df_t, loc=mu, scale=sigma, size=(n_simulations, horizon_days))

        # Retorno total no horizonte
        total_returns = np.exp(simulated.sum(axis=1)) - 1

        results[col] = {
            'mean': total_returns.mean(),
            'median': np.median(total_returns),
            'std': total_returns.std(),
            'var_95': np.percentile(total_returns, 5),
            'var_99': np.percentile(total_returns, 1),
            'cvar_95': total_returns[
                total_returns <= np.percentile(total_returns, 5)
            ].mean(),
            'prob_positive': (total_returns > 0).mean(),
            'prob_gt_10pct': (total_returns > 0.10).mean(),
            'prob_gt_cdi': (total_returns > TAXA_SELIC).mean(),
            'prob_lt_minus_20pct': (total_returns < -0.20).mean(),
        }

        nome = PEER_NAMES.get(col, col)
        print(
            f"  ✅ {nome}: E[ret]={results[col]['mean']:.2%}, "
            f"VaR95={results[col]['var_95']:.2%}, "
            f"P(>CDI)={results[col]['prob_gt_cdi']:.1%}"
        )

    return pd.DataFrame(results).T


# ==============================================================================
# SEÇÃO 6: SCORING COMBINADO E SIMULATED ANNEALING
# ==============================================================================

def create_score_matrix(val_scores, qual_scores, risk_metrics, mc_results, weights):
    """Cria matriz de scores combinados."""

    tickers = val_scores.index.intersection(qual_scores.index)
    tickers = tickers.intersection(risk_metrics.index)

    combined = pd.DataFrame(index=tickers)

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
        dd_data = risk_metrics.loc[risk_metrics.index.isin(tickers), 'max_drawdown']
        if len(dd_data) > 1 and dd_data.std() > 0:
            combined['risk_z'] = (dd_data - dd_data.mean()) / dd_data.std()

    # Score final ponderado
    combined['final_score'] = (
        weights['return'] * combined.get('return_z', 0)
        + weights['valuation'] * combined.get('valuation_z', 0)
        + weights['quality'] * combined.get('quality_z', 0)
        + weights['risk_penalty'] * combined.get('risk_z', 0)
    )

    return combined


def simulated_annealing_selection(
    scores_df, n_select=1, T_init=1.0, T_min=0.001, alpha=0.995, max_iter=10000
):
    """
    Simulated Annealing para seleção 0/1 de ativos.
    Objetivo: maximizar score total selecionando exatamente n_select ativos.
    """
    print(f"\n[8] 🔥 Executando Simulated Annealing (selecionar {n_select} ativo)...")

    tickers = list(scores_df.index)
    n = len(tickers)

    if n == 0:
        print("  ❌ Sem ativos para otimizar")
        return None, None

    scores = scores_df['final_score'].values

    # Estado inicial: selecionar aleatoriamente n_select ativos
    np.random.seed(42)
    current_state = np.zeros(n, dtype=int)
    initial_idx = np.random.choice(n, n_select, replace=False)
    current_state[initial_idx] = 1

    def objective(state):
        return np.dot(state, scores)

    def neighbor(state):
        new_state = state.copy()
        selected = np.where(state == 1)[0]
        not_selected = np.where(state == 0)[0]

        if len(selected) > 0 and len(not_selected) > 0:
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
        new_state = neighbor(current_state)
        new_obj = objective(new_state)

        delta = new_obj - current_obj

        if delta > 0:
            current_state = new_state
            current_obj = new_obj
        else:
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                current_state = new_state
                current_obj = new_obj

        if current_obj > best_obj:
            best_state = current_state.copy()
            best_obj = current_obj

        T = T * alpha
        if T < T_min:
            break

    selected_indices = np.where(best_state == 1)[0]
    selected_tickers = [tickers[i] for i in selected_indices]

    nome = PEER_NAMES.get(selected_tickers[0], selected_tickers[0]) if selected_tickers else 'N/A'
    print(f"  ✅ Ativo selecionado: {nome} ({selected_tickers})")
    print(f"  ✅ Score final: {best_obj:.4f}")

    return selected_tickers, best_obj


def scipy_optimization(scores_df, max_weight=0.40):
    """Otimização contínua com scipy para composição de carteira."""
    print("\n[9] 🧮 Otimização contínua (scipy) para carteira ótima...")

    tickers = list(scores_df.index)
    n = len(tickers)
    scores = scores_df['final_score'].values

    def neg_objective(w):
        return -np.dot(w, scores)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0, max_weight) for _ in range(n)]
    w0 = np.ones(n) / n

    result = minimize(neg_objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        weights = pd.Series(result.x, index=tickers)
        print("  ✅ Pesos ótimos:")
        for ticker, w in weights.items():
            if w > 0.01:
                nome = PEER_NAMES.get(str(ticker), str(ticker))
                print(f"    {nome} ({ticker}): {w:.2%}")
        return weights
    else:
        print("  ❌ Otimização falhou")
        return None


# ==============================================================================
# SEÇÃO 7: ANÁLISE DETALHADA TRACK&FIELD
# ==============================================================================

def analise_detalhada_trackandfield(prices_all, fund_df, returns_all):
    """Análise específica e detalhada da Track&Field (TFCO4.SA)."""

    print("\n" + "=" * 70)
    print("        ANÁLISE DETALHADA - TRACK&FIELD (TFCO4.SA)")
    print("=" * 70)

    if TICKER not in prices_all.columns:
        print("  ❌ Dados de preço da TFCO4 não disponíveis")
        return

    prices = prices_all[TICKER]
    current_price = prices.iloc[-1]

    # --- Preço e Faixa ---
    print(f"\n💰 Preço atual TFCO4: R${current_price:.2f}")

    # .last() removido em pandas recentes; usar iloc com índice temporal
    prices_52w = prices.iloc[-252:] if len(prices) >= 252 else prices
    max_52w = prices_52w.max()
    min_52w = prices_52w.min()
    max_all = prices.max()
    min_all = prices.min()

    print(f"\n--- Faixa de Preço ---")
    print(f"52 semanas: R${min_52w:.2f} - R${max_52w:.2f}")
    print(f"Todo período: R${min_all:.2f} - R${max_all:.2f}")

    pct_52w = (current_price - min_52w) / (max_52w - min_52w) * 100
    pct_all = (current_price - min_all) / (max_all - min_all) * 100

    print(f"\n--- Posição na Faixa ---")
    print(f"52 semanas: {pct_52w:.1f}% (0%=mínimo, 100%=máximo)")
    print(f"Todo período: {pct_all:.1f}%")

    # --- Múltiplos ---
    print(f"\n--- Múltiplos de Valuation ---")
    if TICKER in fund_df.index:
        info = fund_df.loc[TICKER]
        pe = info.get('trailingPE', np.nan)
        forward_pe = info.get('forwardPE', np.nan)
        pb = info.get('priceToBook', np.nan)
        ev_ebitda = info.get('enterpriseToEbitda', np.nan)
        ev_revenue = info.get('enterpriseToRevenue', np.nan)
        market_cap = info.get('marketCap', np.nan)
        enterprise_value = info.get('enterpriseValue', np.nan)
        div_yield = info.get('dividendYield', np.nan)

        print(f"P/E Trailing: {pe:.2f}" if pd.notna(pe) else "P/E Trailing: N/A")
        print(f"P/E Forward: {forward_pe:.2f}" if pd.notna(forward_pe) else "P/E Forward: N/A")
        print(f"P/B: {pb:.2f}" if pd.notna(pb) else "P/B: N/A")
        print(f"EV/EBITDA: {ev_ebitda:.2f}" if pd.notna(ev_ebitda) else "EV/EBITDA: N/A")
        print(f"EV/Revenue: {ev_revenue:.2f}" if pd.notna(ev_revenue) else "EV/Revenue: N/A")
        print(f"Dividend Yield: {div_yield:.2%}" if pd.notna(div_yield) else "Dividend Yield: N/A")
        print(f"\nMarket Cap: R${market_cap/1e9:.2f}B" if pd.notna(market_cap) else "Market Cap: N/A")
        print(f"Enterprise Value: R${enterprise_value/1e9:.2f}B" if pd.notna(enterprise_value) else "EV: N/A")

    # --- Retornos ---
    if TICKER in returns_all.columns:
        returns = returns_all[TICKER].dropna()

        def calc_return(prices_s, days):
            if len(prices_s) >= days:
                return (prices_s.iloc[-1] / prices_s.iloc[-days] - 1) * 100
            return np.nan

        ret_1m = calc_return(prices, 21)
        ret_3m = calc_return(prices, 63)
        ret_6m = calc_return(prices, 126)
        ret_1y = calc_return(prices, 252)
        ret_2y = calc_return(prices, 504)

        print(f"\n--- Retornos Acumulados ---")
        print(f"1 mês: {ret_1m:.1f}%" if pd.notna(ret_1m) else "1 mês: N/A")
        print(f"3 meses: {ret_3m:.1f}%" if pd.notna(ret_3m) else "3 meses: N/A")
        print(f"6 meses: {ret_6m:.1f}%" if pd.notna(ret_6m) else "6 meses: N/A")
        print(f"1 ano: {ret_1y:.1f}%" if pd.notna(ret_1y) else "1 ano: N/A")
        print(f"2 anos: {ret_2y:.1f}%" if pd.notna(ret_2y) else "2 anos: N/A")

        # CDI acumulado 1 ano
        cdi_1y = (1 + TAXA_SELIC) - 1
        print(f"\nCDI 1 ano: {cdi_1y * 100:.1f}%")
        if pd.notna(ret_1y):
            alpha_vs_cdi = ret_1y - cdi_1y * 100
            print(f"Alpha vs CDI (1 ano): {alpha_vs_cdi:+.1f}%")

        # Volatilidade
        vol_annual = returns.std() * np.sqrt(252) * 100
        print(f"\n--- Volatilidade Anualizada ---")
        print(f"Período completo: {vol_annual:.1f}%")

        # Drawdown
        cumulative = np.exp(returns.cumsum())
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        current_dd = drawdown.iloc[-1] * 100

        print(f"\n--- Drawdown ---")
        print(f"Max Drawdown: {max_dd:.1f}%")
        print(f"Drawdown Atual: {current_dd:.1f}%")

        # VaR e CVaR
        var_95 = np.percentile(returns, 5) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100

        print(f"\n--- VaR/CVaR Diário (95%) ---")
        print(f"VaR 95%: {var_95:.2f}%")
        print(f"CVaR 95%: {cvar_95:.2f}%")

    # --- Correlações ---
    print(f"\n--- Correlações (retornos diários) ---")
    available_cols = [c for c in [TICKER, '^BVSP', 'USDBRL=X'] + PEERS if c in returns_all.columns]
    if len(available_cols) > 1:
        corr_data = returns_all[available_cols].dropna()
        if len(corr_data) > 100:
            corr = corr_data.corr()
            if TICKER in corr.index:
                for other in corr.columns:
                    if other != TICKER:
                        nome = PEER_NAMES.get(other, other)
                        print(f"TFCO4 vs {nome}: {corr.loc[TICKER, other]:.3f}")

    # --- Betas ---
    if TICKER in returns_all.columns and '^BVSP' in returns_all.columns:
        common = returns_all[[TICKER, '^BVSP']].dropna()
        if len(common) > 60:
            X = sm.add_constant(common['^BVSP'])
            model = sm.OLS(common[TICKER], X).fit()
            beta_ibov = model.params['^BVSP']
            r2_ibov = model.rsquared
            alpha_ibov = model.params['const'] * 252

            print(f"\n--- Beta e Alpha vs Ibovespa ---")
            print(f"Beta: {beta_ibov:.3f}")
            print(f"R²: {r2_ibov:.3f}")
            print(f"Alpha (anualizado): {alpha_ibov:.4f}")

    # --- Análise Técnica ---
    print(f"\n--- Análise Técnica ---")
    sma_20 = prices.rolling(20).mean().iloc[-1]
    sma_50 = prices.rolling(50).mean().iloc[-1]
    sma_200 = prices.rolling(200).mean().iloc[-1]

    print(f"Preço: R${current_price:.2f}")
    print(f"SMA 20: R${sma_20:.2f} ({'acima' if current_price > sma_20 else 'abaixo'})")
    print(f"SMA 50: R${sma_50:.2f} ({'acima' if current_price > sma_50 else 'abaixo'})")
    print(f"SMA 200: R${sma_200:.2f} ({'acima' if current_price > sma_200 else 'abaixo'})")

    if sma_20 > sma_50 > sma_200:
        trend = "📈 ALTA (Golden Cross)"
    elif sma_20 < sma_50 < sma_200:
        trend = "📉 BAIXA (Death Cross)"
    else:
        trend = "➡️ LATERAL/INDEFINIDA"
    print(f"Tendência: {trend}")

    # RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    print(f"\nRSI (14 dias): {current_rsi:.1f}")
    if current_rsi > 70:
        print("Status: ⚠️ SOBRECOMPRADO")
    elif current_rsi < 30:
        print("Status: ✅ SOBREVENDIDO")
    else:
        print("Status: ➡️ NEUTRO")

    dist_sma200 = (current_price / sma_200 - 1) * 100
    print(f"Distância da SMA 200: {dist_sma200:+.1f}%")

    # --- Cenários ---
    print(f"\n--- CENÁRIOS DE PREÇO ---")
    print(f"\nPreço atual: R${current_price:.2f}")

    if TICKER in fund_df.index:
        info = fund_df.loc[TICKER]
        ev_ebitda_val = info.get('enterpriseToEbitda', np.nan)
        pe_val = info.get('trailingPE', np.nan)

        if pd.notna(ev_ebitda_val) and pd.notna(pe_val):
            print(f"Múltiplos atuais: EV/EBITDA={ev_ebitda_val:.1f}x, P/E={pe_val:.1f}x")

            # Base
            target_base = current_price * 1.12
            print(f"\n[CENÁRIO BASE - Crescimento orgânico]")
            print(f"  Premissas: Crescimento de SSS 8-10%, abertura de 30-40 lojas/ano")
            print(f"  Selic lateral, consumo estável")
            print(f"  Preço-alvo: R${target_base:.2f} (+12%)")
            print(f"  Prob: 50%")

            # Bull
            mult_bull = min(ev_ebitda_val * 1.25, ev_ebitda_val + 3)
            target_bull = current_price * (mult_bull / ev_ebitda_val) * 1.15
            print(f"\n[CENÁRIO BULL - Expansão acelerada + queda de juros]")
            print(f"  Premissas: Selic caindo para 11-12%, consumo aquecido")
            print(f"  Expansão de e-commerce, novos mercados, expansão de margem")
            print(f"  Expansão de múltiplo: EV/EBITDA para {mult_bull:.1f}x")
            print(f"  Preço-alvo: R${target_bull:.2f} (+{(target_bull / current_price - 1) * 100:.0f}%)")
            print(f"  Prob: 25%")

            # Bear
            mult_bear = max(ev_ebitda_val * 0.7, ev_ebitda_val - 3)
            target_bear = current_price * (mult_bear / ev_ebitda_val) * 0.90
            print(f"\n[CENÁRIO BEAR - Recessão + Selic alta]")
            print(f"  Premissas: Selic subindo para 15%+, desaceleração do consumo")
            print(f"  Pressão em margens, SSS negativo")
            print(f"  Contração de múltiplo: EV/EBITDA para {mult_bear:.1f}x")
            print(f"  Preço-alvo: R${target_bear:.2f} ({(target_bear / current_price - 1) * 100:.0f}%)")
            print(f"  Prob: 25%")

            # Valor esperado
            expected_value = 0.50 * target_base + 0.25 * target_bull + 0.25 * target_bear
            expected_return = (expected_value / current_price - 1) * 100

            print(f"\n>>> 🎯 VALOR ESPERADO: R${expected_value:.2f} ({expected_return:+.1f}%)")

    return current_price, pct_52w, pct_all


# ==============================================================================
# SEÇÃO 8: GERAÇÃO DE RELATÓRIO
# ==============================================================================

def format_table(df, title, float_format='.2f'):
    """Formata DataFrame para exibição."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print('=' * 70)

    formatted = df.copy()
    for col in formatted.columns:
        if formatted[col].dtype in ['float64', 'float32']:
            formatted[col] = formatted[col].apply(
                lambda x: f'{x:{float_format}}' if pd.notna(x) else 'N/A'
            )
    print(formatted.to_string())
    return formatted


def generate_report(
    val_df, qual_df, risk_df, beta_ibov, mc_results,
    combined_scores, selected_ticker, mf_results, tfco_analysis
):
    """Gera relatório final no console."""

    print("\n")
    print("=" * 70)
    print("           RELATÓRIO FINAL - ANÁLISE QUANTITATIVA")
    print("           TRACK&FIELD (TFCO4.SA) vs PEERS")
    print("=" * 70)

    # Executive Summary
    print("\n" + "-" * 70)
    print("EXECUTIVE SUMMARY")
    print("-" * 70)

    if selected_ticker:
        nome = PEER_NAMES.get(selected_ticker[0], selected_ticker[0])
        print(f"• 🏆 ATIVO SELECIONADO (SA/QUBO): {nome} ({selected_ticker[0]})")

    if combined_scores is not None and 'final_score' in combined_scores.columns:
        ranking = combined_scores['final_score'].sort_values(ascending=False)
        ranking_names = [PEER_NAMES.get(t, t) for t in ranking.index.tolist()]
        print(f"• 📊 RANKING COMPLETO: {' > '.join(ranking_names)}")

    print("• Análise baseada em: Valuation, Qualidade, Risco, Monte Carlo")
    print("• Método de otimização: Simulated Annealing (quantum-inspired)")
    print(f"• Taxa livre de risco (Selic): {TAXA_SELIC:.2%} a.a.")

    # Tabela de Valuation
    if val_df is not None:
        cols_show = ['earnings_yield', 'fcf_yield', 'ev_ebitda', 'pe_ratio', 'pb_ratio',
                     'div_yield', 'revenue_growth']
        cols_avail = [c for c in cols_show if c in val_df.columns]
        if cols_avail:
            display_df = val_df[cols_avail].copy()
            display_df.index = [PEER_NAMES.get(t, t) for t in display_df.index]
            format_table(display_df.round(4), "📊 MÉTRICAS DE VALUATION")

    # Tabela de Qualidade
    if qual_df is not None:
        cols_show = ['gross_margin', 'profit_margin', 'operating_margin',
                     'roe', 'roa', 'fcf_margin', 'debt_to_equity',
                     'net_debt_ebitda', 'current_ratio']
        cols_avail = [c for c in cols_show if c in qual_df.columns]
        if cols_avail:
            display_df = qual_df[cols_avail].copy()
            display_df.index = [PEER_NAMES.get(t, t) for t in display_df.index]
            format_table(display_df.round(4), "⚙️ MÉTRICAS DE QUALIDADE")

    # Tabela de Risco
    if risk_df is not None:
        cols_show = ['ret_annual', 'vol_annual', 'sharpe', 'max_drawdown', 'var_95', 'cvar_95']
        cols_avail = [c for c in cols_show if c in risk_df.columns]
        if cols_avail:
            display_df = risk_df[cols_avail].copy()
            display_df.index = [PEER_NAMES.get(t, t) for t in display_df.index]
            format_table(display_df.round(4), "⚠️ MÉTRICAS DE RISCO")

    # Betas
    print("\n" + "-" * 70)
    print("📈 BETAS (SENSIBILIDADE AO IBOVESPA)")
    print("-" * 70)
    if beta_ibov is not None:
        for ticker, b in beta_ibov.items():
            if ticker in [TICKER] + PEERS:
                nome = PEER_NAMES.get(ticker, ticker)
                print(f"  {nome}: {b:.3f}" if pd.notna(b) else f"  {nome}: N/A")

    # Hipóteses
    print("\n" + "-" * 70)
    print("🔬 TESTE DAS HIPÓTESES")
    print("-" * 70)

    print("\nH1: Track&Field tem melhor qualidade (margens + ROE) que peers?")
    if qual_df is not None and 'roe' in qual_df.columns:
        tfco_roe = qual_df.loc[TICKER, 'roe'] if TICKER in qual_df.index else np.nan
        peers_roe = qual_df.loc[qual_df.index.isin(PEERS), 'roe'].mean()
        if pd.notna(tfco_roe) and pd.notna(peers_roe):
            print(f"  TFCO4 ROE: {tfco_roe:.2%}")
            print(f"  Média Peers ROE: {peers_roe:.2%}")
            if tfco_roe > peers_roe:
                print("  → ✅ CONFIRMADO: Track&Field tem ROE superior aos peers")
            else:
                print("  → ❌ NÃO CONFIRMADO: Peers têm ROE similar ou maior")

    print("\nH2: Track&Field tem valuation mais atrativo que Alpargatas e Vivara?")
    if val_df is not None and 'ev_ebitda' in val_df.columns:
        for comp in ['ALPA4.SA', 'VIVA3.SA', TICKER]:
            if comp in val_df.index:
                nome = PEER_NAMES.get(comp, comp)
                ev = val_df.loc[comp, 'ev_ebitda']
                print(f"  {nome} EV/EBITDA: {ev:.1f}x" if pd.notna(ev) else f"  {nome}: N/A")

    print("\nH3: Track&Field tem menor risco (volatilidade/drawdown) que o setor?")
    if risk_df is not None and 'vol_annual' in risk_df.columns:
        tfco_vol = risk_df.loc[TICKER, 'vol_annual'] if TICKER in risk_df.index else np.nan
        peers_vol = risk_df.loc[risk_df.index.isin(PEERS), 'vol_annual'].mean()
        if pd.notna(tfco_vol) and pd.notna(peers_vol):
            print(f"  TFCO4 Vol: {tfco_vol:.2%}")
            print(f"  Média Peers Vol: {peers_vol:.2%}")
            if tfco_vol < peers_vol:
                print("  → ✅ CONFIRMADO: Track&Field tem menor volatilidade")
            else:
                print("  → ❌ NÃO CONFIRMADO: Volatilidade similar ou maior que peers")

    # Monte Carlo
    if mc_results is not None:
        display_df = mc_results.copy()
        display_df.index = [PEER_NAMES.get(t, t) for t in display_df.index]
        format_table(display_df.round(4), "🎲 MONTE CARLO - DISTRIBUIÇÃO DE RETORNOS 12M")

    # Scores Combinados
    if combined_scores is not None:
        display_df = combined_scores.copy()
        display_df.index = [PEER_NAMES.get(t, t) for t in display_df.index]
        format_table(display_df.round(4), "🏅 SCORES COMBINADOS (Z-SCORE)")

    # Regressão Multifatorial
    print("\n" + "-" * 70)
    print("📐 REGRESSÃO MULTIFATORIAL")
    print("-" * 70)
    if mf_results:
        for ticker, res in mf_results.items():
            if res:
                nome = PEER_NAMES.get(ticker, ticker)
                print(f"\n{nome} ({ticker}):")
                print(f"  Alpha (anualizado): {res['alpha'] * 252:.4f}")
                for factor, beta in res['betas'].items():
                    pval = res['pvalues'][factor]
                    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    print(f"  Beta {factor}: {beta:.4f} {sig}")
                print(f"  R²: {res['r_squared']:.4f}")

    # Score Breakdown do Vencedor
    print("\n" + "-" * 70)
    print("🏆 SCORE BREAKDOWN - ATIVO SELECIONADO")
    print("-" * 70)
    if selected_ticker and combined_scores is not None:
        winner = selected_ticker[0]
        nome = PEER_NAMES.get(winner, winner)
        if winner in combined_scores.index:
            print(f"\n{nome} ({winner}):")
            for col in combined_scores.columns:
                val = combined_scores.loc[winner, col]
                print(f"  {col}: {val:.4f}" if pd.notna(val) else f"  {col}: N/A")

    # Conclusão
    print("\n" + "-" * 70)
    print("📋 CONCLUSÃO E RECOMENDAÇÃO CONDICIONAL")
    print("-" * 70)
    print(f"""
NOTA: Esta análise NÃO é recomendação de compra. É um framework quantitativo
para apoio à decisão, com limitações importantes listadas abaixo.

=== TESE TRACK&FIELD (TFCO4.SA) ===

POR QUE TRACK&FIELD PODE SER INTERESSANTE:
• 🏋️ Marca forte no segmento fitness/lifestyle premium brasileiro
• 📈 Modelo asset-light: franquias reduzem necessidade de capital próprio
• 💰 Boas margens operacionais para o setor de varejo brasileiro
• 🛒 Crescimento de e-commerce + D2C (direct-to-consumer)
• 🏪 Pipeline de abertura de lojas sustentável (30-50/ano)
• 💵 Geração de caixa consistente e distribuição de dividendos
• 🌱 Tendência secular de wellness/saúde favorece a marca

RISCOS PRINCIPAIS:
• 📊 Alta sensibilidade ao ciclo econômico brasileiro (Selic, emprego)
• 👔 Setor de varejo de moda é competitivo e cíclico
• 💱 Pressão de custo por importação de matéria-prima (exposição ao dólar)
• 📉 Small/mid-cap com liquidez mais restrita na B3
• 🏬 Dependência de shopping centers (risco de mudança de hábito)
• 🔄 Risco de execução na expansão acelerada

QUANDO PREFERIR CADA ATIVO:

• TFCO4 (Track&Field): Se acredita em crescimento do consumo premium
  e tendência de wellness no Brasil. Boa opção para exposição ao setor
  com margens acima da média.

• ALPA4 (Alpargatas/Havaianas): Para exposição a uma marca global
  brasileira (Havaianas), com presença internacional e perfil mais
  defensivo no segmento de calçados.

• SBFG3 (Centauro/Nike): Para exposição ao varejo esportivo de massa,
  com operação Nike no Brasil e forte presença digital.

• LREN3 (Renner): Para exposição ao varejo de moda de massa com
  maior liquidez e resiliência de balanço.

• GRND3 (Grendene): Para perfil mais defensivo: caixa líquido,
  dividendos altos, menor beta.

• VIVA3 (Vivara): Para exposição ao segmento de luxo acessível
  (joias/relógios) com crescimento via Life by Vivara.

💡 ESTRATÉGIA SUGERIDA PARA TFCO4:
1. Se preço na parte baixa do range (< 40%): Posição cheia
2. Se no meio do range (40-60%): Entrada parcial, acumular em correções
3. Se na parte alta do range (> 60%): Esperar correção de 10-15%
4. Sempre comparar retorno esperado vs CDI ({TAXA_SELIC:.1%} a.a.)
5. Posição máxima sugerida: 5-8% do portfólio (small/mid-cap)
""")

    # Limitações
    print("\n" + "-" * 70)
    print("⚠️ LIMITAÇÕES")
    print("-" * 70)
    print("""
• Dados fundamentalistas via yfinance podem estar desatualizados ou incompletos.
• FCF e EBITDA dependem de disponibilidade na API.
• Não incorpora análise de management, ESG, ou fatores qualitativos.
• TFCO4 tem histórico curto na B3 (IPO 2020), limitando análises de longo prazo.
• Monte Carlo assume distribuição t-Student (melhor que normal, mas ainda limitado).
• Betas históricos podem não refletir regime atual de Selic.
• Sem ajuste para dividendos extraordinários ou eventos corporativos.
• Peers podem ter modelos de negócio parcialmente diferentes (ex: Grendene = indústria).
• Liquidez de TFCO4 pode resultar em spreads maiores vs large-caps.
""")


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    # 1. Coleta de dados
    prices = fetch_price_data(ALL_TICKERS, START_DATE, END_DATE)
    tickers_analise = [TICKER] + PEERS
    fundamentals = fetch_fundamental_data(tickers_analise)

    # 2. Filtrar preços
    prices_stocks = prices[[c for c in tickers_analise if c in prices.columns]]
    prices_all = prices

    # 3. Calcular retornos
    returns_all = calculate_returns(prices_all)
    returns_stocks = returns_all[[c for c in tickers_analise if c in returns_all.columns]]

    # 4. Métricas de Valuation
    val_metrics = calculate_valuation_metrics(fundamentals)
    val_scores = calculate_valuation_score(val_metrics)

    # 5. Métricas de Qualidade
    qual_metrics = calculate_quality_metrics(fundamentals)
    qual_scores = calculate_quality_score(qual_metrics)

    # 6. Métricas de Risco
    risk_metrics = calculate_risk_metrics(returns_stocks)

    # 7. Betas vs Ibovespa
    beta_ibov = calculate_betas(returns_all, '^BVSP')

    # 8. Regressão Multifatorial
    mf_results = {}
    for ticker in tickers_analise:
        if ticker in returns_all.columns:
            mf_results[ticker] = multifactor_regression(
                returns_all, ticker, ['^BVSP', 'USDBRL=X']
            )

    # 9. Monte Carlo
    mc_results = monte_carlo_simulation(returns_stocks)

    # 10. Criar scores combinados
    combined_scores = create_score_matrix(
        val_scores, qual_scores, risk_metrics, mc_results, WEIGHTS
    )

    # 11. Simulated Annealing
    selected_ticker, best_score = simulated_annealing_selection(combined_scores, n_select=1)

    # 12. Otimização contínua para carteira
    optimal_weights = scipy_optimization(combined_scores)

    # 13. Análise detalhada Track&Field
    tfco_analysis = analise_detalhada_trackandfield(prices_all, fundamentals, returns_all)

    # 14. Gerar relatório final
    generate_report(
        val_df=val_metrics,
        qual_df=qual_metrics,
        risk_df=risk_metrics,
        beta_ibov=beta_ibov,
        mc_results=mc_results,
        combined_scores=combined_scores,
        selected_ticker=selected_ticker,
        mf_results=mf_results,
        tfco_analysis=tfco_analysis,
    )

    print("\n" + "=" * 70)
    print("✅ ANÁLISE CONCLUÍDA")
    print("=" * 70)
