"""
================================================================================
ANÁLISE DETALHADA - SLB (Schlumberger) / SLBG34
================================================================================
Tese: SLB como veículo para ciclo de CAPEX em petróleo + reconstrução Venezuela

CONTEXTO VENEZUELA:
- Infraestrutura de extração severamente degradada (falta de manutenção 10+ anos)
- Petróleo extrapesado (Orinoco Belt) requer tecnologia especializada
- Produção atual ~900k bpd vs potencial 3M+ bpd
- Petróleo venezuelano vendido com desconto significativo (heavy crude spread)
- Qualquer normalização = demanda massiva por serviços de oil services

SLB como beneficiária:
- Líder global em serviços de completação e estimulação
- Expertise em reservatórios complexos e heavy oil
- Presença histórica na Venezuela (antes das sanções)
- Alavancagem operacional ao CAPEX do setor

================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm

print("="*70)
print("ANÁLISE DETALHADA - SLB (Schlumberger) / SLBG34")
print("="*70)
print(f"Data da análise: {datetime.now().strftime('%Y-%m-%d')}")
print("="*70)

# ==============================================================================
# 1. COLETA DE DADOS
# ==============================================================================

def fetch_data(ticker, years=10):
    """Baixa dados históricos."""
    end = datetime.now()
    start = end - timedelta(days=years*365)
    
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"Erro ao baixar {ticker}: {e}")
        return None

def fetch_fundamentals(ticker):
    """Extrai dados fundamentalistas."""
    try:
        t = yf.Ticker(ticker)
        return t.info
    except:
        return {}

print("\n[1] Baixando dados...")

# SLB (NYSE)
slb_prices = fetch_data('SLB', 10)
slb_info = fetch_fundamentals('SLB')

# SLBG34 (B3) - BDR
slbg34_prices = fetch_data('SLBG34.SA', 5)
slbg34_info = fetch_fundamentals('SLBG34.SA')

# Benchmarks
wti = fetch_data('CL=F', 10)
xle = fetch_data('XLE', 10)
spy = fetch_data('SPY', 10)
brent = fetch_data('BZ=F', 10)

# Dólar (para análise do BDR)
usdbrl = fetch_data('USDBRL=X', 5)

print(f"  ✓ SLB: {len(slb_prices)} registros")
print(f"  ✓ SLBG34: {len(slbg34_prices) if slbg34_prices is not None else 0} registros")
print(f"  ✓ WTI: {len(wti)} registros")

# ==============================================================================
# 2. ANÁLISE DE PREÇO E VALUATION
# ==============================================================================

print("\n" + "="*70)
print("2. ANÁLISE DE PREÇO E VALUATION - SLB")
print("="*70)

# Preço atual
current_price = slb_prices['Close'].iloc[-1] if slb_prices is not None else np.nan
print(f"\nPreço atual SLB (NYSE): ${current_price:.2f}")

# Estatísticas de preço
if slb_prices is not None:
    prices = slb_prices['Close']
    
    # Máximos e mínimos
    max_52w = prices.last('252D').max()
    min_52w = prices.last('252D').min()
    max_5y = prices.last('1260D').max()
    min_5y = prices.last('1260D').min()
    max_10y = prices.max()
    min_10y = prices.min()
    
    print(f"\n--- Faixa de Preço ---")
    print(f"52 semanas: ${min_52w:.2f} - ${max_52w:.2f}")
    print(f"5 anos: ${min_5y:.2f} - ${max_5y:.2f}")
    print(f"10 anos: ${min_10y:.2f} - ${max_10y:.2f}")
    
    # Posição na faixa
    pct_52w = (current_price - min_52w) / (max_52w - min_52w) * 100
    pct_5y = (current_price - min_5y) / (max_5y - min_5y) * 100
    pct_10y = (current_price - min_10y) / (max_10y - min_10y) * 100
    
    print(f"\n--- Posição na Faixa ---")
    print(f"52 semanas: {pct_52w:.1f}% (0%=mínimo, 100%=máximo)")
    print(f"5 anos: {pct_5y:.1f}%")
    print(f"10 anos: {pct_10y:.1f}%")

# Múltiplos de valuation
print(f"\n--- Múltiplos de Valuation ---")
if slb_info:
    pe = slb_info.get('trailingPE', np.nan)
    forward_pe = slb_info.get('forwardPE', np.nan)
    pb = slb_info.get('priceToBook', np.nan)
    ev_ebitda = slb_info.get('enterpriseToEbitda', np.nan)
    ev_revenue = slb_info.get('enterpriseToRevenue', np.nan)
    
    print(f"P/E Trailing: {pe:.2f}" if pd.notna(pe) else "P/E Trailing: N/A")
    print(f"P/E Forward: {forward_pe:.2f}" if pd.notna(forward_pe) else "P/E Forward: N/A")
    print(f"P/B: {pb:.2f}" if pd.notna(pb) else "P/B: N/A")
    print(f"EV/EBITDA: {ev_ebitda:.2f}" if pd.notna(ev_ebitda) else "EV/EBITDA: N/A")
    print(f"EV/Revenue: {ev_revenue:.2f}" if pd.notna(ev_revenue) else "EV/Revenue: N/A")
    
    # Comparação histórica de múltiplos (se disponível)
    market_cap = slb_info.get('marketCap', np.nan)
    enterprise_value = slb_info.get('enterpriseValue', np.nan)
    
    print(f"\nMarket Cap: ${market_cap/1e9:.1f}B" if pd.notna(market_cap) else "Market Cap: N/A")
    print(f"Enterprise Value: ${enterprise_value/1e9:.1f}B" if pd.notna(enterprise_value) else "EV: N/A")

# ==============================================================================
# 3. ANÁLISE DE RETORNOS E RISCO
# ==============================================================================

print("\n" + "="*70)
print("3. ANÁLISE DE RETORNOS E RISCO")
print("="*70)

if slb_prices is not None:
    returns = np.log(slb_prices['Close'] / slb_prices['Close'].shift(1)).dropna()
    
    # Retornos por período
    def calc_return(prices, days):
        if len(prices) >= days:
            return (prices.iloc[-1] / prices.iloc[-days] - 1) * 100
        return np.nan
    
    ret_1m = calc_return(prices, 21)
    ret_3m = calc_return(prices, 63)
    ret_6m = calc_return(prices, 126)
    ret_1y = calc_return(prices, 252)
    ret_3y = calc_return(prices, 756)
    ret_5y = calc_return(prices, 1260)
    
    print(f"\n--- Retornos Acumulados ---")
    print(f"1 mês: {ret_1m:.1f}%")
    print(f"3 meses: {ret_3m:.1f}%")
    print(f"6 meses: {ret_6m:.1f}%")
    print(f"1 ano: {ret_1y:.1f}%")
    print(f"3 anos: {ret_3y:.1f}%")
    print(f"5 anos: {ret_5y:.1f}%")
    
    # Volatilidade
    vol_annual = returns.std() * np.sqrt(252) * 100
    vol_3y = returns.last('756D').std() * np.sqrt(252) * 100
    
    print(f"\n--- Volatilidade Anualizada ---")
    print(f"10 anos: {vol_annual:.1f}%")
    print(f"3 anos: {vol_3y:.1f}%")
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    current_dd = drawdown.iloc[-1] * 100
    
    print(f"\n--- Drawdown ---")
    print(f"Max Drawdown (10y): {max_dd:.1f}%")
    print(f"Drawdown Atual: {current_dd:.1f}%")
    
    # VaR e CVaR
    var_95 = np.percentile(returns, 5) * 100
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    print(f"\n--- VaR/CVaR Diário (95%) ---")
    print(f"VaR 95%: {var_95:.2f}%")
    print(f"CVaR 95%: {cvar_95:.2f}%")

# ==============================================================================
# 4. CORRELAÇÃO E BETAS
# ==============================================================================

print("\n" + "="*70)
print("4. CORRELAÇÃO COM PETRÓLEO E MERCADO")
print("="*70)

# Preparar dados
all_data = pd.DataFrame()
if slb_prices is not None:
    all_data['SLB'] = slb_prices['Close']
if wti is not None:
    all_data['WTI'] = wti['Close']
if xle is not None:
    all_data['XLE'] = xle['Close']
if spy is not None:
    all_data['SPY'] = spy['Close']
if brent is not None:
    all_data['Brent'] = brent['Close']

all_data = all_data.dropna()
all_returns = np.log(all_data / all_data.shift(1)).dropna()

if len(all_returns) > 100:
    # Correlações
    corr = all_returns.corr()
    
    print(f"\n--- Correlações (retornos diários) ---")
    print(f"SLB vs WTI: {corr.loc['SLB', 'WTI']:.3f}")
    print(f"SLB vs Brent: {corr.loc['SLB', 'Brent']:.3f}")
    print(f"SLB vs XLE: {corr.loc['SLB', 'XLE']:.3f}")
    print(f"SLB vs SPY: {corr.loc['SLB', 'SPY']:.3f}")
    
    # Betas
    def calc_beta(stock, benchmark):
        X = sm.add_constant(all_returns[benchmark])
        model = sm.OLS(all_returns[stock], X).fit()
        return model.params[benchmark], model.rsquared
    
    beta_wti, r2_wti = calc_beta('SLB', 'WTI')
    beta_spy, r2_spy = calc_beta('SLB', 'SPY')
    beta_xle, r2_xle = calc_beta('SLB', 'XLE')
    
    print(f"\n--- Betas ---")
    print(f"Beta vs WTI: {beta_wti:.3f} (R²={r2_wti:.3f})")
    print(f"Beta vs SPY: {beta_spy:.3f} (R²={r2_spy:.3f})")
    print(f"Beta vs XLE: {beta_xle:.3f} (R²={r2_xle:.3f})")
    
    # Regressão multifatorial
    X = sm.add_constant(all_returns[['WTI', 'SPY']])
    model = sm.OLS(all_returns['SLB'], X).fit()
    
    print(f"\n--- Modelo Multifatorial: SLB ~ WTI + SPY ---")
    print(f"Alpha (anualizado): {model.params['const']*252:.4f}")
    print(f"Beta WTI: {model.params['WTI']:.4f} (p={model.pvalues['WTI']:.4f})")
    print(f"Beta SPY: {model.params['SPY']:.4f} (p={model.pvalues['SPY']:.4f})")
    print(f"R²: {model.rsquared:.4f}")

# ==============================================================================
# 5. ANÁLISE SLBG34 (BDR na B3)
# ==============================================================================

print("\n" + "="*70)
print("5. ANÁLISE SLBG34 (BDR na B3)")
print("="*70)

if slbg34_prices is not None and len(slbg34_prices) > 50:
    slbg34_close = slbg34_prices['Close']
    
    current_bdr = slbg34_close.iloc[-1]
    print(f"\nPreço atual SLBG34: R${current_bdr:.2f}")
    
    # Faixa de preço
    max_52w_bdr = slbg34_close.last('252D').max()
    min_52w_bdr = slbg34_close.last('252D').min()
    
    print(f"\n--- Faixa de Preço (52 semanas) ---")
    print(f"Mínimo: R${min_52w_bdr:.2f}")
    print(f"Máximo: R${max_52w_bdr:.2f}")
    print(f"Atual: R${current_bdr:.2f}")
    
    pct_range = (current_bdr - min_52w_bdr) / (max_52w_bdr - min_52w_bdr) * 100
    print(f"Posição na faixa: {pct_range:.1f}%")
    
    # Retornos
    ret_bdr = np.log(slbg34_close / slbg34_close.shift(1)).dropna()
    
    ret_1m_bdr = (slbg34_close.iloc[-1] / slbg34_close.iloc[-21] - 1) * 100 if len(slbg34_close) > 21 else np.nan
    ret_3m_bdr = (slbg34_close.iloc[-1] / slbg34_close.iloc[-63] - 1) * 100 if len(slbg34_close) > 63 else np.nan
    ret_6m_bdr = (slbg34_close.iloc[-1] / slbg34_close.iloc[-126] - 1) * 100 if len(slbg34_close) > 126 else np.nan
    ret_1y_bdr = (slbg34_close.iloc[-1] / slbg34_close.iloc[-252] - 1) * 100 if len(slbg34_close) > 252 else np.nan
    
    print(f"\n--- Retornos SLBG34 ---")
    print(f"1 mês: {ret_1m_bdr:.1f}%" if pd.notna(ret_1m_bdr) else "1 mês: N/A")
    print(f"3 meses: {ret_3m_bdr:.1f}%" if pd.notna(ret_3m_bdr) else "3 meses: N/A")
    print(f"6 meses: {ret_6m_bdr:.1f}%" if pd.notna(ret_6m_bdr) else "6 meses: N/A")
    print(f"1 ano: {ret_1y_bdr:.1f}%" if pd.notna(ret_1y_bdr) else "1 ano: N/A")
    
    # Comparação com SLB ajustado pelo dólar
    if usdbrl is not None and len(usdbrl) > 50:
        usdbrl_close = usdbrl['Close']
        current_usd = usdbrl_close.iloc[-1]
        print(f"\n--- Comparação com SLB + Câmbio ---")
        print(f"Dólar atual: R${current_usd:.2f}")
        
        # Preço teórico do BDR (SLB * USDBRL / fator de conversão)
        # BDRs geralmente têm paridade 1:1 ou próxima
        slb_em_reais = current_price * current_usd
        print(f"SLB em reais (teórico): R${slb_em_reais:.2f}")
        
        # Prêmio/desconto do BDR
        premio_desconto = (current_bdr / slb_em_reais - 1) * 100
        print(f"Prêmio/Desconto do BDR: {premio_desconto:.1f}%")
else:
    print("\n⚠️ Dados insuficientes para SLBG34. Usando análise apenas da SLB (NYSE).")

# ==============================================================================
# 6. CENÁRIOS E VALUATION
# ==============================================================================

print("\n" + "="*70)
print("6. CENÁRIOS DE PREÇO E VALUATION")
print("="*70)

print("""
=== TESE VENEZUELA + CAPEX ===

CONTEXTO:
1. Infraestrutura venezuelana degradada após 10+ anos sem manutenção
2. Petróleo do Orinoco Belt é extrapesado (8-16° API) - requer tecnologia especializada
3. Vendido com desconto de $15-25/barril vs Brent (heavy crude discount)
4. Produção atual: ~900k bpd vs potencial histórico de 3M+ bpd
5. Qualquer normalização/flexibilização = DEMANDA MASSIVA por oil services

POR QUE SLB É A PRINCIPAL BENEFICIÁRIA:
• Líder global em completação, estimulação e recuperação avançada
• Expertise específica em heavy oil e reservatórios complexos
• Presença histórica na Venezuela (operações antes das sanções)
• Maior escala para atender demanda reprimida
• Margem de alavancagem operacional: receita incremental = lucro incremental
""")

# Cenários de preço
print("\n--- CENÁRIOS DE PREÇO SLB ---")
print(f"\nPreço atual: ${current_price:.2f}")

if slb_info:
    ev_ebitda = slb_info.get('enterpriseToEbitda', 9.0)
    pe = slb_info.get('trailingPE', 15.0)
    
    print(f"\nMúltiplos atuais: EV/EBITDA={ev_ebitda:.1f}x, P/E={pe:.1f}x")
    
    # Cenário Base (status quo)
    print("\n[CENÁRIO BASE - Status Quo]")
    print("  Premissas: Petróleo lateral $70-80, CAPEX estável")
    print(f"  Preço-alvo: ${current_price * 1.10:.2f} (+10%)")
    print("  Prob: 50%")
    
    # Cenário Bull (CAPEX boom + Venezuela)
    print("\n[CENÁRIO BULL - CAPEX Boom + Venezuela]")
    print("  Premissas: Petróleo $90+, CAPEX +20%, abertura Venezuela parcial")
    print("  Expansão de múltiplo: EV/EBITDA para 11-12x")
    target_bull = current_price * (11.0 / ev_ebitda) * 1.15  # múltiplo + crescimento
    print(f"  Preço-alvo: ${target_bull:.2f} (+{(target_bull/current_price-1)*100:.0f}%)")
    print("  Prob: 25%")
    
    # Cenário Bear (recessão)
    print("\n[CENÁRIO BEAR - Recessão Global]")
    print("  Premissas: Petróleo $50-60, CAPEX cortado 15-20%")
    print("  Contração de múltiplo: EV/EBITDA para 6-7x")
    target_bear = current_price * (6.5 / ev_ebitda) * 0.90
    print(f"  Preço-alvo: ${target_bear:.2f} ({(target_bear/current_price-1)*100:.0f}%)")
    print("  Prob: 25%")
    
    # Valor esperado
    expected_value = 0.50 * current_price * 1.10 + 0.25 * target_bull + 0.25 * target_bear
    expected_return = (expected_value / current_price - 1) * 100
    
    print(f"\n>>> VALOR ESPERADO: ${expected_value:.2f} ({expected_return:+.1f}%)")

# ==============================================================================
# 7. ANÁLISE TÉCNICA SIMPLES
# ==============================================================================

print("\n" + "="*70)
print("7. ANÁLISE TÉCNICA SIMPLES")
print("="*70)

if slb_prices is not None:
    prices = slb_prices['Close']
    
    # Médias móveis
    sma_20 = prices.rolling(20).mean().iloc[-1]
    sma_50 = prices.rolling(50).mean().iloc[-1]
    sma_200 = prices.rolling(200).mean().iloc[-1]
    
    print(f"\n--- Médias Móveis ---")
    print(f"Preço: ${current_price:.2f}")
    print(f"SMA 20: ${sma_20:.2f} ({'acima' if current_price > sma_20 else 'abaixo'})")
    print(f"SMA 50: ${sma_50:.2f} ({'acima' if current_price > sma_50 else 'abaixo'})")
    print(f"SMA 200: ${sma_200:.2f} ({'acima' if current_price > sma_200 else 'abaixo'})")
    
    # Tendência
    if sma_20 > sma_50 > sma_200:
        trend = "ALTA (Golden Cross)"
    elif sma_20 < sma_50 < sma_200:
        trend = "BAIXA (Death Cross)"
    else:
        trend = "LATERAL/INDEFINIDA"
    
    print(f"\nTendência: {trend}")
    
    # RSI simplificado
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    print(f"\n--- RSI (14 dias) ---")
    print(f"RSI: {current_rsi:.1f}")
    if current_rsi > 70:
        print("Status: SOBRECOMPRADO")
    elif current_rsi < 30:
        print("Status: SOBREVENDIDO")
    else:
        print("Status: NEUTRO")
    
    # Distância das médias
    dist_sma200 = (current_price / sma_200 - 1) * 100
    print(f"\nDistância da SMA 200: {dist_sma200:+.1f}%")

# ==============================================================================
# 8. CONCLUSÃO
# ==============================================================================

print("\n" + "="*70)
print("8. CONCLUSÃO - VOCÊ PERDEU A ONDA?")
print("="*70)

print(f"""
=== ANÁLISE FINAL SLB / SLBG34 ===

PREÇO ATUAL: ${current_price:.2f} (SLB NYSE)
""")

# Determinar se está caro/barato
if slb_prices is not None:
    # Métricas de posição
    prices = slb_prices['Close']
    pct_10y = (current_price - prices.min()) / (prices.max() - prices.min()) * 100
    pct_5y = (current_price - prices.last('1260D').min()) / (prices.last('1260D').max() - prices.last('1260D').min()) * 100
    
    print(f"Posição na faixa de 10 anos: {pct_10y:.0f}%")
    print(f"Posição na faixa de 5 anos: {pct_5y:.0f}%")
    
    # Veredicto
    if pct_5y < 30:
        veredicto = "🟢 OPORTUNIDADE - Preço na parte baixa do range histórico"
    elif pct_5y < 50:
        veredicto = "🟡 NEUTRO/ENTRADA PARCIAL - Preço em região intermediária"
    elif pct_5y < 70:
        veredicto = "🟠 CUIDADO - Preço já andou bastante, esperar correção pode ser prudente"
    else:
        veredicto = "🔴 PERDEU A ONDA? - Preço na parte alta do range, risco/retorno menos favorável"
    
    print(f"\n>>> VEREDICTO: {veredicto}")

print("""
=== CONSIDERAÇÕES PARA DECISÃO ===

✅ ARGUMENTOS A FAVOR DE COMPRAR SLB/SLBG34:
• Tese de CAPEX em petróleo ainda intacta (ciclo de investimento iniciando)
• Venezuela é opcionalidade: qualquer abertura = upside significativo
• Infraestrutura degradada da Venezuela requer exatamente o que SLB oferece
• Petróleo extrapesado venezuelano precisa de tecnologia especializada
• SLB tem maior escala e expertise para capturar essa demanda
• Alavancagem operacional: cada $1 de receita adicional vai quase direto pro lucro
• Dividendos razoáveis enquanto espera

⚠️ ARGUMENTOS PARA ESPERAR/NÃO COMPRAR:
• Se o preço já subiu muito, risco/retorno piora
• Sanções Venezuela podem continuar por anos
• Recessão global cortaria CAPEX de petróleo
• Oil services são extremamente cíclicos
• Volatilidade alta (Max DD histórico de -80%+)

💡 ESTRATÉGIA SUGERIDA:
1. Se está na parte baixa do range (< 40%): Posição cheia
2. Se está no meio do range (40-60%): Entrada parcial, média em correções
3. Se está na parte alta do range (> 60%): Esperar correção de 10-15%
4. Usar SLB (NYSE) se tiver conta no exterior, ou SLBG34 pela conveniência

📊 PARA SLBG34 ESPECIFICAMENTE:
• Considerar efeito do câmbio (dólar forte = BDR mais caro)
• Liquidez menor que NYSE, spreads maiores
• Boa opção para quem quer exposição sem conta no exterior
""")

print("\n" + "="*70)
print("ANÁLISE CONCLUÍDA")
print("="*70)
