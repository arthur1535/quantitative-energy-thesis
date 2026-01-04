# Guia de Uso da Estrutura Modular

## Visão Geral

A estrutura modular facilita a reutilização de código, manutenção e testes. O projeto está organizado da seguinte forma:

## Módulos Principais

### 1. `src/data_fetcher.py`
Responsável pela coleta de dados do yfinance.

**Principais funções:**
- `fetch_price_data(tickers, start, end)` - Baixa dados de preços históricos
- `fetch_fundamental_data(tickers)` - Extrai dados fundamentalistas

### 2. `src/metrics.py`
Calcula todas as métricas de valuation, qualidade e risco.

**Principais funções:**
- `calculate_returns(prices)` - Calcula retornos logarítmicos
- `calculate_valuation_metrics(fund_df)` - Métricas de valuation (P/E, FCF Yield, etc.)
- `calculate_quality_metrics(fund_df)` - Métricas de qualidade (ROE, margens, etc.)
- `calculate_risk_metrics(returns)` - Métricas de risco (Sharpe, VaR, Drawdown, etc.)
- `calculate_betas(returns, benchmark)` - Calcula betas
- `multifactor_regression(returns, stock, factors)` - Regressão multifatorial

### 3. `src/optimization.py`
Implementa simulações e otimização de portfólio.

**Principais funções:**
- `monte_carlo_simulation(returns, n_simulations, horizon_days)` - Simulação Monte Carlo
- `create_score_matrix(...)` - Cria matriz de scores combinados
- `simulated_annealing_selection(scores_df, n_select)` - Seleção via Simulated Annealing
- `scipy_optimization(scores_df, max_weight)` - Otimização contínua

### 4. `src/report_generator.py`
Gera visualizações e salva resultados.

**Principais funções:**
- `save_results(...)` - Salva DataFrames em CSV com timestamp
- `plot_efficient_frontier(returns, risk)` - Gráfico de fronteira eficiente
- `plot_rolling_beta(returns, stock, benchmark)` - Beta rolling
- `plot_correlation_matrix(returns)` - Matriz de correlação
- `plot_drawdown_chart(returns, stock)` - Gráfico de drawdown
- `generate_report(...)` - Gera relatório completo no console

## Como Usar

### Exemplo Básico

```python
from src.data_fetcher import fetch_price_data
from src.metrics import calculate_returns, calculate_risk_metrics
from src.report_generator import save_results

# 1. Baixar dados
tickers = ['CVX', 'XOM', 'SLB']
prices = fetch_price_data(tickers, start_date, end_date)

# 2. Calcular retornos
returns = calculate_returns(prices)

# 3. Calcular métricas de risco
risk = calculate_risk_metrics(returns)

# 4. Salvar resultados
save_results(val_metrics, qual_metrics, risk, scores)
```

### Script de Demonstração

Execute o script `example_modular_analysis.py` para ver um exemplo completo:

```bash
python example_modular_analysis.py
```

Este script:
1. Baixa dados de preços e fundamentalistas
2. Calcula todas as métricas
3. Executa Monte Carlo
4. Otimiza portfólio com Simulated Annealing
5. Gera visualizações
6. Salva resultados em CSV

### Análise Exploratória Interativa

Use o notebook Jupyter para análise interativa:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Testes

Execute os testes unitários:

```bash
pytest tests/ -v
```

Para verificar cobertura:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Configuração

O arquivo `config.yaml` centraliza todos os parâmetros:
- Tickers de ações, ETFs e benchmarks
- Período de análise
- Pesos para scoring final
- Parâmetros de Monte Carlo
- Parâmetros de Simulated Annealing

## Saída de Resultados

Todos os resultados são salvos em:
- `output/results/` - CSVs com timestamp (valuation, quality, risk, scores)
- `output/*.png` - Gráficos (fronteira eficiente, beta rolling, correlação, drawdown)

## Scripts Originais

Os scripts originais (`analiseempresasamericanas.py` e `analise_slb_detalhada.py`) continuam funcionando normalmente. A estrutura modular é adicional e pode ser usada conforme necessário.

## Próximos Passos

Para integrar completamente a estrutura modular:
1. Refatorar scripts originais para usar os módulos
2. Adicionar mais testes unitários
3. Criar dashboard interativo (Streamlit/Dash)
4. Implementar backtesting de estratégias
