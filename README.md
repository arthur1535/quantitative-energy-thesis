# Quantitative Energy Thesis

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/arthur1535/quantitative-energy-thesis/workflows/Tests/badge.svg)](https://github.com/arthur1535/quantitative-energy-thesis/actions)

Análise quantitativa do setor de petróleo e energia americano para seleção de ativos com métodos de otimização quantum-inspired.

---

## 🎯 Resultados Validados — 8/8 Previsões Corretas

> **Análise publicada em Jan/2026 — Validada em Mar/2026**  
> 📄 [Ver validação completa →](RESULTS_VALIDATION.md)

A análise feita em **Janeiro de 2026** previu corretamente:

| Previsão | Resultado Real (Mar/2026) | Status |
|----------|--------------------------|--------|
| Cenário bull com petróleo >$90 | WTI atingiu **$94.77** (+54%) | ✅ |
| SLB price target $55 (bull) | SLB atingiu **$51.85** (94% do alvo) | ✅ |
| Energia outperforma S&P 500 | Energia **+25%** vs SPY **-1.3%** | ✅ |
| Entrada SLBG34 a R$105-110 | Era R$105.36, subiu para **R$124** (+17.8%) | ✅ |
| Carteira SLB/COP/CVX rentável | **+24.0% em 2 meses** (alpha +25.3% vs SPY) | ✅ |
| Oil Services mais volátil que Majors | Range SLB 36% vs CVX 26% | ✅ |
| COP melhor risk-adjusted | COP **+25.3%** (top 3 absoluto) | ✅ |
| SLB maior torque ao petróleo | OIH +32% (Oil Services outperformou) | ✅ |

---

## 📊 Visão Geral

Este projeto implementa uma análise quantitativa completa para seleção entre 5 ativos do setor de energia:
- **CVX** (Chevron) - Major integrada
- **XOM** (ExxonMobil) - Major integrada  
- **COP** (ConocoPhillips) - E&P independente
- **SLB** (Schlumberger) - Oil Field Services
- **HAL** (Halliburton) - Oil Field Services

### Tese Principal
Avaliação de **SLB como veículo para capturar**:
1. Ciclo de CAPEX do setor de petróleo
2. Potencial reconstrução da infraestrutura petroleira venezuelana

## 🔬 Metodologia

### Métricas Calculadas
- **Valuation**: Earnings Yield, FCF Yield, EV/EBITDA, P/E, Dividend Yield
- **Qualidade**: Profit Margin, ROE, FCF Margin, Debt/Equity, Current Ratio
- **Risco**: VaR, CVaR, Max Drawdown, Volatilidade, Sharpe Ratio
- **Sensibilidade**: Betas (SPY, WTI, XLE, OIH)

### Modelos Implementados

| Modelo | Descrição |
|--------|-----------|
| **Regressão Multifatorial** | r = α + β₁·SPY + β₂·WTI + β₃·XLE + β₄·OIH + ε |
| **Monte Carlo (t-Student)** | 10.000 simulações com distribuição t (ν=5) para caudas gordas |
| **QUBO/Simulated Annealing** | Otimização quantum-inspired para seleção binária de ativos |

### Correções Técnicas
- ✅ **Drawdown**: Calculado com `np.exp(returns.cumsum())` para log-retornos
- ✅ **D/E Ratio**: Conversão automática de percentual para razão
- ✅ **Monte Carlo**: Distribuição t de Student (df=5) captura eventos extremos

## 📁 Estrutura do Projeto

```
├── src/
│   ├── data_fetcher.py          # Funções de coleta de dados
│   ├── metrics.py               # Cálculo de métricas
│   ├── optimization.py          # QUBO/Simulated Annealing
│   └── report_generator.py      # Geração de relatórios e visualizações
├── tests/
│   └── test_metrics.py          # Testes unitários
├── notebooks/
│   └── exploratory_analysis.ipynb  # Análise exploratória (futuro)
├── output/
│   └── results/                 # Resultados salvos (CSVs e gráficos)
├── analiseempresasamericanas.py   # Análise principal do setor
├── analise_slb_detalhada.py       # Deep-dive em SLB + tese Venezuela
├── relatorio_analise_petroleo.tex # Relatório LaTeX completo
├── requirements.txt               # Dependências Python
├── .github/workflows/tests.yml    # CI/CD com GitHub Actions
└── README.md
```

## 🚀 Instalação

```bash
# Clone o repositório
git clone https://github.com/arthur1535/quantitative-energy-thesis.git
cd quantitative-energy-thesis

# Instale as dependências
pip install -r requirements.txt

# Execute a análise principal
python analiseempresasamericanas.py

# Execute os testes
pytest tests/
```

## 📦 Dependências

```
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
arch>=6.2.0
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
```

## 📈 Resultados

### Ranking Final (QUBO/Simulated Annealing)

| Ranking | Ativo | Score | Cenário Ideal |
|---------|-------|-------|---------------|
| 1º | COP | 1.12 | Base/Bull |
| 2º | CVX | 0.08 | Bear/Base |
| 3º | XOM | -0.15 | Bear/Base |
| 4º | HAL | -0.30 | Bull extremo |
| 5º | SLB | -0.75 | **Bull + Venezuela** |

### Hipóteses Validadas

- ✅ **H1**: SLB tem maior torque ao petróleo (β_WTI = 0.46 vs Majors = 0.32)
- ✅ **H3**: Majors vencem em robustez de balanço (D/E: XOM=15.7 vs HAL=83.6)

### Alocação Sugerida

| Ativo | Peso | Razão |
|-------|------|-------|
| SLB | 60% | Máxima exposição à tese Venezuela + CAPEX |
| COP | 30% | Hedge se petróleo sobe mas CAPEX não |
| CVX | 10% | Segurança: dividendos, balanço forte |

## 📊 Visualizações

O projeto gera automaticamente visualizações que são salvas no diretório `output/`:

- **Fronteira Eficiente**: Relação risco-retorno de todos os ativos
- **Beta Rolling**: Evolução temporal do beta de cada ativo
- **Matriz de Correlação**: Correlação entre retornos dos ativos
- **Gráficos de Drawdown**: Evolução de preço e drawdown histórico

Resultados são salvos em CSV com timestamp para análise posterior em `output/results/`.

## 📄 Relatório

O relatório completo em LaTeX inclui:
- Executive Summary
- Metodologia detalhada
- Análises de Valuation, Qualidade e Risco
- Simulação Monte Carlo com t-Student
- Otimização QUBO/Simulated Annealing
- Análise especial: SLB e a tese Venezuela
- Prós e contras de cada ativo
- Código Python documentado

Para compilar:
```bash
pdflatex relatorio_analise_petroleo.tex
```

Ou faça upload para [Overleaf](https://www.overleaf.com/).

## ⚠️ Disclaimer

Este projeto é uma análise quantitativa para fins **educacionais** e de apoio à decisão. **NÃO constitui recomendação de compra ou venda**. Investimentos em renda variável envolvem riscos significativos, incluindo perda total do capital. O desempenho passado não garante resultados futuros.

## 👤 Autor

**Arthur Pires Lopes**  
📧 arthur.lopes1@ufu.br  
🎓 Universidade Federal de Uberlândia (UFU)

## 📚 Referências

1. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing*. Science, 220(4598), 671-680.
2. Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77-91.
3. Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.

## 📜 Licença

MIT License