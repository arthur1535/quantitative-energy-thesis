# Quantitative Energy Thesis

Análise quantitativa do setor de petróleo e energia americano para seleção de ativos com métodos de otimização quantum-inspired.

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
├── analiseempresasamericanas.py   # Análise principal do setor
├── analise_slb_detalhada.py       # Deep-dive em SLB + tese Venezuela
├── relatorio_analise_petroleo.tex # Relatório LaTeX completo
├── requirements.txt               # Dependências Python
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
```

## 📦 Dependências

```
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
statsmodels>=0.14.0
matplotlib>=3.7.0
arch>=6.2.0
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