# HMM Regime Trading (AMZN)

<img width="3180" height="1781" alt="figure2_portfolio_value" src="https://github.com/user-attachments/assets/e8d765fc-ab1a-4010-8764-9b6938967d16" />

<img width="3179" height="1779" alt="figure1_trading_strategy" src="https://github.com/user-attachments/assets/6efeea8b-d4fb-4c04-859c-c46d769f9c22" />




A tiny **Hidden Markov Model (HMM)** script that detects 3 market regimes using:
- daily **returns** (`close.pct_change()`)
- daily **range** (`(high-low)/close`)

**Rule:**  
- **BUY** when the detected regime is the **Bull** state  
- **SELL** when the regime changes out of Bull (go to cash)  
- Optional **fees** via `fee = ...`

## Input
Place `AMZN_daily.csv` in the same folder with columns:
`date,open,high,low,close,volume`

## Run
```bash
python hmm_regime_strategy.py
```
Outputs

Saved to the figures/ folder:

figure1_trading_strategy.png — Price colored by regime + Buy/Sell markers

figure2_portfolio_value.png — Portfolio value over time
