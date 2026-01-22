import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from hmmlearn import hmm

# -----------------------------
# Styling (same vibe as yours)
# -----------------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "axes.facecolor": "#000000",
    "figure.facecolor": "#000000",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333333",
    "legend.facecolor": "#111111",
    "legend.edgecolor": "#222222",
    "font.size": 10
})

# Create output folder
os.makedirs("figures", exist_ok=True)

# -----------------------------
# Load data (expects columns: date,open,high,low,close,volume)
# -----------------------------
df = pd.read_csv("AMZN_daily.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# -----------------------------
# Feature Engineering for HMM
# -----------------------------
df["returns"] = df["close"].pct_change()
df["range"] = (df["high"] - df["low"]) / df["close"]
df.dropna(inplace=True)
df = df.reset_index(drop=True)

# Prepare data for HMM
X = np.column_stack([df["returns"].values, df["range"].values])

# -----------------------------
# Train HMM (3 states)
# -----------------------------
model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
)

model.fit(X)

# Predict States
df["state"] = model.predict(X)

# -----------------------------
# Determine bull/bear states (keep same idea, but safer)
# - Bull: best risk-adjusted return (mean/std)
# - Bear: worst risk-adjusted return
# -----------------------------
state_stats = []
for i in range(model.n_components):
    state_mask = df["state"] == i
    mu = df.loc[state_mask, "returns"].mean()
    sig = df.loc[state_mask, "returns"].std()
    score = mu / (sig + 1e-9)  # avoid divide-by-zero
    state_stats.append({"state": i, "mean_return": mu, "volatility": sig, "score": score})

bull_state = max(state_stats, key=lambda d: d["score"])["state"]
bear_state = min(state_stats, key=lambda d: d["score"])["state"]

print("State Statistics:")
for s in state_stats:
    print(
        f"State {s['state']}: "
        f"Mean Return = {s['mean_return']*100:.4f}%, "
        f"Volatility = {s['volatility']*100:.4f}%, "
        f"Score = {s['score']:.4f}"
    )
print(f"\nBULL State (Buy): {bull_state}")
print(f"BEAR State (Sell/Cash): {bear_state}")

# -----------------------------
# Realism fix: trade on next day using yesterday's regime
# -----------------------------
df["trade_state"] = df["state"].shift(1)
df = df.dropna(subset=["trade_state"]).reset_index(drop=True)
df["trade_state"] = df["trade_state"].astype(int)

# Entry/exit signals (regime transitions)
df["Buy_Signal"] = ((df["trade_state"] == bull_state) &
                    (df["trade_state"].shift(1) != bull_state)).astype(int)
df["Sell_Signal"] = ((df["trade_state"] != bull_state) &
                     (df["trade_state"].shift(1) == bull_state)).astype(int)

# -----------------------------
# Portfolio Simulation
# - Stay in market only while in bull regime
# - Includes small trading fee (optional)
# -----------------------------
initial_cash = 1000.0
cash = initial_cash
in_trade = False
entry_price = None
entry_date = None
trades = []
portfolio_values = []

fee = 0.0000  # set e.g. 0.0005 for 5 bps per trade if you want

for i in range(len(df)):
    price = df["close"].iloc[i]
    date = df["date"].iloc[i]
    state = df["trade_state"].iloc[i]

    if not in_trade:
        if state == bull_state:
            entry_price = price
            entry_date = date
            in_trade = True
            cash *= (1 - fee)
            print(f"BUY at {entry_price:.2f} on {date.date()} | State: {state}")
    else:
        if state != bull_state:
            cash = cash * (price / entry_price) * (1 - fee)
            result = "win" if price > entry_price else "loss"
            print(f"SELL (REGIME CHANGE) at {price:.2f} on {date.date()} | State: {state} | Portfolio: {cash:.2f}")
            trades.append((entry_date, date, entry_price, price, result))
            in_trade = False
            entry_price = None
            entry_date = None

    # Track portfolio value (mark-to-market if holding)
    if in_trade and entry_price is not None:
        current_val = cash * (price / entry_price)
    else:
        current_val = cash
    portfolio_values.append(current_val)

df["Portfolio"] = portfolio_values

# -----------------------------
# Plotting: Regimes and Price
# -----------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
    figsize=(16, 9)
)

# Color map (Bull=Green, Bear=Red, Other=Orange)
color_map = {}
color_map[bull_state] = "#00FF88"  # Green
color_map[bear_state] = "#FF5555"  # Red
other_state = [s for s in [0, 1, 2] if s not in [bull_state, bear_state]][0]
color_map[other_state] = "#FFB347"  # Orange

# Plot price segments by regime (use trade_state so it matches trading logic)
for i in range(len(df) - 1):
    c = color_map[df["trade_state"].iloc[i]]
    ax1.plot(
        [df["date"].iloc[i], df["date"].iloc[i + 1]],
        [df["close"].iloc[i], df["close"].iloc[i + 1]],
        color=c, linewidth=1.5
    )

ax1.set_ylabel("Price ($)")
ax1.set_title("HMM Regime Switching Strategy (Green=Bull, Red=Bear, Orange=Neutral)",
              fontsize=14, color="white")
ax1.grid(True, linestyle="--", alpha=0.3)

# Entry/exit markers
ax1.scatter(df.loc[df["Buy_Signal"] == 1, "date"],
            df.loc[df["Buy_Signal"] == 1, "close"],
            marker="^", s=60, color="#00FF88", label="Buy")

ax1.scatter(df.loc[df["Sell_Signal"] == 1, "date"],
            df.loc[df["Sell_Signal"] == 1, "close"],
            marker="v", s=60, color="#FF5555", label="Sell")

ax1.legend(loc="upper left")

# Regime state subplot (step plot)
ax2.plot(df["date"], df["trade_state"], color="#FFFFFF", linewidth=0.8,
         drawstyle="steps-post", label="Market Regime")
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels([f"State {0}", f"State {1}", f"State {2}"])
ax2.set_ylabel("Regime")
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.autofmt_xdate()
plt.tight_layout()
fig.savefig("figures/figure1_trading_strategy.png", dpi=200, bbox_inches="tight")

# -----------------------------
# Plotting: Portfolio Value
# -----------------------------
fig2, ax4 = plt.subplots(figsize=(16, 9))
ax4.plot(df["date"], df["Portfolio"], linewidth=2.5, label="HMM Strategy Portfolio")
ax4.set_title("HMM Strategy Portfolio Performance", color="white", fontsize=16)
ax4.set_xlabel("Date", color="white")
ax4.set_ylabel("Portfolio Value ($)", color="white")
ax4.grid(True, linestyle="--", alpha=0.3)
ax4.legend(facecolor="#111111", edgecolor="#222222", fontsize=10, loc="upper left")
ax4.xaxis.set_major_locator(mdates.YearLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig2.autofmt_xdate()
plt.tight_layout()
fig2.savefig("figures/figure2_portfolio_value.png", dpi=200, bbox_inches="tight")

# -----------------------------
# Accuracy scoring (same idea)
# -----------------------------
if len(trades) > 0:
    total_trades = len(trades)
    wins = sum(1 for t in trades if t[4] == "win")
    accuracy = (wins / total_trades) * 100
    print(f"\nAccuracy: {accuracy:.2f}% trades were profitable")
    print(f"Final Portfolio Value: ${cash:.2f}")
else:
    print("\nNo trades were taken (bull regime may not have appeared with the shifted trading state).")
