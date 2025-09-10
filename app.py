
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="PV+Speicher Arbitrage-Rechner", layout="wide")

# ---------- Helper functions ----------
def default_price_series(days=7, seed=42):
    """
    Create a synthetic hourly price series in ct/kWh for demonstration.
    Pattern: daily sinus + random noise; can include negative prices.
    """
    rng = np.random.default_rng(seed)
    hours = days * 24
    t = np.arange(hours)
    base = 20 + 8*np.sin(2*np.pi*(t%24)/24 - 1.2)  # day-night swing
    noise = rng.normal(0, 2.5, hours)
    occasional_spike = (rng.random(hours) < 0.05) * rng.uniform(10, 30, hours)
    occasional_negative = (rng.random(hours) < 0.04) * (-rng.uniform(1, 8, hours))
    price_ct = base + noise + occasional_spike + occasional_negative
    price_ct = np.maximum(price_ct, -15)  # cap negatives
    dt_index = pd.date_range("2025-01-01", periods=hours, freq="H")
    return pd.DataFrame({"timestamp": dt_index, "price_ct_per_kwh": price_ct})

def load_price_series(upload):
    if upload is None:
        return default_price_series()
    try:
        df = pd.read_csv(upload)
    except Exception:
        upload.seek(0)
        df = pd.read_excel(upload)
    # Normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    # Expected key: timestamp, price_ct_per_kwh
    # Try to infer
    ts_col = None
    price_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if ts_col is None and ("time" in cl or "datum" in cl or "timestamp" in cl):
            ts_col = c
        if price_col is None and ("price" in cl or "ct" in cl or "€/kwh" in cl or "eur_kwh" in cl or "preis" in cl):
            price_col = c
    if ts_col is None or price_col is None:
        raise ValueError("Spalten nicht gefunden. Erwartet: 'timestamp' und 'price_ct_per_kwh' (beliebige Schreibweise).")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col]),
        "price_ct_per_kwh": pd.to_numeric(df[price_col], errors="coerce")
    }).dropna()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out

def simulate_arbitrage(prices_ct, params):
    """
    Battery arbitrage with hourly prices.
    - prices_ct: pd.Series of length N with ct/kWh
    - params: dict with battery/market params
    Returns: dict with results and time series
    """
    eta_rt = params["roundtrip_eff"]
    eta_chg = np.sqrt(eta_rt)
    eta_dis = np.sqrt(eta_rt)
    cap = params["battery_kwh"]
    p_chg = params["max_charge_kw"]
    p_dis = params["max_discharge_kw"]
    soc = params["initial_soc"]
    soc_min = params["soc_min"]
    soc_max = 1.0
    dt = 1.0  # hour
    # Variable costs in €/kWh (apply to discharged energy to market)
    vermarkter_fee_eur_per_kwh = params["vermarkter_fee_ct"]/100.0
    other_var_fee_eur_per_kwh = params["other_var_fee_ct"]/100.0
    deg_cost_eur_per_kwh_throughput = params["degradation_cost_ct"]/100.0
    # Fixed monthly costs (converted to per-hour over 30-day month for sim)
    fixed_monthly_eur = params["fixed_monthly_eur"]
    hours = len(prices_ct)
    fixed_cost_per_hour = fixed_monthly_eur / (30*24)

    # Strategy: threshold-based. Charge when price <= buy_threshold; discharge when price >= sell_threshold.
    # If user sets "auto", compute buy/sell thresholds from percentiles.
    if params["auto_thresholds"]:
        buy_threshold_ct = np.percentile(prices_ct, params["buy_percentile"])
        sell_threshold_ct = np.percentile(prices_ct, params["sell_percentile"])
    else:
        buy_threshold_ct = params["buy_threshold_ct"]
        sell_threshold_ct = params["sell_threshold_ct"]

    # Dynamic break-even check: required spread so that profit >= 0 for 1 kWh round-trip.
    # Profit per kWh charged:
    # revenue = price_sell_eur * eta_dis
    # cost_energy = price_buy_eur / eta_chg  <-- charging uses more than 1 kWh_in to get 1 kWh battery? We model per grid kWh charged.
    # We define per kWh charged from grid: energy_out = eta_rt * 1 kWh.
    # profit_per_kwh_in = price_sell_eur * eta_rt - price_buy_eur - (fees applied to kWh_out) - (degradation on throughput kWh_in)
    # fees apply to discharge energy (kWh_out): (vermarkter+other_var) * eta_rt
    # degradation applies to throughput at charge: deg_cost * 1
    # So required spread (sell-buy) in €/kWh ≈ ( (vermarkter+other_var)*eta_rt + deg_cost )
    # then divide by eta_rt to translate to buy->sell threshold on sell:buy when optimizing pair-wise. We will compute and show it.
    avg_price_eur = np.mean(prices_ct)/100.0
    req_spread_eur = (vermarkter_fee_eur_per_kwh + other_var_fee_eur_per_kwh)*eta_rt + deg_cost_eur_per_kwh_throughput
    breakeven_spread_ct = req_spread_eur*100.0/ max(eta_rt, 1e-6)

    # Time series accumulators
    soc_series = []
    charge_kwh_series = []
    discharge_kwh_series = []
    bought_eur_series = []
    sold_eur_series = []
    var_fee_eur_series = []
    deg_cost_eur_series = []
    fixed_cost_eur_series = []

    energy_throughput_kwh = 0.0
    total_profit_eur = 0.0

    for i, price_ct in enumerate(prices_ct):
        price_eur = price_ct/100.0
        # Fixed costs accrue continuously
        fixed_cost_eur_series.append(fixed_cost_per_hour)

        charge_kwh = 0.0
        discharge_kwh = 0.0
        bought_eur = 0.0
        sold_eur = 0.0
        var_fee_eur = 0.0
        deg_cost_eur = 0.0

        # Decide action
        if price_ct <= buy_threshold_ct and soc < soc_max - 1e-9:
            # Charge as much as possible limited by power and headroom
            headroom_kwh = (soc_max - soc) * cap
            charge_possible = min(p_chg*dt, headroom_kwh/eta_chg)  # account for charging inefficiency
            if charge_possible > 0:
                charge_kwh = charge_possible
                # Update SOC (energy stored = charge_kwh * eta_chg)
                stored = charge_kwh * eta_chg
                soc = min(soc + stored/cap, soc_max)
                bought_eur = charge_kwh * price_eur
                deg_cost_eur = charge_kwh * deg_cost_eur_per_kwh_throughput
                energy_throughput_kwh += charge_kwh

        elif price_ct >= sell_threshold_ct and soc > soc_min + 1e-9:
            # Discharge as much as possible limited by power and energy above soc_min
            available_kwh = (soc - soc_min) * cap
            discharge_possible_dc = min(p_dis*dt, available_kwh)  # energy at DC (battery)
            if discharge_possible_dc > 0:
                discharge_kwh = discharge_possible_dc * eta_dis  # energy delivered to meter/grid
                # Update SOC (energy removed from battery = discharge_possible_dc)
                soc = max(soc - discharge_possible_dc/cap, soc_min)
                sold_eur = discharge_kwh * price_eur
                var_fee_eur = discharge_kwh * (vermarkter_fee_eur_per_kwh + other_var_fee_eur_per_kwh)

        # Accumulate
        soc_series.append(soc)
        charge_kwh_series.append(charge_kwh)
        discharge_kwh_series.append(discharge_kwh)
        bought_eur_series.append(bought_eur)
        sold_eur_series.append(sold_eur)
        deg_cost_eur_series.append(deg_cost_eur)
        # Profit this hour
        total_profit_eur += sold_eur - bought_eur - var_fee_eur - deg_cost_eur - fixed_cost_per_hour

    res = {
        "soc": np.array(soc_series),
        "charge_kwh": np.array(charge_kwh_series),
        "discharge_kwh": np.array(discharge_kwh_series),
        "bought_eur": np.array(bought_eur_series),
        "sold_eur": np.array(sold_eur_series),
        "var_fee_eur": np.array(var_fee_eur_series),
        "deg_cost_eur": np.array(deg_cost_eur_series),
        "fixed_cost_eur": np.array(fixed_cost_eur_series),
        "total_profit_eur": total_profit_eur,
        "breakeven_spread_ct": breakeven_spread_ct,
        "buy_threshold_ct": buy_threshold_ct,
        "sell_threshold_ct": sell_threshold_ct,
    }
    return res

# ---------- Sidebar inputs ----------
st.sidebar.title("Parameter")
st.sidebar.caption("Alle Preise netzseitig (ohne USt) in ct/kWh, sofern nicht anders beschrieben.")

uploaded = st.sidebar.file_uploader("Strompreise als CSV/XLSX (Spalten: timestamp, price_ct_per_kwh). Leer lassen für Demo.", type=["csv","xlsx"])

with st.sidebar.expander("Batterie & Leistung", expanded=True):
    battery_kwh = st.number_input("Speicherkapazität [kWh]", 5.0, 1000.0, 10.0, 0.5)
    max_charge_kw = st.number_input("Max. Ladeleistung [kW]", 0.5, 1000.0, 5.0, 0.5)
    max_discharge_kw = st.number_input("Max. Entladeleistung [kW]", 0.5, 1000.0, 5.0, 0.5)
    roundtrip_eff = st.slider("Round-Trip-Wirkungsgrad [%]", 70, 98, 90)/100.0
    soc_min = st.slider("Min. SOC [%]", 0, 50, 10)/100.0
    initial_soc = st.slider("Start-SOC [%]", int(soc_min*100), 100, 50)/100.0

with st.sidebar.expander("Kosten & Gebühren", expanded=True):
    vermarkter_fee_ct = st.number_input("Direktvermarktung: variable Gebühr [ct/kWh_out]", 0.0, 10.0, 0.5, 0.1)
    other_var_fee_ct = st.number_input("Sonstige variable Kosten [ct/kWh_out]", 0.0, 20.0, 0.3, 0.1)
    degradation_cost_ct = st.number_input("Batterie-Verschleißkosten [ct/kWh_throughput]", 0.0, 30.0, 8.0, 0.5)
    fixed_monthly_eur = st.number_input("Fixkosten (SMGW, Vermarktung) [€/Monat]", 0.0, 200.0, 10.0, 1.0)

with st.sidebar.expander("Handelsstrategie (Schwellen)", expanded=True):
    auto = st.checkbox("Schwellen automatisch (Perzentile)", value=True)
    if auto:
        buy_percentile = st.slider("Kauf-Perzentil [%]", 1, 49, 25)
        sell_percentile = st.slider("Verkaufs-Perzentil [%]", 51, 99, 75)
        buy_threshold_ct = None
        sell_threshold_ct = None
    else:
        buy_threshold_ct = st.number_input("Kaufschwelle [ct/kWh]", -50.0, 200.0, 12.0, 0.5)
        sell_threshold_ct = st.number_input("Verkaufsschwelle [ct/kWh]", -50.0, 200.0, 28.0, 0.5)
        buy_percentile = sell_percentile = None

# ---------- Load data ----------
try:
    df = load_price_series(uploaded)
except Exception as e:
    st.error(f"Fehler beim Laden der Preisdatei: {e}")
    st.stop()

st.write("### Zeitreihe der Strompreise")
st.caption("Wenn keine Datei hochgeladen wurde, wird eine synthetische Demo-Zeitreihe verwendet.")
st.dataframe(df.head(24))

fig = plt.figure()
plt.plot(df["timestamp"], df["price_ct_per_kwh"])
plt.title("Preise [ct/kWh]")
plt.xlabel("Zeit")
plt.ylabel("ct/kWh")
st.pyplot(fig)

# ---------- Run simulation ----------
params = dict(
    battery_kwh=battery_kwh,
    max_charge_kw=max_charge_kw,
    max_discharge_kw=max_discharge_kw,
    roundtrip_eff=roundtrip_eff,
    soc_min=soc_min,
    initial_soc=initial_soc,
    vermarkter_fee_ct=vermarkter_fee_ct,
    other_var_fee_ct=other_var_fee_ct,
    degradation_cost_ct=degradation_cost_ct,
    fixed_monthly_eur=fixed_monthly_eur,
    auto_thresholds=auto,
    buy_percentile=buy_percentile if auto else None,
    sell_percentile=sell_percentile if auto else None,
    buy_threshold_ct=buy_threshold_ct if not auto else None,
    sell_threshold_ct=sell_threshold_ct if not auto else None,
)

res = simulate_arbitrage(df["price_ct_per_kwh"].values, params)

# ---------- KPIs ----------
hours = len(df)
profit_total = res["total_profit_eur"]
profit_per_day = profit_total / (hours/24)
profit_per_month = profit_per_day * 30
profit_per_year = profit_per_day * 365

col1, col2, col3, col4 = st.columns(4)
col1.metric("Gesamtgewinn im Zeitraum", f"{profit_total:,.2f} €")
col2.metric("Ø Gewinn / Tag", f"{profit_per_day:,.2f} €")
col3.metric("Hochrechnung / Monat", f"{profit_per_month:,.2f} €")
col4.metric("Hochrechnung / Jahr", f"{profit_per_year:,.2f} €")

st.write("### Break-even-Analyse (Daumenwert)")
st.write(f"- Erforderlicher **Preis-Spread** (ct/kWh), damit 1 kWh Round-Trip ≥ 0 €: **≈ {res['breakeven_spread_ct']:.1f} ct/kWh**")
st.write(f"- Eingestellte Schwellen: Kauf ≤ **{res['buy_threshold_ct']:.1f}** ct/kWh, Verkauf ≥ **{res['sell_threshold_ct']:.1f}** ct/kWh")

# ---------- Timeseries results ----------
out = df.copy()
out["soc"] = res["soc"]
out["charge_kwh"] = res["charge_kwh"]
out["discharge_kwh"] = res["discharge_kwh"]
out["bought_eur"] = res["bought_eur"]
out["sold_eur"] = res["sold_eur"]
out["var_fee_eur"] = res["var_fee_eur"]
out["deg_cost_eur"] = res["deg_cost_eur"]
out["fixed_cost_eur"] = res["fixed_cost_eur"]

st.write("### Zeitreihen – Lade-/Entladeleistung & SOC")
fig2 = plt.figure()
plt.plot(out["timestamp"], out["soc"]*100.0)
plt.title("State of Charge [%]")
plt.xlabel("Zeit")
plt.ylabel("SOC [%]")
st.pyplot(fig2)

fig3 = plt.figure()
plt.plot(out["timestamp"], out["charge_kwh"], label="Ladung [kWh]")
plt.plot(out["timestamp"], out["discharge_kwh"], label="Entladung [kWh]")
plt.title("Lade-/Entladeenergie pro Stunde [kWh]")
plt.xlabel("Zeit")
plt.ylabel("kWh")
plt.legend()
st.pyplot(fig3)

st.download_button(
    label="Ergebnisse als CSV herunterladen",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="arbitrage_ergebnisse.csv",
    mime="text/csv",
)

st.write("---")
with st.expander("Hinweise & Annahmen"):
    st.markdown("""
- Dieses Tool modelliert **reine Strompreis-Arbitrage** mit einem Heimspeicher und setzt eine rechtliche/vertragliche **Direktvermarktung** für Rückspeisung voraus.
- **EEG-Vergütung** für aus dem Netz geladenen Strom ist nicht zulässig; dafür ist hier **0 ct/kWh** angesetzt. Für PV-Überschuss lässt sich optional eine getrennte Betrachtung ergänzen.
- Variable Gebühren werden pro abgegebene kWh (kWh_out) erhoben; Degradation pro durchgesetzter kWh (kWh_in).
- Fixkosten (z. B. SMGW/Messstellenbetrieb, Direktvermarkter Grundgebühr) werden linear auf Stunden verteilt.
- Strategie: einfache Schwellenlogik. Für professionelle Optimierung wären **Prognosen** & **Optimierer** (MILP) sinnvoll.
- Preise verstehen sich **netto** (ohne Umsatzsteuer). Bitte bei Bedarf Brutto anpassen.
""")
