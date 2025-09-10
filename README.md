
# PV+Speicher Arbitrage-Rechner (Streamlit)

Eine schlanke Streamlit-App zur **Bewertung von Arbitrage** mit Heimspeicher und **dynamischem Stromtarif** (Strom billig laden, teuer verkaufen – i. d. R. via Direktvermarktung).

## Funktionen
- **CSV/XLSX-Upload** stündlicher Preise (`timestamp`, `price_ct_per_kwh`).
- **Batterie-Parameter**: Kapazität, Lade-/Entladeleistung, Round-Trip-Wirkungsgrad, min. SOC.
- **Kosten**: variable Vermarktergebühr, sonstige variable Kosten, Degradationskosten, fixe Monatskosten (z. B. SMGW).
- **Strategie**: automatische Schwellen (Perzentile) oder manuelle Kauf-/Verkaufsschwellen.
- **Kennzahlen**: Gewinn im Zeitraum, Hochrechnung auf Monat/Jahr, Break-even-Spread.
- **Plots**: Preise, SOC-Verlauf, Lade-/Entladeenergie.
- **Export**: Ergebnisse als CSV.

## Wichtige Annahmen & rechtliche Hinweise
- **EEG-Vergütung** gibt es ausschließlich für **PV-Erzeugung**. *Aus Netz geladenen* Speicherstrom EEG-vergütet einzuspeisen ist **nicht zulässig**.
- **Arbitrage-Erlöse** setzen praktisch eine **Direktvermarktung** voraus (bilanzielle Abwicklung, Messkonzept, Verträge).
- Diese App modelliert **nur** die Preis-Arbitrage (Netz→Speicher→Netz). PV-Überschuss/Eigenverbrauch sind **nicht** Bestandteil der Gewinnrechnung (können aber separat ergänzt werden).

## Datenformat
CSV oder XLSX mit mindestens den Spalten:
- `timestamp` – ISO-Datum/Zeit (z. B. `2025-01-01 00:00:00`)
- `price_ct_per_kwh` – Preis in **ct/kWh** (negativ möglich)

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Parameter-Tipps
- **Degradationskosten** Heimspeicher: grob 6–12 ct/kWh Durchsatz (LFP, je nach CAPEX & Zyklen).
- **Automatische Schwellen**: Kauf 25. Perzentil, Verkauf 75. Perzentil – als Startpunkt.
- **Break-even-Spread** wird direkt angezeigt; wenn die reale Preis-Volatilität darunter liegt, ist Arbitrage meist **nicht** wirtschaftlich.

## Lizenz
MIT – ohne Gewähr. Nutzung auf eigenes Risiko.
