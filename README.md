Machine-learning time-series forecasting pipeline for global natural gas and LNG prices with Power BI integration.
# Energy Price Forecasting Engine

This project implements a machine-learning time-series forecasting pipeline for global natural gas and LNG prices, integrated into Power BI for downstream analytics and reporting.

It extends prior descriptive analysis of LNG market dynamics into a production-style forecasting system designed for repeatability, scalability, and interpretability.

---

## Project Overview

Energy and LNG markets are highly dynamic, with commercial decisions driven by forward-looking price expectations rather than historical trends alone.  
This project addresses that need by generating region-specific price forecasts using a Python-based machine-learning pipeline and surfacing results in an interactive Power BI dashboard.

Forecasts are generated independently for each market based on the latest available data, reflecting real-world differences in reporting cadence and data latency.

---

## Forecasting Pipeline

The forecasting pipeline is built in Python and designed to mirror a lightweight production workflow:

- Monthly time-series modeling using a Random Forest regressor
- Feature engineering with lagged values and rolling statistics
- Per-series model training and 12-month forward forecasting
- Automated generation of forecast outputs with model metadata
- Versioned results written to structured files consumed by Power BI

Each pipeline run produces:
- Forecast values by region and month
- Model name and forecast horizon
- Run timestamps for versioning and reproducibility

---

## Markets Covered

- Henry Hub (U.S. Natural Gas)
- European Natural Gas
- Asia LNG (JKM proxy)

Forecast horizons begin immediately after the last observed actual for each market, reflecting real-world data availability differences across regions.

---

## Power BI Integration

Forecast outputs are integrated directly into Power BI, where they are combined with historical actuals to enable:

- Actual vs. forecast visual comparisons
- Clear delineation of forecast start points
- Cross-market analysis despite differing data recency
- Transparent interpretation through embedded methodology notes

---

## Design Considerations

- **Data Latency:** Markets transition to forecasted values at different times based on source availability.
- **Forecast Horizon:** Forecasts extend 12 months from each series’ final observed actual.
- **Model Transparency:** Metadata is retained to support filtering, comparison, and future model iteration.

---

## Relationship to Prior Work

This project builds on an earlier LNG market analytics dashboard focused on descriptive insights and arbitrage considerations.  
That project is available here: https://github.com/sammaddux64/US-LNG-Arbitrage-Export-Economics-Analysis

---

## Future Enhancements

Potential extensions include:
- Prediction intervals and uncertainty bands
- Scenario analysis and stress testing
- Incorporation of exogenous variables (storage, weather, policy indicators)
- Model comparison and ensemble approaches

---

## Disclaimer

Forecasts are model-based estimates derived from historical patterns and do not account for unforeseen geopolitical events, policy changes, or extreme supply disruptions.
