from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dataclasses import dataclass
from pathlib import Path

SCENARIOS = ["Optimiste", "Neutre", "Pessimiste"]

@dataclass(frozen=True)
class StressTestConfig:
    alpha: float = 0.95
    cap_pd: float = 1.0

def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")

@st.cache_data
def load_inputs(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Portfolio and Scenario_Uplifts sheets."""
    portfolio = pd.read_excel(xlsx_path, sheet_name="Portfolio")
    uplifts = pd.read_excel(xlsx_path, sheet_name="Scenario_Uplifts")

    _require_columns(
        portfolio,
        ["loan_id", "sector", "country", "region", "EAD_EUR", "PD_base", "LGD", "maturity_years"],
        "Portfolio",
    )
    _require_columns(
        uplifts,
        [
            "sector",
            "pd_uplift_Optimiste", "pd_uplift_Neutre", "pd_uplift_Pessimiste",
            "lgd_uplift_Optimiste", "lgd_uplift_Neutre", "lgd_uplift_Pessimiste",
        ],
        "Scenario_Uplifts",
    )

    portfolio["sector"] = portfolio["sector"].astype(str)
    portfolio["country"] = portfolio["country"].astype(str)
    portfolio["region"] = portfolio["region"].astype(str)
    portfolio["EAD_EUR"] = pd.to_numeric(portfolio["EAD_EUR"], errors="raise")
    portfolio["PD_base"] = pd.to_numeric(portfolio["PD_base"], errors="raise")
    portfolio["LGD"] = pd.to_numeric(portfolio["LGD"], errors="raise")
    portfolio["maturity_years"] = pd.to_numeric(portfolio["maturity_years"], errors="raise")
    uplifts["sector"] = uplifts["sector"].astype(str)

    return portfolio, uplifts

def apply_scenario(
    portfolio: pd.DataFrame,
    uplifts: pd.DataFrame,
    scenario: str,
    cfg: StressTestConfig = StressTestConfig(),
) -> pd.DataFrame:
    """Return loan-level dataframe with stressed PD/LGD and projected losses."""
    pd_col = f"pd_uplift_{scenario}"
    lgd_col = f"lgd_uplift_{scenario}"

    u = uplifts[["sector", pd_col, lgd_col]].rename(
        columns={pd_col: "pd_uplift", lgd_col: "lgd_uplift"}
    )

    df = portfolio.merge(u, on="sector", how="left")
    
    df["PD_stress"] = np.clip(df["PD_base"] * (1.0 + df["pd_uplift"]), 0.0, cfg.cap_pd)
    df["LGD_stress"] = np.clip(df["LGD"] * (1.0 + df["lgd_uplift"]), 0.0, 1.0)
    df["dPD"] = np.maximum(df["PD_stress"] - df["PD_base"], 0.0)
    df["loss_projected"] = df["EAD_EUR"] * df["dPD"] * df["LGD_stress"]
    df["scenario"] = scenario

    cols = [
        "scenario", "loan_id", "sector", "country", "region",
        "EAD_EUR", "PD_base", "PD_stress", "dPD",
        "LGD", "LGD_stress", "maturity_years",
        "pd_uplift", "lgd_uplift", "loss_projected",
    ]
    return df[cols]

def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregate projected losses by a dimension (sector/country/region)."""
    g = df.groupby(by, dropna=False).agg(
        n_loans=("loan_id", "count"),
        EAD_EUR=("EAD_EUR", "sum"),
        loss_projected=("loss_projected", "sum"),
        avg_PD_base=("PD_base", "mean"),
        avg_PD_stress=("PD_stress", "mean"),
        avg_LGD=("LGD", "mean"),
        avg_LGD_stress=("LGD_stress", "mean"),
    ).reset_index()

    g["loss_rate_on_EAD"] = np.where(g["EAD_EUR"] > 0, g["loss_projected"] / g["EAD_EUR"], 0.0)
    return g.sort_values("loss_projected", ascending=False)

def climate_var(losses: list[float], alpha: float = 0.95) -> float:
    """Climate VaR: alpha-quantile of scenario losses."""
    arr = np.asarray(losses, dtype=float)
    return float(np.percentile(arr, 100.0 * alpha))

# --- DÉBUT DE L'INTERFACE STREAMLIT ---
def main():
    st.set_page_config(page_title="Climate Stress Test", layout="wide", page_icon="🌍")
    
    st.title("🌍 Dashboard: Climate Stress Test")
    st.markdown("Analyse des risques physiques et de transition (Inspiré ACPR/BCE)")

    # Menu latéral pour les paramètres interactifs
    st.sidebar.header("Paramètres")
    alpha_input = st.sidebar.slider("Niveau de confiance VaR (Alpha)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    cfg = StressTestConfig(alpha=alpha_input)

    # Chargement des données
    try:
        portfolio, uplifts = load_inputs("portfolio_climat.xlsx")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier Excel: {e}")
        return

    total_ead = float(portfolio["EAD_EUR"].sum())
    st.sidebar.metric("Exposition Totale (EAD)", f"{total_ead:,.0f} €")

    # Calcul des scénarios
    scenario_totals = []
    all_results = {}

    for sc in SCENARIOS:
        df_sc = apply_scenario(portfolio, uplifts, sc, cfg=cfg)
        total_loss = float(df_sc["loss_projected"].sum())
        scenario_totals.append({"Scénario": sc, "Pertes Projetées (€)": total_loss})
        all_results[sc] = df_sc

    # Affichage de la vue d'ensemble
    st.header("1. Vue d'ensemble des Scénarios")
    df_totals = pd.DataFrame(scenario_totals)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_totals, use_container_width=True)
        losses_only = [x["Pertes Projetées (€)"] for x in scenario_totals]
        var_alpha = climate_var(losses_only, alpha=cfg.alpha)
        st.info(f"**Climate VaR à {int(cfg.alpha*100)}% :** {var_alpha:,.0f} €")

    with col2:
        # Création d'un graphique interactif avec Plotly
        fig = px.bar(df_totals, x="Scénario", y="Pertes Projetées (€)", color="Scénario", title="Pertes estimées par Scénario climatique")
        st.plotly_chart(fig, use_container_width=True)

    # Affichage des détails filtrables
    st.header("2. Analyse Détaillée")
    selected_scenario = st.selectbox("Sélectionnez un scénario pour voir les détails par secteur", SCENARIOS)
    
    df_selected = all_results[selected_scenario]
    df_sector = summarize(df_selected, "sector")
    
    col3, col4 = st.columns(2)
    with col3:
        fig_pie = px.pie(df_sector, values="loss_projected", names="sector", title=f"Répartition des pertes par secteur ({selected_scenario})")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col4:
        st.write("Données sectorielles agrégées :")
        st.dataframe(df_sector[["sector", "loss_projected", "loss_rate_on_EAD"]], use_container_width=True)

    st.subheader("Extrait des données au niveau du prêt (Loan-level)")
    st.dataframe(df_selected.head(100), use_container_width=True)

if __name__ == "__main__":
    main()