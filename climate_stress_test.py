from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="Climate Risk Intelligence", layout="wide", page_icon="🏛️")

# CSS pour un look "Finance" épuré
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stHeader"] { background: rgba(255,255,255,0); }
    </style>
    """, unsafe_allow_html=True)

SCENARIOS = ["Optimiste", "Neutre", "Pessimiste"]

@dataclass(frozen=True)
class StressTestConfig:
    alpha: float = 0.95
    cap_pd: float = 1.0

# --- FONCTIONS DE CALCUL (Back-end) ---
def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")

@st.cache_data
def load_inputs(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio = pd.read_excel(xlsx_path, sheet_name="Portfolio")
    uplifts = pd.read_excel(xlsx_path, sheet_name="Scenario_Uplifts")
    
    _require_columns(portfolio, ["loan_id", "sector", "country", "region", "EAD_EUR", "PD_base", "LGD", "maturity_years"], "Portfolio")
    _require_columns(uplifts, ["sector", "pd_uplift_Optimiste", "pd_uplift_Neutre", "pd_uplift_Pessimiste", "lgd_uplift_Optimiste", "lgd_uplift_Neutre", "lgd_uplift_Pessimiste"], "Scenario_Uplifts")
    
    for col in ["sector", "country", "region"]: portfolio[col] = portfolio[col].astype(str)
    return portfolio, uplifts

def apply_scenario(portfolio: pd.DataFrame, uplifts: pd.DataFrame, scenario: str, cfg: StressTestConfig) -> pd.DataFrame:
    pd_col, lgd_col = f"pd_uplift_{scenario}", f"lgd_uplift_{scenario}"
    u = uplifts[["sector", pd_col, lgd_col]].rename(columns={pd_col: "pd_uplift", lgd_col: "lgd_uplift"})
    
    df = portfolio.merge(u, on="sector", how="left")
    df["PD_stress"] = np.clip(df["PD_base"] * (1.0 + df["pd_uplift"]), 0.0, cfg.cap_pd)
    df["LGD_stress"] = np.clip(df["LGD"] * (1.0 + df["lgd_uplift"]), 0.0, 1.0)
    df["loss_projected"] = df["EAD_EUR"] * (df["PD_stress"] - df["PD_base"]) * df["LGD_stress"]
    return df

def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by).agg(
        EAD_EUR=("EAD_EUR", "sum"),
        loss_projected=("loss_projected", "sum")
    ).reset_index()
    g["Loss_Rate_bps"] = (g["loss_projected"] / g["EAD_EUR"]) * 10000
    return g.sort_values("loss_projected", ascending=False)

def climate_var(losses: list[float], alpha: float) -> float:
    return float(np.percentile(np.asarray(losses), 100.0 * alpha))

# --- INTERFACE UTILISATEUR (Front-end) ---
def main():
    st.title("🏛️ Climate Risk Intelligence Platform")
    st.caption("Analyse prudentielle des risques de transition et physiques — Framework ACPR/BCE")
    st.markdown("---")

    # SIDEBAR
    st.sidebar.header("🕹️ Contrôles Analytiques")
    alpha_input = st.sidebar.select_slider("Niveau de confiance VaR", options=[0.90, 0.95, 0.99], value=0.95)
    cfg = StressTestConfig(alpha=alpha_input)

    try:
        portfolio, uplifts = load_inputs("portfolio_climat.xlsx")
    except Exception as e:
        st.error(f"Fichier source introuvable : {e}")
        return

    # Calculs globaux
    all_results = {sc: apply_scenario(portfolio, uplifts, sc, cfg) for sc in SCENARIOS}
    scenario_totals = [{"Scénario": sc, "Pertes": all_results[sc]["loss_projected"].sum()} for sc in SCENARIOS]
    
    # NAVIGATION PAR MODULES
    tab_macro, tab_micro = st.tabs(["📊 Module 1: Executive Dashboard", "🔍 Module 2: Portfolio Deep-Dive"])

    # --- MODULE 1: EXECUTIVE DASHBOARD ---
    with tab_macro:
        st.subheader("Analyse de Risque Agrégée")
        
        # Metrics de haut niveau
        m1, m2, m3 = st.columns(3)
        total_ead = portfolio["EAD_EUR"].sum()
        max_loss = max([s["Pertes"] for s in scenario_totals])
        m1.metric("Exposition Totale (EAD)", f"{total_ead/1e6:,.1f}M €")
        m2.metric("Perte Maximale Projetée", f"{max_loss:,.0f} €", delta="Stress Test", delta_color="inverse")
        m3.metric("Climate VaR", f"{climate_var([s['Pertes'] for s in scenario_totals], cfg.alpha):,.0f} €")

        st.markdown("#### Comparaison des Pertes par Scénario")
        df_totals = pd.DataFrame(scenario_totals)
        
        fig_bar = px.bar(df_totals, x="Scénario", y="Pertes", color="Scénario",
                         color_discrete_map={"Optimiste": "#2ecc71", "Neutre": "#f1c40f", "Pessimiste": "#e74c3c"})
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- MODULE 2: PORTFOLIO DEEP-DIVE ---
    with tab_micro:
        st.subheader("Analyse Granulaire & Concentration")
        
        col_ctrl, col_viz = st.columns([1, 3])
        
        with col_ctrl:
            target_sc = st.selectbox("Scénario cible", SCENARIOS)
            dim = st.radio("Grouper par", ["sector", "country", "region"])
            df_dim = summarize(all_results[target_sc], dim)

        with col_viz:
            # Treemap pour la concentration
            fig_tree = px.treemap(df_dim, path=[dim], values='loss_projected',
                                 color='Loss_Rate_bps', color_continuous_scale='RdYlGn_r',
                                 title=f"Concentration des pertes par {dim} (bps)")
            st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown("#### Détail des positions (Top 50)")
        st.dataframe(
            all_results[target_sc].sort_values("loss_projected", ascending=False).head(50).style.format({
                "EAD_EUR": "{:,.0f} €",
                "PD_base": "{:.2%}",
                "PD_stress": "{:.2%}",
                "loss_projected": "{:,.2f} €"
            }), use_container_width=True
        )

if __name__ == "__main__":
    main()