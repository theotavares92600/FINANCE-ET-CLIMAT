from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="Climate Risk Intelligence", layout="wide", page_icon="🏛️")

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

@st.cache_data
def convert_df(df: pd.DataFrame):
    # Convertit le DataFrame en CSV formaté pour le web
    return df.to_csv(index=False).encode('utf-8')

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
    
    # NAVIGATION PAR MODULES (Ajout du 3ème onglet)
    tab_macro, tab_micro, tab_geo = st.tabs([
        "📊 Module 1: Executive Dashboard", 
        "🔍 Module 2: Portfolio Deep-Dive",
        "🌍 Module 3: Geographic Exposure"
    ])

    # --- MODULE 1: EXECUTIVE DASHBOARD ---
    with tab_macro:
        st.subheader("Analyse de Risque Agrégée")
        
        # 1. Metrics de haut niveau enrichies
        m1, m2, m3 = st.columns(3)
        total_ead = portfolio["EAD_EUR"].sum()
        max_loss = max([s["Pertes"] for s in scenario_totals])
        var_value = climate_var([s['Pertes'] for s in scenario_totals], cfg.alpha)
        
        m1.metric("Exposition Totale (EAD)", f"{total_ead/1e6:,.1f}M €")
        m2.metric(
            "Perte Maximale Projetée", 
            f"{max_loss:,.0f} €", 
            delta=f"{(max_loss/total_ead)*10000:.1f} bps de l'EAD", 
            delta_color="inverse"
        )
        m3.metric(f"Climate VaR ({int(cfg.alpha*100)}%)", f"{var_value:,.0f} €")

        st.markdown("---")

        # 2. Section Graphiques en 2 colonnes
        col1_macro, col2_macro = st.columns(2)

        with col1_macro:
            st.markdown("#### Pertes par Scénario Climatique")
            df_totals = pd.DataFrame(scenario_totals)
            fig_bar = px.bar(
                df_totals, x="Scénario", y="Pertes", color="Scénario",
                color_discrete_map={"Optimiste": "#2ecc71", "Neutre": "#f1c40f", "Pessimiste": "#e74c3c"}
            )
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2_macro:
            st.markdown("#### Composition du Portefeuille (EAD par Secteur)")
            df_ead_sector = portfolio.groupby("sector")["EAD_EUR"].sum().reset_index()
            fig_donut = px.pie(
                df_ead_sector, values="EAD_EUR", names="sector", hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_donut.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")
        
        # 3. Zone d'alerte : Top 3 des secteurs à risque (Basé sur le pire scénario)
        st.markdown("#### 🚨 Top 3 des Secteurs les plus vulnérables (Scénario Pessimiste)")
        
        df_pessimiste = all_results["Pessimiste"]
        df_top_sectors = summarize(df_pessimiste, "sector").head(3)
        top_sectors = df_top_sectors.to_dict('records')

        t1, t2, t3 = st.columns(3)
        
        if len(top_sectors) > 0:
            t1.error(f"**1. {top_sectors[0]['sector']}**\n\n**Perte:** {top_sectors[0]['loss_projected']:,.0f} €\n\n**Taux:** {top_sectors[0]['Loss_Rate_bps']:.1f} bps")
        if len(top_sectors) > 1:
            t2.warning(f"**2. {top_sectors[1]['sector']}**\n\n**Perte:** {top_sectors[1]['loss_projected']:,.0f} €\n\n**Taux:** {top_sectors[1]['Loss_Rate_bps']:.1f} bps")
        if len(top_sectors) > 2:
            t3.info(f"**3. {top_sectors[2]['sector']}**\n\n**Perte:** {top_sectors[2]['loss_projected']:,.0f} €\n\n**Taux:** {top_sectors[2]['Loss_Rate_bps']:.1f} bps")

    # --- MODULE 2: PORTFOLIO DEEP-DIVE ---
    with tab_micro:
        st.subheader("Analyse Granulaire & Concentration")
        
        col_ctrl, col_viz = st.columns([1, 3])
        
        with col_ctrl:
            target_sc = st.selectbox("Scénario cible", SCENARIOS, key="micro_sc")
            dim = st.radio("Grouper par", ["sector", "country", "region"])
            df_dim = summarize(all_results[target_sc], dim)

        with col_viz:
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
        
        # --- BOUTON D'EXPORT ---
        st.markdown("---")
        csv_data = convert_df(all_results[target_sc])
        st.download_button(
            label=f"📥 Télécharger les résultats complets ({target_sc}) en CSV",
            data=csv_data,
            file_name=f"stress_test_resultats_{target_sc.lower()}.csv",
            mime="text/csv",
        )

    # --- MODULE 3: GEOGRAPHIC EXPOSURE ---
 
    with tab_geo:
        st.subheader("Cartographie des Risques Européens")
        
        col_geo_ctrl, col_geo_viz = st.columns([1, 3])
        
        with col_geo_ctrl:
            geo_sc = st.selectbox("Scénario climatique", SCENARIOS, key="geo_sc")
            geo_metric = st.radio("Indicateur à cartographier", [
                "Pertes Projetées (€)", 
                "Taux de Perte (bps)", 
                "Exposition (EAD)"
            ])
            
        df_geo = summarize(all_results[geo_sc], "country")
        
        metric_col_map = {
            "Pertes Projetées (€)": "loss_projected",
            "Taux de Perte (bps)": "Loss_Rate_bps",
            "Exposition (EAD)": "EAD_EUR"
        }
        target_col = metric_col_map[geo_metric]
        color_scale = "Blues" if target_col == "EAD_EUR" else "Reds"

        with col_geo_viz:
            # Création de colonnes formatées proprement pour l'affichage au survol
            df_geo["EAD (Euro)"] = df_geo["EAD_EUR"].apply(lambda x: f"{x:,.0f} €")
            df_geo["Perte (Euro)"] = df_geo["loss_projected"].apply(lambda x: f"{x:,.0f} €")
            df_geo["Taux (bps)"] = df_geo["Loss_Rate_bps"].apply(lambda x: f"{x:.1f} bps")

            fig_map = px.choropleth(
                df_geo,
                locations="country", 
                locationmode="country names",
                color=target_col,
                hover_name="country",
                # Configuration de l'infobulle enrichie
                hover_data={
                    "country": False,
                    target_col: False,
                    "EAD (Euro)": True,
                    "Perte (Euro)": True,
                    "Taux (bps)": True
                },
                color_continuous_scale=color_scale,
                title=f"Distribution spatiale : {geo_metric} ({geo_sc})"
            )
            fig_map.update_geos(scope="europe", fitbounds="locations", showcountries=True, countrycolor="Black")
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            
            st.plotly_chart(fig_map, use_container_width=True)
            
        # Tableau avec dégradé de chaleur (Heatmap) sur la colonne des risques
        st.dataframe(
            df_geo[["country", "EAD_EUR", "loss_projected", "Loss_Rate_bps"]]
            .style.background_gradient(subset=['Loss_Rate_bps'], cmap='Reds')
            .format({
                "EAD_EUR": "{:,.0f} €",
                "loss_projected": "{:,.0f} €",
                "Loss_Rate_bps": "{:.1f}"
            }), 
            use_container_width=True
        )

if __name__ == "__main__":
    main()