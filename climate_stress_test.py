from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION AND ADVANCED STYLE ---
st.set_page_config(page_title="Climate Risk Intelligence", layout="wide", page_icon="🏛️")



SCENARIOS = ["Optimistic", "Neutral", "Pessimistic", "Custom"]

@dataclass(frozen=True)
class StressTestConfig:
    alpha: float = 0.95
    cap_pd: float = 1.0

# --- CALCULATION FUNCTIONS (Back-end) ---
def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")

@st.cache_data
def load_inputs(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio = pd.read_excel(xlsx_path, sheet_name="Portfolio")
    uplifts = pd.read_excel(xlsx_path, sheet_name="Scenario_Uplifts")
    
    _require_columns(portfolio, ["loan_id", "sector", "country", "region", "EAD_EUR", "PD_base", "LGD", "maturity_years"], "Portfolio")
    # The code still looks for the French column names in the Excel file to avoid errors
    _require_columns(uplifts, [
        "sector",
        "pd_uplift_Optimiste", "pd_uplift_Neutre", "pd_uplift_Pessimiste", 
        "lgd_uplift_Optimiste", "lgd_uplift_Neutre", "lgd_uplift_Pessimiste"
    ], "Scenario_Uplifts")
    
    for col in ["sector", "country", "region"]: portfolio[col] = portfolio[col].astype(str)
    return portfolio, uplifts

def apply_scenario(portfolio: pd.DataFrame, uplifts: pd.DataFrame, scenario: str, cfg: StressTestConfig, custom_pd: float = 0.0, custom_lgd: float = 0.0) -> pd.DataFrame:
    df = portfolio.copy()
    
    if scenario == "Custom":
        df["pd_uplift"] = custom_pd
        df["lgd_uplift"] = custom_lgd
    else:
        # Invisible translation dictionary: English UI -> French Excel columns
        mapping = {"Optimistic": "Optimiste", "Neutral": "Neutre", "Pessimistic": "Pessimiste"}
        excel_suffix = mapping[scenario]
        
        pd_col, lgd_col = f"pd_uplift_{excel_suffix}", f"lgd_uplift_{excel_suffix}"
        u = uplifts[["sector", pd_col, lgd_col]].rename(columns={pd_col: "pd_uplift", lgd_col: "lgd_uplift"})
        df = df.merge(u, on="sector", how="left")
        
    df["PD_stress"] = np.clip(df["PD_base"] * (1.0 + df["pd_uplift"]), 0.0, cfg.cap_pd)
    df["LGD_stress"] = np.clip(df["LGD"] * (1.0 + df["lgd_uplift"]), 0.0, 1.0)
    df["loss_projected"] = df["EAD_EUR"] * (df["PD_stress"] - df["PD_base"]) * df["LGD_stress"]
    df["scenario"] = scenario
    return df

def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by).agg(
        EAD_EUR=("EAD_EUR", "sum"),
        loss_projected=("loss_projected", "sum")
    ).reset_index()
    g["Loss_Rate_bps"] = np.where(g["EAD_EUR"] > 0, (g["loss_projected"] / g["EAD_EUR"]) * 10000, 0)
    return g.sort_values("loss_projected", ascending=False)

def climate_var(losses: list[float], alpha: float) -> float:
    return float(np.percentile(np.asarray(losses), 100.0 * alpha))

@st.cache_data
def convert_df(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

# --- USER INTERFACE (Front-end) ---
def main():
    st.title("🏛️ Climate Risk Intelligence Platform")
    st.caption("Prudential analysis of transition and physical risks — ACPR/ECB Framework")
    st.markdown("---")

    # ENRICHED SIDEBAR
    st.sidebar.header("🕹️ Analytical Controls")
    alpha_input = st.sidebar.select_slider("VaR Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
    cfg = StressTestConfig(alpha=alpha_input)

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Custom Scenario")
    st.sidebar.info("Adjust these sliders to simulate a custom macroeconomic shock.")
    custom_pd_shock = st.sidebar.slider("Global PD Shock (%)", min_value=-20, max_value=200, value=20, step=5) / 100.0
    custom_lgd_shock = st.sidebar.slider("Global LGD Shock (%)", min_value=-10, max_value=100, value=10, step=5) / 100.0

    try:
        portfolio, uplifts = load_inputs("portfolio_climat.xlsx")
    except Exception as e:
        st.error(f"Source file not found: {e}")
        return

    all_results = {}
    for sc in SCENARIOS:
        all_results[sc] = apply_scenario(portfolio, uplifts, sc, cfg, custom_pd_shock, custom_lgd_shock)
        
    scenario_totals = [{"Scenario": sc, "Losses": all_results[sc]["loss_projected"].sum()} for sc in SCENARIOS]
    
    # 4 TABS DECLARATION
    
    tab_macro, tab_micro, tab_geo, tab_meth, tab_sim = st.tabs([
    "📊 Module 1: Executive Dashboard", 
    "🔍 Module 2: Portfolio Deep-Dive",
    "🌍 Module 3: Geographic Exposure",
    "📖 Module 4: Methodology & Framework",
    "📈 Module 5: Client Simulation"
])

    # --- MODULE 1: EXECUTIVE DASHBOARD ---
    with tab_macro:
        st.subheader("Aggregated Risk Analysis")
        m1, m2, m3 = st.columns(3)
        total_ead = portfolio["EAD_EUR"].sum()
        max_loss = max([s["Losses"] for s in scenario_totals])
        var_value = climate_var([s['Losses'] for s in scenario_totals], cfg.alpha)
        
        m1.metric("Total Exposure (EAD)", f"{total_ead/1e6:,.1f}M €")
        m2.metric("Maximum Projected Loss", f"{max_loss:,.0f} €", delta=f"{(max_loss/total_ead)*10000:.1f} bps of EAD", delta_color="inverse")
        m3.metric(f"Climate VaR ({int(cfg.alpha*100)}%)", f"{var_value:,.0f} €")

        st.markdown("---")
        col1_macro, col2_macro = st.columns(2)

        with col1_macro:
            st.markdown("#### Losses by Climate Scenario")
            df_totals = pd.DataFrame(scenario_totals)
            fig_bar = px.bar(
                df_totals, x="Scenario", y="Losses", color="Scenario",
                color_discrete_map={"Optimistic": "#2ecc71", "Neutral": "#f1c40f", "Pessimistic": "#e74c3c", "Custom": "#3498db"}
            )
            fig_bar.update_layout(
                font_family="Inter", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, title=""), yaxis=dict(gridcolor='#e2e8f0', title=""),
                showlegend=False, margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2_macro:
            st.markdown("#### Portfolio Composition (EAD by Sector)")
            df_ead_sector = portfolio.groupby("sector")["EAD_EUR"].sum().reset_index()
            fig_donut = px.pie(df_ead_sector, values="EAD_EUR", names="sector", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_donut.update_layout(font_family="Inter", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🚨 Top 3 Most Vulnerable Sectors (Pessimistic Scenario)")
        df_pessimistic = all_results["Pessimistic"]
        top_sectors = summarize(df_pessimistic, "sector").head(3).to_dict('records')

        t1, t2, t3 = st.columns(3)
        if len(top_sectors) > 0: t1.error(f"**1. {top_sectors[0]['sector']}**\n\n**Loss:** {top_sectors[0]['loss_projected']:,.0f} €\n\n**Rate:** {top_sectors[0]['Loss_Rate_bps']:.1f} bps")
        if len(top_sectors) > 1: t2.warning(f"**2. {top_sectors[1]['sector']}**\n\n**Loss:** {top_sectors[1]['loss_projected']:,.0f} €\n\n**Rate:** {top_sectors[1]['Loss_Rate_bps']:.1f} bps")
        if len(top_sectors) > 2: t3.info(f"**3. {top_sectors[2]['sector']}**\n\n**Loss:** {top_sectors[2]['loss_projected']:,.0f} €\n\n**Rate:** {top_sectors[2]['Loss_Rate_bps']:.1f} bps")

    # --- MODULE 2: PORTFOLIO DEEP-DIVE ---
    with tab_micro:
        st.subheader("Granular Analysis & Concentration")
        col_ctrl, col_viz = st.columns([1, 3])
        
        with col_ctrl:
            target_sc = st.selectbox("Target Scenario", SCENARIOS, key="micro_sc")
            dim = st.radio("Group by", ["sector", "country", "region"])
            df_dim = summarize(all_results[target_sc], dim)

        with col_viz:
            fig_tree = px.treemap(df_dim, path=[dim], values='loss_projected', color='Loss_Rate_bps', color_continuous_scale='RdYlGn_r', title=f"Loss concentration by {dim} (bps)")
            fig_tree.update_layout(font_family="Inter", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown("#### Position Details (Top 50)")
        st.dataframe(
            all_results[target_sc].sort_values("loss_projected", ascending=False).head(50).style.format({
                "EAD_EUR": "{:,.0f} €", "PD_base": "{:.2%}", "PD_stress": "{:.2%}", "loss_projected": "{:,.2f} €"
            }), use_container_width=True
        )

        st.markdown("---")
        csv_data = convert_df(all_results[target_sc])
        st.download_button(label=f"📥 Download full results ({target_sc}) as CSV", data=csv_data, file_name=f"stress_test_results_{target_sc.lower()}.csv", mime="text/csv")

    # --- MODULE 3: GEOGRAPHIC EXPOSURE ---
    with tab_geo:
            st.subheader("European Risk Mapping")
            
            # On donne encore plus de largeur à la carte visuelle (ratio 1 pour les boutons, 4 pour la carte)
            col_geo_ctrl, col_geo_viz = st.columns([1, 4])
            
            with col_geo_ctrl:
                geo_sc = st.selectbox("Climate Scenario", SCENARIOS, key="geo_sc_2")
                geo_metric = st.radio("Indicator to map", ["Projected Losses (€)", "Loss Rate (bps)", "Exposure (EAD)"])
                
            df_geo = summarize(all_results[geo_sc], "country")
            metric_col_map = {"Projected Losses (€)": "loss_projected", "Loss Rate (bps)": "Loss_Rate_bps", "Exposure (EAD)": "EAD_EUR"}
            target_col = metric_col_map[geo_metric]
            color_scale = "Blues" if target_col == "EAD_EUR" else "Reds"

            with col_geo_viz:
                df_geo["EAD (Euro)"] = df_geo["EAD_EUR"].apply(lambda x: f"{x:,.0f} €")
                df_geo["Loss (Euro)"] = df_geo["loss_projected"].apply(lambda x: f"{x:,.0f} €")
                df_geo["Rate (bps)"] = df_geo["Loss_Rate_bps"].apply(lambda x: f"{x:.1f} bps")

                # Nouvelle carte interactive Mapbox en mode sombre
                fig_map = px.choropleth_mapbox(
                    df_geo, 
                    geojson="https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json",
                    featureidkey="properties.name", # Relie vos noms de pays au fichier géographique
                    locations="country", 
                    color=target_col, 
                    hover_name="country",
                    hover_data={"country": False, target_col: False, "EAD (Euro)": True, "Loss (Euro)": True, "Rate (bps)": True},
                    color_continuous_scale=color_scale, 
                    title=f"Spatial distribution: {geo_metric} ({geo_sc})",
                    mapbox_style="carto-darkmatter", # Le style sombre professionnel type "Google Maps de nuit"
                    zoom=3.0, # Niveau de zoom idéal pour l'Europe
                    center={"lat": 50.0, "lon": 10.0}, # Centrage sur le continent européen
                    opacity=0.7 # Légère transparence des couleurs pour voir les villes/routes en dessous
                )
                
                # Suppression des anciennes marges et forçage de la grande hauteur
                fig_map.update_layout(
                    font_family="Inter", 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    margin={"r":0,"t":40,"l":0,"b":0},
                    height=700 
                ) 
                
                # Affichage en utilisant 100% de la largeur du conteneur
                st.plotly_chart(fig_map, use_container_width=True)
                
            st.dataframe(
                df_geo[["country", "EAD_EUR", "loss_projected", "Loss_Rate_bps"]].style.background_gradient(subset=['Loss_Rate_bps'], cmap='Reds').format({
                    "EAD_EUR": "{:,.0f} €", "loss_projected": "{:,.0f} €", "Loss_Rate_bps": "{:.1f}"
                }), use_container_width=True
            )
    # --- MODULE 4: METHODOLOGY & FRAMEWORK ---
    with tab_meth:
        st.subheader("Methodological Framework and Mathematical Model")
        st.markdown("This dashboard relies on a static modeling framework inspired by regulatory climate stress testing exercises (ACPR/ECB). It assesses the financial impact of physical and transition risks on fundamental credit risk parameters.")
        
        st.markdown("#### 🗺️ Risk Transmission Map")
        st.info("Transmission mechanism of climate shocks to the bank's balance sheet:")
        
        # Astuce : on isole le HTML dans une variable pour éviter les bugs d'affichage de Streamlit
        html_content = """
        <div style="text-align: center; padding: 20px; background-color: #ffffff; border-radius: 12px; border: 1px solid #e0e6ed; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.02);">
            <h4 style="color: #1e293b; margin-bottom: 5px;">🌍 1. Climate Scenarios (NGFS)</h4>
            <p style="color: #64748b; font-size: 0.9em;">Trajectory definition (Optimistic, Neutral, Pessimistic)</p>
            <h3 style="color: #cbd5e1; margin: 10px 0;">⬇️</h3>
            <h4 style="color: #3498db; margin-bottom: 5px;">🏭 2. Macro-Sectoral Shocks</h4>
            <p style="color: #64748b; font-size: 0.9em;">Application of climate penalties ("Uplifts") by business sector</p>
            <h3 style="color: #cbd5e1; margin: 10px 0;">⬇️</h3>
            <h4 style="color: #e74c3c; margin-bottom: 5px;">📉 3. Credit Risk Degradation</h4>
            <p style="color: #64748b; font-size: 0.9em;">Stress on Probability of Default (PD) and Loss Given Default (LGD)</p>
            <h3 style="color: #cbd5e1; margin: 10px 0;">⬇️</h3>
            <h4 style="color: #2ecc71; margin-bottom: 5px;">💶 4. Balance Sheet Impact</h4>
            <p style="color: #64748b; font-size: 0.9em;">Calculation of Projected Losses on EAD and Climate Value at Risk (VaR)</p>
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)
        
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### 1. Shocks on Probability of Default (PD)")
            st.markdown("The model applies a macro-sectoral penalty (uplift) to the base probability of default of each counterparty. The result is mathematically capped at 1 (certain default).")
            st.latex(r"PD_{stress} = \min(PD_{base} \times (1 + \Delta PD_{sectoral}), 1.0)")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### 2. Shocks on Loss Given Default (LGD)")
            st.markdown("The LGD undergoes a degradation linked to the potential loss of value of underlying collaterals, particularly affected by physical risks (floods, droughts).")
            st.latex(r"LGD_{stress} = \min(LGD_{base} \times (1 + \Delta LGD_{sectoral}), 1.0)")

        with col_m2:
            st.markdown("#### 3. Calculation of Projected Loss (Expected Loss)")
            st.markdown("The additional financial loss specifically induced by the climate scenario is calculated based on the risk delta (PD increase) and the new stressed LGD.")
            st.latex(r"\text{Loss}_{projected} = EAD \times \max(PD_{stress} - PD_{base}, 0) \times LGD_{stress}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### 4. Climate Value at Risk (VaR)")
            st.markdown("The Climate VaR represents the estimation of the extreme loss at a defined confidence level, calculated empirically on the loss distribution of all simulated scenarios.")
            st.latex(r"\text{Climate VaR}_{\alpha} = \text{Quantile}(\text{Losses}_{scenarios}, \alpha)")
            
        st.markdown("---")
        st.info("💡 **Static Balance Sheet Assumption:** EAD (Exposure At Default) is kept constant in this exercise. The model does not simulate portfolio renewal or mitigation (hedging) strategies by the bank. The climate impact is therefore read exclusively through credit risk migration.")
        
        
    # --- MODULE 5: DASHBOARD CLIENT ---

    with tab_sim:
        st.subheader("Simulateur de Résilience du Portefeuille (Monte Carlo)")
        st.markdown("Testez l'évolution de vos finances face aux incertitudes du marché et aux chocs climatiques.")

    # Formulaire de saisie des données client optimisé
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        init_inv = st.number_input("Investissement Initial (€)", value=1000000, step=100000)
        time_horizon = st.slider("Horizon de placement (Années)", min_value=1, max_value=30, value=10)
        
    with col_in2:
        mu = st.number_input("Rendement Annuel Espéré (%)", value=5.0, step=0.5) / 100
        sigma = st.number_input("Volatilité Annuelle (%)", value=15.0, step=1.0) / 100
        
    with col_in3:
        climate_shock_mu = st.number_input("Choc Climat sur Rendement (%)", value=1.5, step=0.5, help="Baisse du rendement due aux risques de transition") / 100
        climate_shock_vol = st.number_input("Choc Climat sur Volatilité (%)", value=2.0, step=0.5, help="Hausse de l'incertitude due aux événements climatiques") / 100
        n_sims = st.selectbox("Nombre de Scénarios", [100, 500, 1000, 5000], index=2)

    # Moteur de calcul au clic
    if st.button("Lancer la Simulation de Monte Carlo", use_container_width=True):
        np.random.seed(42) # Fixe l'aléatoire pour la reproductibilité
        dt = 1
        
        # Ajustement des paramètres stochastiques avec le risque climat
        adj_mu = mu - climate_shock_mu
        adj_sigma = sigma + climate_shock_vol 
        
        # Création de la matrice des trajectoires
        paths = np.zeros((time_horizon + 1, n_sims))
        paths[0] = init_inv
        
        # Simulation stochastique (Mouvement Brownien Géométrique)
        for t in range(1, time_horizon + 1):
            Z = np.random.standard_normal(n_sims)
            paths[t] = paths[t-1] * np.exp((adj_mu - 0.5 * adj_sigma**2) * dt + adj_sigma * np.sqrt(dt) * Z)
        
        # Extraction des métriques de risques pour l'affichage
        final_values = paths[-1]
        var_95 = np.percentile(final_values, 5)
        exp_shortfall = np.mean(final_values[final_values < var_95])
        mean_path = np.mean(paths, axis=1)
        quant_5 = np.percentile(paths, 5, axis=1)
        quant_95 = np.percentile(paths, 95, axis=1)

        # --- Affichage des KPI Financiers ---
        st.markdown("### Analyse des Risques (Fin de période)")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Valeur Finale Moyenne", f"{mean_path[-1]:,.0f} €")
        col_res2.metric("Pire Scénario 5% (Value at Risk)", f"{var_95:,.0f} €", delta=f"{var_95 - init_inv:,.0f} €", delta_color="inverse")
        col_res3.metric("Déficit Attendu (CVaR)", f"{exp_shortfall:,.0f} €")

        # --- Graphique 1 : Trajectoires avec Intervalle de Confiance ---
        fig_paths = go.Figure()
        
        # Ajout du cône d'incertitude (90% des scénarios)
        fig_paths.add_trace(go.Scatter(
            x=np.concatenate([np.arange(time_horizon + 1), np.arange(time_horizon + 1)[::-1]]),
            y=np.concatenate([quant_95, quant_5[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 75, 75, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalle de Confiance (90%)'
        ))

        # Affichage d'un échantillon réduit de trajectoires pour la performance web
        for i in range(min(50, n_sims)):
            fig_paths.add_trace(go.Scatter(
                x=np.arange(time_horizon + 1), y=paths[:, i], 
                mode='lines', line=dict(width=1, color='rgba(0, 150, 255, 0.1)'), showlegend=False
            ))
        
        # Trajectoire moyenne
        fig_paths.add_trace(go.Scatter(
            x=np.arange(time_horizon + 1), y=mean_path, 
            mode='lines', line=dict(width=3, color='#ff4b4b'), name='Trajectoire Moyenne'
        ))
        
        fig_paths.update_layout(title="Trajectoires Simulées et Impact Climatique", xaxis_title="Années", yaxis_title="Valeur du Portefeuille (€)", template="plotly_dark", height=450)
        
        # --- Graphique 2 : Histogramme de Distribution (VaR) ---
        fig_hist = go.Figure(data=[go.Histogram(x=final_values, nbinsx=60, marker_color='#0096ff')])
        fig_hist.add_vline(x=var_95, line_dash="dash", line_color="#ff4b4b", line_width=3, annotation_text="VaR 95%", annotation_position="top right")
        fig_hist.update_layout(title="Distribution des Valeurs Finales du Portefeuille (Horizon atteint)", xaxis_title="Valeur du Portefeuille (€)", yaxis_title="Fréquence des Scénarios", template="plotly_dark", height=350)

        # Rendu sur Streamlit
        st.plotly_chart(fig_paths, use_container_width=True)
        st.plotly_chart(fig_hist, use_container_width=True)


if __name__ == "__main__":
    main()