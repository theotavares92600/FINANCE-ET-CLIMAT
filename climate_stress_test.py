from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass

# --- CONFIGURATION AND ADVANCED STYLE ---
st.set_page_config(page_title="Climate Risk Intelligence", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    /* 1. Import 'Inter' font (widely used in finance/tech) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* 2. Soft global background to make charts stand out */
    .stApp {
        background-color: #f4f7f6;
    }

    /* 3. KPIs Cards Design (Metrics) */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e6ed;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }
    
    /* Hover lift effect for KPIs */
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }
    
    [data-testid="stMetricValue"] {
        color: #1e293b;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 1.05rem;
        font-weight: 600;
    }

    /* 4. Tabs styling for a pro Dashboard look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        border: 1px solid #e0e6ed;
        border-bottom: none;
        padding: 0 20px;
        color: #64748b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f172a; /* Very dark blue/Grey */
        color: #ffffff !important;
    }

    /* Top bar cleanup */
    [data-testid="stHeader"] { 
        background: rgba(255,255,255,0); 
    }
    </style>
    """, unsafe_allow_html=True)

SCENARIOS = ["Optimistic", "Neutral", "Pessimistic"]

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
    _require_columns(uplifts, ["sector", "pd_uplift_Optimistic", "pd_uplift_Neutral", "pd_uplift_Pessimistic", "lgd_uplift_Optimistic", "lgd_uplift_Neutral", "lgd_uplift_Pessimistic"], "Scenario_Uplifts")
    
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
    # Converts the DataFrame to a web-formatted CSV
    return df.to_csv(index=False).encode('utf-8')

# --- USER INTERFACE (Front-end) ---
def main():
    st.title("🏛️ Climate Risk Intelligence Platform")
    st.caption("Prudential analysis of transition and physical risks — ACPR/ECB Framework")
    st.markdown("---")

    # SIDEBAR
    st.sidebar.header("🕹️ Analytical Controls")
    alpha_input = st.sidebar.select_slider("VaR Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
    cfg = StressTestConfig(alpha=alpha_input)

    try:
        portfolio, uplifts = load_inputs("portfolio_climat.xlsx")
    except Exception as e:
        st.error(f"Source file not found: {e}")
        return

    # Global calculations
    all_results = {sc: apply_scenario(portfolio, uplifts, sc, cfg) for sc in SCENARIOS}
    scenario_totals = [{"Scenario": sc, "Losses": all_results[sc]["loss_projected"].sum()} for sc in SCENARIOS]
    
    # MODULE NAVIGATION (Tabs)
    tab_macro, tab_micro, tab_geo, tab_meth = st.tabs([
        "📊 Module 1: Executive Dashboard", 
        "🔍 Module 2: Portfolio Deep-Dive",
        "🌍 Module 3: Geographic Exposure",
        "📖 Module 4: Methodology & Framework"
    ])

    # --- MODULE 1: EXECUTIVE DASHBOARD ---
    with tab_macro:
        st.subheader("Aggregated Risk Analysis")
        
        # 1. Enriched high-level Metrics
        m1, m2, m3 = st.columns(3)
        total_ead = portfolio["EAD_EUR"].sum()
        max_loss = max([s["Losses"] for s in scenario_totals])
        var_value = climate_var([s['Losses'] for s in scenario_totals], cfg.alpha)
        
        m1.metric("Total Exposure (EAD)", f"{total_ead/1e6:,.1f}M €")
        m2.metric(
            "Maximum Projected Loss", 
            f"{max_loss:,.0f} €", 
            delta=f"{(max_loss/total_ead)*10000:.1f} bps of EAD", 
            delta_color="inverse"
        )
        m3.metric(f"Climate VaR ({int(cfg.alpha*100)}%)", f"{var_value:,.0f} €")

        st.markdown("---")

        # 2. Charts Section in 2 columns
        col1_macro, col2_macro = st.columns(2)

        with col1_macro:
            st.markdown("#### Losses by Climate Scenario")
            df_totals = pd.DataFrame(scenario_totals)
            fig_bar = px.bar(
                df_totals, x="Scenario", y="Losses", color="Scenario",
                color_discrete_map={"Optimistic": "#2ecc71", "Neutral": "#f1c40f", "Pessimistic": "#e74c3c"}
            )
            # --> DESIGN LAYOUT UPDATE:
            fig_bar.update_layout(
                font_family="Inter",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(gridcolor='#e2e8f0', title=""),
                showlegend=False, 
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2_macro:
            st.markdown("#### Portfolio Composition (EAD by Sector)")
            df_ead_sector = portfolio.groupby("sector")["EAD_EUR"].sum().reset_index()
            fig_donut = px.pie(
                df_ead_sector, values="EAD_EUR", names="sector", hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_donut.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")
        
        # 3. Alert Zone: Top 3 vulnerable sectors (Based on the worst scenario)
        st.markdown("#### 🚨 Top 3 Most Vulnerable Sectors (Pessimistic Scenario)")
        
        df_pessimistic = all_results["Pessimistic"]
        df_top_sectors = summarize(df_pessimistic, "sector").head(3)
        top_sectors = df_top_sectors.to_dict('records')

        t1, t2, t3 = st.columns(3)
        
        if len(top_sectors) > 0:
            t1.error(f"**1. {top_sectors[0]['sector']}**\n\n**Loss:** {top_sectors[0]['loss_projected']:,.0f} €\n\n**Rate:** {top_sectors[0]['Loss_Rate_bps']:.1f} bps")
        if len(top_sectors) > 1:
            t2.warning(f"**2. {top_sectors[1]['sector']}**\n\n**Loss:** {top_sectors[1]['loss_projected']:,.0f} €\n\n**Rate:** {top_sectors[1]['Loss_Rate_bps']:.1f} bps")
        if len(top_sectors) > 2:
            t3.info(f"**3. {top_sectors[2]['sector']}**\n\n**Loss:** {top_sectors[2]['loss_projected']:,.0f} €\n\n**Rate:** {top_sectors[2]['Loss_Rate_bps']:.1f} bps")

    # --- MODULE 2: PORTFOLIO DEEP-DIVE ---
    with tab_micro:
        st.subheader("Granular Analysis & Concentration")
        
        col_ctrl, col_viz = st.columns([1, 3])
        
        with col_ctrl:
            target_sc = st.selectbox("Target Scenario", SCENARIOS, key="micro_sc")
            dim = st.radio("Group by", ["sector", "country", "region"])
            df_dim = summarize(all_results[target_sc], dim)

        with col_viz:
            fig_tree = px.treemap(df_dim, path=[dim], values='loss_projected',
                                 color='Loss_Rate_bps', color_continuous_scale='RdYlGn_r',
                                 title=f"Loss concentration by {dim} (bps)")
            st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown("#### Position Details (Top 50)")
        st.dataframe(
            all_results[target_sc].sort_values("loss_projected", ascending=False).head(50).style.format({
                "EAD_EUR": "{:,.0f} €",
                "PD_base": "{:.2%}",
                "PD_stress": "{:.2%}",
                "loss_projected": "{:,.2f} €"
            }), use_container_width=True
        )
        
        # --- EXPORT BUTTON ---
        st.markdown("---")
        csv_data = convert_df(all_results[target_sc])
        st.download_button(
            label=f"📥 Download full results ({target_sc}) as CSV",
            data=csv_data,
            file_name=f"stress_test_results_{target_sc.lower()}.csv",
            mime="text/csv",
        )

    # --- MODULE 3: GEOGRAPHIC EXPOSURE ---
    with tab_geo:
        st.subheader("European Risk Mapping")
        
        col_geo_ctrl, col_geo_viz = st.columns([1, 3])
        
        with col_geo_ctrl:
            geo_sc = st.selectbox("Climate Scenario", SCENARIOS, key="geo_sc")
            geo_metric = st.radio("Indicator to map", [
                "Projected Losses (€)", 
                "Loss Rate (bps)", 
                "Exposure (EAD)"
            ])
            
        df_geo = summarize(all_results[geo_sc], "country")
        
        metric_col_map = {
            "Projected Losses (€)": "loss_projected",
            "Loss Rate (bps)": "Loss_Rate_bps",
            "Exposure (EAD)": "EAD_EUR"
        }
        target_col = metric_col_map[geo_metric]
        color_scale = "Blues" if target_col == "EAD_EUR" else "Reds"

        with col_geo_viz:
            # Creation of cleanly formatted columns for hover display
            df_geo["EAD (Euro)"] = df_geo["EAD_EUR"].apply(lambda x: f"{x:,.0f} €")
            df_geo["Loss (Euro)"] = df_geo["loss_projected"].apply(lambda x: f"{x:,.0f} €")
            df_geo["Rate (bps)"] = df_geo["Loss_Rate_bps"].apply(lambda x: f"{x:.1f} bps")

            fig_map = px.choropleth(
                df_geo,
                locations="country", 
                locationmode="country names",
                color=target_col,
                hover_name="country",
                # Enriched tooltip configuration
                hover_data={
                    "country": False,
                    target_col: False,
                    "EAD (Euro)": True,
                    "Loss (Euro)": True,
                    "Rate (bps)": True
                },
                color_continuous_scale=color_scale,
                title=f"Spatial distribution: {geo_metric} ({geo_sc})"
            )
            fig_map.update_geos(scope="europe", fitbounds="locations", showcountries=True, countrycolor="Black")
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            
            st.plotly_chart(fig_map, use_container_width=True)
            
        # Table with heat gradient (Heatmap) on the risk column
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
        
    # --- MODULE 4: METHODOLOGY & FRAMEWORK ---
    with tab_meth:
        st.subheader("Methodological Framework and Mathematical Model")
        st.markdown("This dashboard relies on a static modeling framework inspired by regulatory climate stress testing exercises (ACPR/ECB). It assesses the financial impact of physical and transition risks on fundamental credit risk parameters.")
        
        # ---> ADDITION OF THE METHODOLOGICAL MAP (DIAGRAM) <---
        st.markdown("#### 🗺️ Risk Transmission Map")
        st.info("Transmission mechanism of climate shocks to the bank's balance sheet:")
        
        st.markdown("""
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
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ---> CONTINUATION OF THE CODE WITH MATHEMATICAL FORMULAS <---
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

if __name__ == "__main__":
    main()