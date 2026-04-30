"""Streamlit dashboard for soccer recruitment analytics."""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import requests

# Configure page
st.set_page_config(
    page_title="SoccerRecruit ML",
    page_icon="⚽",
    layout="wide"
)

# API base URL — configurable via env var, fallback to localhost
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")


def fetch_players(limit: int = 100) -> List[Dict]:
    """Fetch players from API."""
    try:
        response = requests.get(f"{API_BASE}/players?limit={limit}", timeout=5)
        if response.status_code == 200:
            return response.json().get("players", [])
        elif response.status_code == 503:
            st.warning("Model not trained yet. Go to Similarity Analysis → train the model first.")
    except requests.exceptions.RequestException:
        st.error("Cannot connect to API. Make sure the server is running.")
    return []


def fetch_similar_players(player_id: int, top_k: int = 5) -> Dict:
    """Fetch similar players from API."""
    try:
        response = requests.get(
            f"{API_BASE}/similarity/{player_id}/similar?top_k={top_k}",
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.warning("Similarity model not trained. Train it via POST /similarity/train first.")
    except requests.exceptions.RequestException:
        st.error("Cannot connect to API.")
    return {}


def fetch_prediction_by_id(player_id: int, overrides: Dict = None) -> Dict:
    """Fetch prediction by player ID from API."""
    try:
        payload = {"player_id": player_id}
        if overrides:
            payload["overrides"] = overrides
        response = requests.post(
            f"{API_BASE}/predict/value/{player_id}",
            json=payload,
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        st.error("Cannot connect to API.")
    return {}


def main():
    """Main dashboard."""
    st.title("⚽ SoccerRecruit ML")
    st.markdown("### Player Similarity & Recruitment Modeling Platform")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Player Explorer", "Similarity Analysis", "Value Prediction", "Analytics"],
    )

    if page == "Player Explorer":
        player_explorer_page()
    elif page == "Similarity Analysis":
        similarity_page()
    elif page == "Value Prediction":
        prediction_page()
    else:
        analytics_page()


def player_explorer_page():
    """Player explorer page."""
    st.header("Player Explorer")

    players = fetch_players()

    if not players:
        st.info("No player data available. Load sample data first.")
        return

    df = pd.DataFrame(players)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        position_filter = st.selectbox(
            "Position",
            ["All"] + sorted(df["position"].dropna().unique().tolist()),
        )

    with col2:
        league_filter = st.selectbox(
            "League",
            ["All"] + sorted(df["league"].dropna().unique().tolist()),
        )

    with col3:
        min_value = st.slider("Min Value (€)", 0, 200000000, 0, step=5000000)

    # Apply filters
    filtered_df = df.copy()
    if position_filter != "All":
        filtered_df = filtered_df[filtered_df["position"] == position_filter]
    if league_filter != "All":
        filtered_df = filtered_df[filtered_df["league"] == league_filter]
    if min_value > 0:
        filtered_df = filtered_df[filtered_df["value"] >= min_value]

    st.write(f"Showing {len(filtered_df)} players")

    # Display table
    display_cols = ["name", "age", "position", "team", "league", "value", "goals", "assists"]
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[available_cols], use_container_width=True)


def similarity_page():
    """Similarity analysis page."""
    st.header("Similarity Analysis")

    players = fetch_players(limit=50)

    if not players:
        st.info("No player data available.")
        return

    # Select player
    player_names = {p["name"]: p["id"] for p in players}
    selected_name = st.selectbox("Select Player", list(player_names.keys()))

    if selected_name:
        player_id = player_names[selected_name]
        similar_data = fetch_similar_players(player_id)

        if similar_data:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Player Info")
                st.write(f"**Name:** {similar_data.get('player_name', 'N/A')}")

                st.subheader("Similar Players")
                for sp in similar_data.get("similar_players", []):
                    st.write(
                        f"Player ID: {sp['player_id']} | "
                        f"Similarity: {sp['similarity']:.2%}"
                    )

            with col2:
                sim_players = similar_data.get("similar_players", [])
                if sim_players:
                    fig = px.bar(
                        x=[f"Player {sp['player_id']}" for sp in sim_players],
                        y=[sp["similarity"] for sp in sim_players],
                        labels={"x": "Player", "y": "Similarity"},
                        title="Similarity Scores",
                    )
                    st.plotly_chart(fig, use_container_width=True)


def prediction_page():
    """Value prediction page."""
    st.header("Value Prediction")

    players = fetch_players(limit=100)

    if not players:
        st.info("No player data available. Load sample data first.")
        return

    use_mode = st.radio("Input mode", ["Enter features manually", "Use existing player"])

    if use_mode == "Use existing player":
        player_names = {p["name"]: p["id"] for p in players}
        selected_name = st.selectbox("Select Player", [""] + list(player_names.keys()))

        if selected_name:
            player_id = player_names[selected_name]
            result = fetch_prediction_by_id(player_id)

            if result:
                st.success(f"**Predicted Value:** €{result['predicted_value']:,.0f}")
                if result.get("confidence_interval"):
                    ci = result["confidence_interval"]
                    st.write(
                        f"**CI:** €{ci['lower']:,.0f} – €{ci['upper']:,.0f}"
                    )
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 16, 45, 25)
            height = st.number_input("Height (cm)", 150, 220, 180)
            weight = st.number_input("Weight (kg)", 50, 120, 75)
            appearances = st.number_input("Appearances", 0, 500, 100)

        with col2:
            minutes = st.number_input("Minutes Played", 0, 20000, 5000)
            goals = st.number_input("Goals", 0, 200, 10)
            assists = st.number_input("Assists", 0, 100, 5)
            pass_acc = st.slider("Pass Accuracy (%)", 0, 100, 80)

        with col3:
            shots = st.number_input("Shots per Game", 0.0, 10.0, 2.0)
            tackles = st.number_input("Tackles", 0, 500, 50)
            interceptions = st.number_input("Interceptions", 0, 300, 30)
            wage = st.number_input("Weekly Wage (€)", 0, 1000000, 50000, step=5000)

        if st.button("Predict Value"):
            features = {
                "age": age,
                "height": height,
                "weight": weight,
                "appearances": appearances,
                "minutes_played": minutes,
                "goals": goals,
                "assists": assists,
                "pass_accuracy": pass_acc,
                "shots_per_game": shots,
                "tackles": tackles,
                "interceptions": interceptions,
                "saves": 0,
                "clean_sheets": 0,
                "wage": wage,
            }

            result = fetch_prediction_by_id(players[0]["id"], overrides=features)

            if result:
                st.success(f"**Predicted Value:** €{result['predicted_value']:,.0f}")
                if result.get("confidence_interval"):
                    ci = result["confidence_interval"]
                    st.write(
                        f"**CI:** €{ci['lower']:,.0f} – €{ci['upper']:,.0f}"
                    )


def analytics_page():
    """Analytics dashboard page."""
    st.header("Analytics Dashboard")

    players = fetch_players()

    if not players:
        st.info("No player data available.")
        return

    df = pd.DataFrame(players)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Players", len(df))
    with col2:
        avg_val = df["value"].mean() if "value" in df.columns else 0
        st.metric("Avg Value", f"€{avg_val:,.0f}")
    with col3:
        total_goals = df["goals"].sum() if "goals" in df.columns else 0
        st.metric("Total Goals", f"{total_goals:,.0f}")
    with col4:
        avg_age = df["age"].mean() if "age" in df.columns else 0
        st.metric("Avg Age", f"{avg_age:.1f}")

    col1, col2 = st.columns(2)

    with col1:
        if "position" in df.columns and "value" in df.columns:
            fig = px.box(
                df,
                x="position",
                y="value",
                title="Player Value by Position",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "league" in df.columns and "value" in df.columns:
            league_val = df.groupby("league")["value"].sum().reset_index()
            fig = px.pie(
                league_val,
                values="value",
                names="league",
                title="Total Value by League",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
