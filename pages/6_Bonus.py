# pages/6_Bonus.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# Hide this page from sidebar navigation
st.set_page_config(
    page_title="F1 Prediction", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove the page from sidebar (this must be the first Streamlit command)
st.markdown("<style>#MainMenu {visibility: hidden;}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="F1 Prediction", layout="wide")

st.title("🏎️ F1 Singapore Grand Prix 2025 Winner Predictor")
st.markdown("### *Advanced Machine Learning Simulation*")
st.markdown("---")

st.warning("🚨 **WARNING**: This is a highly sophisticated AI-powered prediction system. Results may be too accurate!")

# List of F1 drivers (current + some classics for fun)
drivers = [
    "Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Lando Norris",
    "George Russell", "Carlos Sainz", "Fernando Alonso", "Oscar Piastri",
    "Sergio Pérez", "Yuki Tsunoda", "Daniel Ricciardo", "Alexander Albon",
    "Pierre Gasly", "Esteban Ocon", "Lance Stroll", "Nico Hülkenberg",
    "Kevin Magnussen", "Valtteri Bottas", "Zhou Guanyu", "Logan Sargeant",
    "Sebastian Vettel", "Michael Schumacher", "Ayrton Senna", "Alain Prost"
]

# Singapore-specific factors
singapore_factors = [
    "Marina Bay Street Circuit complexity",
    "Humidity levels at 85%",
    "Safety Car probability: HIGH",
    "Tyre degradation on 23 turns",
    "Floodlights performance under pressure",
    "Singapore Sling corner mastery",
    "Heat management in 30°C weather",
    "Overtaking opportunities at Turn 14",
    "Energy recovery system efficiency",
    "Pit stop strategy under Virtual Safety Car"
]

# Team radio messages
team_radios = [
    "Box box box, we are switching strategy!",
    "Manage your brakes, temperatures are critical",
    "Push now! Push push push!",
    "Safety Car deployed, repeat Safety Car deployed!",
    "Watch for debris at Turn 10",
    "DRS enabled, you are clear to attack",
    "Fuel mix 3, we need to save energy",
    "Yellow flag sector 2, be careful",
    "We are checking, we are checking...",
    "Get in there Lewis! Oh wait, wrong driver...",
    "Multi 21 Sebastian, multi 21!",
    "No Michael no, that was so not right!",
    "Leave me alone, I know what I'm doing!",
    "GP2 engine, GP2! Aaargh!"
]

def simulate_ml_analysis():
    """Simulate a complex ML analysis with fun outputs"""
    
    # Create progress bars and status updates
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Phase 1: Data Collection
    status_text.text("📊 Collecting historical F1 data from 2008-2024...")
    for i in range(10):
        progress_bar.progress((i + 1) * 5)
        time.sleep(0.1)
    
    # Phase 2: Feature Engineering
    status_text.text("🔧 Engineering Singapore-specific features...")
    for i in range(10, 30):
        progress_bar.progress(i)
        time.sleep(0.05)
    
    # Show some Singapore factors being analyzed
    with st.expander("🔍 Analyzing Singapore GP Factors"):
        for factor in random.sample(singapore_factors, 5):
            st.write(f"✓ {factor}")
            time.sleep(0.3)
    
    # Phase 3: Model Training
    status_text.text("🤖 Training Neural Network on 50,000 lap simulations...")
    for i in range(30, 70):
        progress_bar.progress(i)
        time.sleep(0.02)
    
    # Phase 4: Monte Carlo Simulations
    status_text.text("🎲 Running 10,000 Monte Carlo race simulations...")
    for i in range(70, 90):
        progress_bar.progress(i)
        time.sleep(0.03)
    
    # Phase 5: Final Analysis
    status_text.text("🎯 Calculating optimal strategy and winner probability...")
    for i in range(90, 100):
        progress_bar.progress(i)
        time.sleep(0.1)
    
    progress_bar.progress(100)
    time.sleep(0.5)
    
    return True

def display_winner_animation(winner):
    """Display the winner with fun animation"""
    
    st.balloons()
    
    # Winner announcement
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(45deg, #FFD700, #FFED4E); border-radius: 10px; margin: 20px 0;'>
        <h1 style='color: #000; margin: 0;'>🏆 CONGRATULATIONS! 🏆</h1>
        <h2 style='color: #000; margin: 10px 0;'>{winner}</h2>
        <p style='color: #000; font-size: 20px;'>Predicted Winner of 2025 Singapore Grand Prix!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Race stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction Confidence", f"{random.randint(85, 98)}%")
    
    with col2:
        st.metric("Winning Margin", f"+{random.uniform(0.5, 15.2):.1f}s")
    
    with col3:
        st.metric("Strategy", random.choice(["2-Stop Soft", "1-Stop Medium", "3-Stop Aggressive"]))
    
    # Team radio simulation
    st.markdown("---")
    st.subheader("📻 Simulated Team Radio")
    
    radio_message = random.choice(team_radios)
    st.info(f"**Team Radio**: *'{radio_message}'*")
    
    # Podium prediction
    st.markdown("---")
    st.subheader("🏁 Predicted Podium")
    
    # Get 3 unique drivers including the winner
    podium_drivers = [winner]
    available_drivers = [d for d in drivers if d != winner]
    podium_drivers.extend(random.sample(available_drivers, 2))
    
    cols = st.columns(3)
    positions = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
    
    for i, (col, driver, position) in enumerate(zip(cols, podium_drivers, positions)):
        with col:
            if i == 0:  # Winner gets special treatment
                st.markdown(f"<h3 style='text-align: center; color: #FFD700;'>{position}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>{driver}</h4>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='text-align: center;'>{position}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>{driver}</h4>", unsafe_allow_html=True)

# Main app
st.markdown("""
This advanced prediction system analyzes:
- **50+ years** of F1 historical data
- **Singapore-specific** track characteristics  
- **Real-time** weather patterns for 2025
- **Machine Learning** models trained on 1M+ laps
- **Monte Carlo** simulations of race strategies
""")

if st.button("🚀 Run F1 2025 Singapore GP Prediction", type="primary"):
    
    with st.spinner("Initializing quantum computing cluster..."):
        time.sleep(2)
    
    # Run the simulation
    if simulate_ml_analysis():
        
        # Randomly select a winner (but weight towards popular drivers)
        weighted_drivers = drivers.copy()
        # Add some extra entries for more popular drivers
        weighted_drivers.extend(["Max Verstappen"] * 3)
        weighted_drivers.extend(["Lewis Hamilton"] * 2)
        weighted_drivers.extend(["Charles Leclerc"] * 2)
        weighted_drivers.extend(["Lando Norris"] * 2)
        
        winner = random.choice(weighted_drivers)
        
        # Display results
        display_winner_animation(winner)

# Easter egg footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <i>Disclaimer: This is a fun simulation for F1 weekend entertainment. 
    Actual race results may vary. No AI was harmed in the making of this prediction.</i>
    <br><br>
    🏁 Enjoy the 2025 Singapore Grand Prix! 🏁
</div>
""", unsafe_allow_html=True)

# Secret developer note (hidden)
with st.expander("🔧 Developer Notes (shhh...)"):
    st.code("""
    # ACTUAL PREDICTION ALGORITHM:
    import random
    
    def predict_f1_winner_2025():
        drivers = ["Max", "Lewis", "Charles", "Lando", "Everyone else"]
        return random.choice(drivers)
    
    # Advanced ML complete! 🎉
    """)
