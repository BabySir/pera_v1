import streamlit as st
import json
import os
import time
from src.engine import PeraBrain
from src.nano_llm import NanoLLM 

def save_upgrade_profile(profile_data):
    file_path = 'data/sample_patient_data.json'
    os.makedirs('data', exist_ok=True)
    
    # Generate a unique ID if it's missing
    if 'patient_id' not in profile_data:
        profile_data['patient_id'] = f"PID-{int(time.time())}"
    
    all_profiles = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                all_profiles = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                all_profiles = []
                
    all_profiles.append(profile_data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_profiles, f, indent=4)

@st.cache_resource
def load_pera_brain():
    my_model = NanoLLM() 
    return PeraBrain(nano_llm=my_model)

def main():
    st.set_page_config(page_title="PeRA - Your Life Upgrade", page_icon="🌱", layout="wide")

    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'new_user_data' not in st.session_state:
        st.session_state.new_user_data = {}

    brain = load_pera_brain()

    # --- Step 1: Physical Focus ---
    if st.session_state.step == 1:
        st.header("🌱 Step 1: Checking In")
        with st.form("step1_form"):
            focus_options = [
                "General Wellness & Ergonomics",
                "Addiction Recovery (Nicotine/Alcohol)",
                "Sports Injury (Hamstring/Ankle)",
                "Post-Operative (Knee/Spinal/Shoulder)",
                "Chronic Pain (Sciatica/Fibromyalgia)",
                "Tech Neck / Upper Body Tension",
                "Lower Body Joint Pain"
            ]
            focus = st.selectbox("Where shall we focus today?", focus_options)
            
            # 🎚️ Dynamic Slider Logic
            if "Addiction" in focus:
                comfort = st.select_slider("Craving intensity today?", options=["Low", "Manageable", "Strong", "Severe ⛈️"])
            elif "Pain" in focus or "Injury" in focus or "Post-Operative" in focus:
                comfort = st.select_slider("Pain/Discomfort level (1-10)?", options=[str(i) for i in range(1, 11)])
            else:
                comfort = st.select_slider("How are you feeling?", options=["1 ⛈️", "2 ☁️", "3 🌤️", "4 ☀️"])
                
            story = st.text_area("Tell me more details... (e.g., 'Day 5 post-op' or 'High cravings today')")
            
            if st.form_submit_button("Next 🤍"):
                st.session_state.new_user_data["physical"] = {"focus": focus, "comfort": comfort, "story": story}
                st.session_state.step = 2
                st.rerun()

    # --- Step 2: Daily Rhythm ---
    elif st.session_state.step == 2:
        st.header("🕰️ Step 2: Your Daily Rhythm")
        with st.form("step2_form"):
            work = st.selectbox("Working hours style?", ["Sitting", "Mixed", "Active"])
            sleep = st.select_slider("Morning energy?", options=["Low", "Medium", "High"])
            if st.form_submit_button("Next 🤍"):
                st.session_state.new_user_data["lifestyle"] = {"work": work, "sleep": sleep}
                st.session_state.step = 3
                st.rerun()

    # --- Step 3: Mindset & Goal ---
    elif st.session_state.step == 3:
        st.header("🧠 Step 3: Mind-Body Connection")
        with st.form("step3_form"):
            mindset = st.radio("Current feeling?", ["Worried", "Determined", "Frustrated"])
            goal = st.text_input("Your 'Big Win' goal?", placeholder="e.g. Hiking...")
            if st.form_submit_button("Complete ✨"):
                st.session_state.new_user_data["mental"] = {"mindset": mindset, "goal": goal}
                save_upgrade_profile(st.session_state.new_user_data)
                st.session_state.step = 4
                st.rerun()

    # --- Step 4: Completion ---
    elif st.session_state.step == 4:
        st.balloons()
        st.success("### Profile Locked In! 🎉")
        if st.button("Start Chatting"):
            st.session_state.step = 5 
            st.rerun()

    # --- Step 5: Chat, Ritual, & Progress ---
    elif st.session_state.step == 5:
        progress_pct, current_badge = brain.get_progress_data()
        
        # Ribbon and Badge 🎗️
        col1, col2 = st.columns([4, 1])
        with col1:
            st.progress(progress_pct / 100, text=f"Path to Mastery: {progress_pct}%")
        with col2:
            st.subheader(current_badge)

        if 'ritual_complete' not in st.session_state:
            st.info("🌅 Good morning! Ritual time.")
            energy = st.select_slider("Energy 🔋", options=[1, 2, 3, 4, 5], value=3)
            gratitude = st.text_input("One small win? ☀️")
            
            if st.button("Rise and Shine 🤍"):
                if gratitude:
                    st.audio("https://www.myinstants.com/media/sounds/naruto-main-theme-cut.mp3", autoplay=True)
                greeting = brain.perform_morning_ritual(energy, gratitude, st.session_state.new_user_data)
                st.session_state.morning_message = greeting
                st.session_state.ritual_complete = True
                st.rerun()
        else:
            st.write(f"💬 **PeRA:** {st.session_state.morning_message}")
            if "messages" not in st.session_state:
                st.session_state.messages = []
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.write(msg["content"])
            if prompt := st.chat_input("How are you feeling?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.write(prompt)
                with st.chat_message("assistant"):
                    response = brain.generate_response(prompt, st.session_state.new_user_data)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()