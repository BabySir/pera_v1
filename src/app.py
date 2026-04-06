import streamlit as st
import json
import os

# --- Helper Function: Saving the Journey ---
def save_upgrade_profile(profile_data):
    file_path = 'data/sample_patient_data.json'
    os.makedirs('data', exist_ok=True)
    
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

# --- App Setup ---
st.set_page_config(page_title="PeRA - Your Life Upgrade", page_icon="🌱", layout="wide")

# Initialize session state for navigation
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'new_user_data' not in st.session_state:
    st.session_state.new_user_data = {}

# --- Step 1: Checking In With Your Body ---
if st.session_state.step == 1:
    st.header("🌱 Step 1: Checking In With Your Body")
    st.markdown("#### *Let's listen to what your body is telling us today.*")
    st.progress(0.25, text="Beginning our journey...")

    with st.popover("✨ A little note for you"):
        st.write("Think of me as a trusted friend. 🤍 There are no 'wrong' answers here.")

    with st.form("step1_form"):
        col1, col2 = st.columns(2)
        with col1:
            focus = st.selectbox("Where shall we focus our care today?", ["Knee", "Lower Back", "Shoulder", "Ankle", "General Wellness"])
            comfort = st.select_slider("How does that area feel right now?", 
                options=["Quite Uncomfortable ⛈️", "A Bit Heavy ☁️", "Doing Okay 🌤️", "Feeling Good 🌤️", "Wonderful & Light ☀️"],
                value="Doing Okay 🌤️")
        with col2:
            obs = st.multiselect("Notice any 'little signals' lately?", ["Morning Stiffness", "Clicking/Popping", "Feeling Unsteady", "Sharpness", "Tiring Easily"])
        
        story = st.text_area("Your Story", placeholder="Tell me about how you've been moving lately... ☕")
        st.info("🔒 **Privacy Shield Active:** Your stories stay safe and local on this device.")

        if st.form_submit_button("Share this with me 🤍"):
            st.session_state.new_user_data["physical"] = {"focus": focus, "comfort": comfort, "obs": obs, "story": story}
            st.session_state.step = 2
            st.rerun()

# --- Step 2: Your Daily Rhythm & Rituals ---
elif st.session_state.step == 2:
    st.header("🕰️ Step 2: Your Daily Rhythm & Rituals")
    st.markdown("#### *Let's find the hidden opportunities in your day.*")
    st.progress(0.50, text="Walking further together...")

    with st.form("step2_form"):
        col1, col2 = st.columns(2)
        with col1:
            work = st.selectbox("How do you spend your working hours?", ["Mostly sitting", "Mixed", "On my feet", "Active/Physical"])
            breaks = st.select_slider("How often do you stretch or move?", ["Rarely", "Occasionally", "Every few hours", "Whenever I can"])
        with col2:
            sleep = st.select_slider("How is your morning energy?", ["Exhausted", "A bit tired", "Okay", "Refreshed", "Full of energy!"])
            habits = st.multiselect("Any 'hidden' energy drains?", ["Late night screens", "Extra caffeine", "Skipping meals", "Stress snacking", "Not enough water"])
        
        st.info("🔒 **Privacy Shield:** Sharing the truth is the first step toward your upgrade.")
        
        c1, c2 = st.columns(2)
        if c1.form_submit_button("Go Back"):
            st.session_state.step = 1
            st.rerun()
        if c2.form_submit_button("Next Step 🤍"):
            st.session_state.new_user_data["lifestyle"] = {"work": work, "breaks": breaks, "sleep": sleep, "habits": habits}
            st.session_state.step = 3
            st.rerun()

# --- Step 3: The Mind-Body Connection ---
elif st.session_state.step == 3:
    st.header("🧠 Step 3: The Mind-Body Connection")
    st.markdown("#### *Our thoughts are the invisible architects of our healing.*")
    st.progress(0.75, text="Almost there...")

    with st.form("step3_form"):
        st.write("**Your Inner Dialogue** 🗣️")
        mindset = st.radio("When your body feels 'stuck', what do you usually tell yourself?", 
            ["'I'm worried I'll never get better'", 
             "'This is just a temporary hurdle on my path'", 
             "'I'm frustrated, but I know I can find a way through'"],
             help="Be honest—your 'inner voice' is a key part of your recovery journey.")
        
        st.write("**The Power of Perspective** 🌈")
        future_goal = st.text_input("What is one 'big win' you're looking forward to?", 
                                    placeholder="e.g., Hiking with my dog, dancing at a wedding, playing with my kids...")
        
        st.info("🔒 **Privacy Shield:** Your feelings are safe, valid, and private here.")

        c1, c2 = st.columns(2)
        if c1.form_submit_button("Go Back"):
            st.session_state.step = 2
            st.rerun()
        if c2.form_submit_button("Complete My Upgrade ✨"):
            st.session_state.new_user_data["mental"] = {"mindset": mindset, "goal": future_goal}
            save_upgrade_profile(st.session_state.new_user_data)
            st.session_state.step = 4
            st.rerun()

# --- Step 4: Completion State ---
elif st.session_state.step == 4:
    st.balloons()
    st.success("### Your Life Upgrade Profile is Locked In! 🎉")
    st.markdown("""
        I've gathered your story, your rhythm, and your goals. 
        I'm so excited to start this journey with you.
    """)
    if st.button("Start Chatting with PeRA"):
        st.session_state.step = 1 # In a full app, this would switch to the chat UI
        st.rerun()