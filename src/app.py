import streamlit as st
import json
import os
from datetime import datetime
from src.engine import PeraBrain
from src.nano_llm import NanoLLM

# --- DYNAMIC SAVE & PAIN LOG LOGIC ---
def save_upgrade_profile(new_data):
    file_path = 'data/sample_patient_data.json'
    os.makedirs('data', exist_ok=True)
    
    all_profiles = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                all_profiles = json.load(f)
            except json.JSONDecodeError:
                all_profiles = []
                
    user_found = False
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    for i, profile in enumerate(all_profiles):
        if profile.get('patient_id') == new_data.get('patient_id'):
            # 1. Update root level
            all_profiles[i].update(new_data)
            
            # 2. Push to pain_log array
            if "pain_log" not in all_profiles[i]:
                all_profiles[i]["pain_log"] = []
                
            phys = new_data.get("physical", {})
            life = new_data.get("lifestyle", {})
            ment = new_data.get("mental", {})
            
            log_entry = {
                "date": today_date,
                "focus_area": phys.get("focus", ""),
                "comfort_level": phys.get("comfort", ""),
                "mood": ment.get("mindset", ""),
                "activity": f"Work: {life.get('work', '')} | Sleep: {life.get('sleep', '')}",
                "notes": phys.get("story", "")
            }
            
            logged_today = False
            for j, log in enumerate(all_profiles[i]["pain_log"]):
                if log.get("date") == today_date:
                    clean_entry = {k: v for k, v in log_entry.items() if v}
                    all_profiles[i]["pain_log"][j].update(clean_entry)
                    logged_today = True
                    break
                    
            if not logged_today:
                if phys or life or ment:
                    all_profiles[i]["pain_log"].append(log_entry)
            
            user_found = True
            break
            
    if not user_found:
        all_profiles.append(new_data)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_profiles, f, indent=4)

@st.cache_resource
def load_pera_brain():
    my_model = NanoLLM()
    return PeraBrain(my_model)

def main():
    st.set_page_config(page_title="PeRA - Your Life Upgrade", page_icon="🌱", layout="wide")

    # --- SET ACTIVE USER ---
    ACTIVE_USER_ID = "P-IND-25-AM"

    if 'step' not in st.session_state:
        st.session_state.step = 1
        
    if 'new_user_data' not in st.session_state:
        st.session_state.new_user_data = {"patient_id": ACTIVE_USER_ID}

    brain = load_pera_brain()

    st.title("PeRA 🌱: Personalized e-Rehab Assistant")

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.markdown("### 🌱 Journey Controls")
        if st.button("Start New Life 🌅"):
            # Clear memory to trigger the Ritual again
            st.session_state.messages = []
            st.session_state.step = 4 # Jump straight to ritual if they are already onboarded
            st.rerun()

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
            
            if "Addiction" in focus:
                comfort = st.select_slider("Craving intensity today?", options=["Low", "Manageable", "Strong", "Severe ⛈️"])
            elif "Pain" in focus or "Injury" in focus or "Post-Operative" in focus:
                comfort = st.select_slider("Pain/Discomfort level (1-10)?", options=[str(i) for i in range(1, 11)])
            else:
                comfort = st.select_slider("How are you feeling?", options=["1 ⛈️", "2 ☁️", "3 🌤️", "4 ☀️"])
                
            story = st.text_area("Tell me more details...")
            
            if st.form_submit_button("Next 🤍"):
                is_safe, flag_msg = brain.safety.screen_input(story)
                
                if not is_safe:
                    st.error(flag_msg) # Block progression and show warning
                else:
                    st.session_state.new_user_data["physical"] = {"focus": focus, "comfort": comfort, "story": story}
                    save_upgrade_profile(st.session_state.new_user_data)
                    st.session_state.step = 2
                    st.rerun()

    # --- Step 2: Lifestyle Check ---
    elif st.session_state.step == 2:
        st.header("🕰️ Step 2: Your Day")
        with st.form("step2_form"):
            work = st.selectbox("Current Work/Activity Level:", ["Sedentary/Desk", "Light Activity", "Heavy Manual Labor"])
            sleep = st.select_slider("Sleep Quality:", options=["Poor", "Fair", "Good", "Excellent"])
            
            if st.form_submit_button("Next 🤍"):
                st.session_state.new_user_data["lifestyle"] = {"work": work, "sleep": sleep}
                save_upgrade_profile(st.session_state.new_user_data)
                st.session_state.step = 3
                st.rerun()

    # --- Step 3: Mindset & Goal ---
    elif st.session_state.step == 3:
        st.header("🏔️ Step 3: The North Star")
        with st.form("step3_form"):
            mindset = st.text_input("Current state of mind:")
            goal = st.text_input("What is your North Star Goal?")
            
            if st.form_submit_button("Complete Setup ✨"):
                # 🛡️ NEW: Screen both inputs together
                is_safe, flag_msg = brain.safety.screen_input(mindset + " " + goal)
                
                if not is_safe:
                    st.error(flag_msg)
                else:
                    st.session_state.new_user_data["mental"] = {"mindset": mindset, "goal": goal}
                    save_upgrade_profile(st.session_state.new_user_data)
                    st.session_state.step = 4
                    st.rerun()

    # --- Step 4: Ritual & Continuous Chat ---
    elif st.session_state.step == 4:
        st.header("🌅 Daily Ritual & Chat")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Part A: The Kickoff Form
        if len(st.session_state.messages) == 0:
            with st.container():
                st.markdown("### Start your session")
                energy = st.slider("Energy Level today:", 1, 5, 3)
                gratitude = st.text_input("One small win or thing you're grateful for:")
                
                if st.button("Begin Ritual ✨"):
                    is_safe, flag_msg = brain.safety.screen_input(gratitude)
                    if not is_safe:
                        st.error(flag_msg)
                    else:
                        with st.spinner("PeRA is gathering her thoughts..."):
                            # Unpack both the greeting AND the real explanation
                            greeting, explanation = brain.perform_ritual(energy, gratitude, st.session_state.new_user_data)
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": greeting,
                                "explanation": explanation  # Use the dynamically generated explanation here!
                            })
                    st.rerun()

        # Part B: The Continuous Chat Interface
        else:
            progress, badge = brain.get_progress_data()
            st.info(f"🏆 Current Rank: {badge} (Progress: {progress}%)")
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("explanation"):
                        with st.expander("🧠 Why did PeRA suggest this?"):
                            st.markdown(msg["explanation"])

            if user_input := st.chat_input("How can I help you today?", max_chars=500):
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your profile..."):
                        response, explanation = brain.generate_response(
                            user_input, 
                            st.session_state.new_user_data, 
                            st.session_state.messages
                        )
                        st.markdown(response)
                        if explanation:
                            with st.expander("🧠 Why did PeRA suggest this?"):
                                st.markdown(explanation)
                                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "explanation": explanation
                })

if __name__ == "__main__":
    main()