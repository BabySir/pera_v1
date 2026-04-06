# src/app.py
import streamlit as st
from src.nano_llm import NanoLLM
from src.personalization import PatientDataManager

#st.set_page_config(page_title="PeRA-Your Friend", layout="wide")

@st.cache_resource
def load_models():
    # Removed XAIExplainerLIME and SHAP
    return NanoLLM(), PatientDataManager()

def main():
    st.title("🤖 Personalized e-Rehabilitation Assistant")
    st.markdown("---")
    
    llm, patient_mgr = load_models()
    
    with st.sidebar:
        st.header("Patient Profile")
        patient_id = st.selectbox("Select Patient", ["P001"], index=0)
        patient_context = patient_mgr.get_patient_context(patient_id)
        with st.expander("View Profile"):
            st.write(patient_context)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "explanation" in message:
                with st.expander("Why this recommendation?"):
                    st.markdown(message["explanation"])
    
    if prompt := st.chat_input("How are you feeling today? What's hurting?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your health data..."):
                # Unpack the 3 variables returned by our new generate_response function
                response, explanation, context = llm.generate_response(prompt, patient_id)
                
                st.markdown(response)
                with st.expander("🔍 Explanation & Sources"):
                    st.markdown(explanation)
                    st.markdown("**Retrieved Context:**")
                    for i, doc in enumerate(context[:2], 1):
                        st.markdown(f"**{i}.** {doc[:300]}...")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "explanation": explanation
            })

if __name__ == "__main__":
    main()