# 8. src/app.py - Streamlit UI
"""
Production-ready Streamlit app integrating all components
Deployable to Google Cloud Run / Streamlit Cloud
"""
import streamlit as st
from src.nano_llm import NanoLLM
from src.xai_explainer_lime import XAIExplainerLIME
from src.xai_explainer_shap import XAIExplainerSHAP
from src.personalization import PatientDataManager

st.set_page_config(page_title="PeRA-Your Friend", layout="wide")

@st.cache_resource
def load_models():
    return NanoLLM(), XAIExplainerLIME(), PatientDataManager()

def main():
    st.title("🤖 Personalized e-Rehabilitation Assistant")
    st.markdown("---")
    
    # Load components
    llm, xai, patient_mgr = load_models()
    
    # Sidebar - Patient selection
    with st.sidebar:
        st.header("Patient Profile")
        patient_id = st.selectbox("Select Patient", ["P001"], index=0)
        patient_context = patient_mgr.get_patient_context(patient_id)
        with st.expander("View Profile"):
            st.write(patient_context)
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "explanation" in message:
                with st.expander("Why this recommendation?"):
                    st.markdown(message["explanation"])
    
    # Chat input
    if prompt := st.chat_input("How are you feeling today? What's hurting?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your health data..."):
                # Generate response
                response = llm.generate_response(prompt, patient_id)
                context = llm.rag.retrieve(prompt, patient_id)
                
                # Generate explanation
                explanation = xai.explain_response(prompt, response, context)
                
                # Display response
                st.markdown(response)
                with st.expander("🔍 Explanation & Sources"):
                    st.markdown(explanation)
                    st.markdown("**Retrieved Context:**")
                    for i, doc in enumerate(context[:2], 1):
                        st.markdown(f"**{i}.** {doc[:300]}...")
            
            # Store full message with explanation
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "explanation": explanation
            })

if __name__ == "__main__":
    main()
