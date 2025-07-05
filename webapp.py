import streamlit as st
import json
import pandas as pd
from dsp import run_enhanced_simulation

# Output filenames
OUTPUT_JSON = "voicebot_output_enhanced.json"
OUTPUT_CSV = "voicebot_output_enhanced.csv"

# Session State Initialization
if 'scenarios' not in st.session_state:
    st.session_state['scenarios'] = []
if 'entities' not in st.session_state:
    st.session_state['entities'] = []
if 'simulation_ran' not in st.session_state:
    st.session_state['simulation_ran'] = False

st.title("Voicebot Simulation UI + DSPy Pipeline")

# Section 1: Main Prompt Upload or Text
st.header(" Main Prompt")
main_prompt_input = st.text_area(" Write Main Prompt or Upload Below", height=200)
main_prompt_file = st.file_uploader("Upload Main Prompt File", type=["txt"])

if main_prompt_file:
    main_prompt_input = main_prompt_file.read().decode("utf-8")

# Section 2: Scenarios
st.header("Scenarios")
new_scenario = st.text_input("➕ Add Scenario")
if st.button("Add Scenario"):
    if new_scenario:
        st.session_state['scenarios'].append(new_scenario)

if st.session_state['scenarios']:
    st.markdown("### Current Scenarios")
    for idx, scenario in enumerate(st.session_state['scenarios']):
        st.write(f"{idx + 1}. {scenario}")

# Section 3: Entities
st.header("Entities")
col1, col2, col3 = st.columns(3)
with col1:
    entity_name = st.text_input("Entity Name")
with col2:
    entity_type = st.text_input("Entity Type")
with col3:
    entity_desc = st.text_input("Entity Description")

if st.button("Add Entity"):
    if entity_name and entity_type and entity_desc:
        st.session_state['entities'].append({
            "name": entity_name,
            "type": entity_type,
            "description": entity_desc
        })

if st.session_state['entities']:
    st.markdown("### Current Entities")
    entity_df = pd.DataFrame(st.session_state['entities'])
    st.dataframe(entity_df)

# Save Input & Trigger Simulation
if st.button("Submit and Run Simulation"):
    # Save input JSON (optional for audit)
    input_json = {
        "main_prompt": main_prompt_input,
        "scenarios": st.session_state['scenarios'],
        "entities": st.session_state['entities']
    }
    with open("voicebot_input_structure.json", "w", encoding='utf-8') as f:
        json.dump(input_json, f, indent=2)

    st.write("Running DSPy Simulation... Please wait ⏳")

    
    try:
        csv_path, json_path = run_enhanced_simulation(
            main_prompt=main_prompt_input,
            scenarios=st.session_state['scenarios'],
            entities=st.session_state['entities']
        )

        st.session_state['simulation_ran'] = True
        st.session_state['csv_output'] = csv_path
        st.session_state['json_output'] = json_path
        st.success("Simulation completed!")

    except Exception as e:
        st.error(f"Simulation failed: {e}")

# Show download buttons
if st.session_state.get('simulation_ran'):
    st.header("📥 Download Simulation Output")
    with open(st.session_state['csv_output'], "rb") as f:
        st.download_button("Download CSV", f, file_name="voicebot_output_enhanced.csv", mime="text/csv")
    with open(st.session_state['json_output'], "rb") as f:
        st.download_button("Download JSON", f, file_name="voicebot_output_enhanced.json", mime="application/json")
