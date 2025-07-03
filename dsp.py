import dspy
from dspy import InputField, OutputField, Signature
from typing import List, Dict
import json
import pandas as pd
import random
import openai
from openai import OpenAI
import os

openai.api_key = os.getenv("OPENAI_API_KEY", "")  

AGENT_NAME = "Priya"
GENDER = "female"


def build_entity_signature(entity_list):
    """Create a dynamic Signature class from user-defined entity list."""
    fields = {
        'transcript': InputField(desc="Complete conversation transcript between agent and lead")
    }
    for entity in entity_list:
        name = entity["name"]
        desc = entity["description"]
        fields[name] = OutputField(desc=desc)
    return type("DynamicEntityExtraction", (Signature,), fields)


class DynamicEntityExtractionModule(dspy.Module):
    def __init__(self, entity_signature_class):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(entity_signature_class)

    def forward(self, transcript):
        entities = self.extract_entities(transcript=transcript)

        answers = []
        for key, value in vars(entities).items():
            if key == "transcript":
                continue
            if value and str(value).strip():
                answers.append({
                    "report_key": key,
                    "answer_value": str(value).strip()
                })

        return {
            "data": {
                "outcome": "Entity Extraction Only",  # Optional, if no classification logic used
                "answers": answers
            }
        }


def load_prompt(prompt_text: str) -> str:
    return prompt_text.strip() or "You are a helpful voicebot agent."


def generate_conversation_prompt(scenarios: List[str], main_prompt: str) -> str:
    scenario_text = "\n".join(f"- {s}" for s in scenarios)
    return f"""You are a {GENDER} voicebot agent named {AGENT_NAME}.

Refer to the behavioral instructions below for how to act and speak:
{main_prompt.strip()}

Instructions:
- The conversation should feel REAL, not perfect or scripted. Users may interrupt, ask off-topic questions, or speak in slang.
- Simulate realistic behavior like repetition, silence, partial understanding, or requesting a callback.
- Use a **mix of Hindi and English** based on the user's tone and input. Start in Hindi-English blend, unless the user speaks only English.
- End the call naturally -- could be resolution, a callback, or user dropping off.
- Conversation must follow a **dialogue format** with alternating 'Agent:' and 'Lead:' lines.

Scenarios to incorporate:
{scenario_text}

Generate a complete conversational transcript in alternating lines of Agent and Lead that incorporates these scenarios naturally."""


def call_openai(prompt: str, model="gpt-4", temperature=0.8, max_tokens=1500) -> str:
    try:
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return "Error generating response."


def setup_dspy():
    lm = dspy.LM("openai/gpt-4", api_key=openai.api_key, drop_params=True)
    dspy.configure(lm=lm)


def run_enhanced_simulation(main_prompt: str, scenarios: List[str], entities: List[Dict]) -> (str, str):
    setup_dspy()
    SignatureClass = build_entity_signature(entities)
    entity_extractor = DynamicEntityExtractionModule(SignatureClass)

    records = []
    for i, scenario in enumerate(scenarios):
        print(f"Generating sample {i + 1}/{len(scenarios)} for scenario: {scenario}")
        selected_scenarios = [scenario]
        conversation_prompt = generate_conversation_prompt(selected_scenarios, main_prompt)
        call_transcript = call_openai(conversation_prompt)

        try:
            entity_result = entity_extractor(call_transcript)
            records.append({
                "Scenario Summary": scenario,
                "Call Transcript": call_transcript,
                "Entity Extraction Output": json.dumps(entity_result, indent=2),
                "Structured Output": entity_result
            })
            print(f"Sample {i + 1} completed successfully")
        except Exception as e:
            print(f"Error processing sample {i + 1}: {e}")
            records.append({
                "Scenario Summary": scenario,
                "Call Transcript": call_transcript,
                "Entity Extraction Output": f"Error: {e}",
                "Structured Output": {}
            })

    df = pd.DataFrame(records)
    csv_file = "voicebot_output_enhanced.csv"
    df.to_csv(csv_file, index=False)

    json_file = csv_file.replace(".csv", ".json")
    json_output = [
        {
            "scenarios": r["Scenario Summary"],
            "transcript": r["Call Transcript"],
            "entities": r["Structured Output"]
        }
        for r in records
    ]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    return csv_file, json_file
