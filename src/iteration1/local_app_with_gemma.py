import ollama
import json

# --- CONFIGURATION ---
MODEL_NAME = "gemma2:9b"  # Ensure you have this pulled
RAW_DATA_FILE = "patient101.json"

# --- PHASE 1: THE "TIME-AWARE" EXTRACTOR (With Streaming) ---
def extract_clinical_timeline(raw_json_str):
    print("⚙️  Extracting Clinical Timeline... (Watch it happen below)")
    print("-------------------------------------------------------")
    
    prompt = f"""
    ACT AS A CLINICAL DATA SPECIALIST.
    
    TASK:
    1. Read the raw patient data below.
    2. Extract ONLY clinical text entities (Diagnosis, Symptoms, Medications, Lab Results).
    3. IGNORE all image links, URLs, and file paths.
    4. CRITICAL: Keep the 'date' for EVERY entry.
    5. Map each entity to its SNOMED CT Term and Concept ID (CUI).
    
    RAW DATA:
    {raw_json_str}
    
    OUTPUT FORMAT (Strict JSON List):
    [
      {{
        "date": "DD-Mon-YYYY", 
        "type": "Diagnosis/Medication/Lab",
        "text": "...",
        "snomed_id": "...", 
        "snomed_term": "..."
      }}
    ]
    """
    
    # 1. ENABLE STREAMING
    stream = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}], stream=True)
    
    full_response = ""
    
    # 2. PRINT CHUNKS AS THEY ARRIVE
    for chunk in stream:
        part = chunk['message']['content']
        print(part, end="", flush=True)  # <--- This is the magic line
        full_response += part
        
    print("\n-------------------------------------------------------")
    return full_response

# --- PHASE 2: THE CHATBOT (Reasoning on the Timeline) ---
def start_chat(timeline_context):
    print("\n✅ Clinical Timeline Loaded. Chat is Ready.")
    print("------------------------------------------------")
    
    messages = [
        {
            'role': 'system',
            'content': f"""
            You are 'Dr. Knows', a clinically aware AI assistant.
            
            PATIENT CLINICAL TIMELINE (SNOMED-CODED):
            {timeline_context}
            
            INSTRUCTIONS:
            1. Answer based ONLY on the timeline above.
            2. CHECK DATES: Distinguish between "Current" conditions (recent dates) and "History" (old dates).
            3. Use the SNOMED IDs to confirm specific conditions (e.g., matching 'Diabetes' to ID 73211009).
            """
        }
    ]

    while True:
        user_input = input("\nPatient (You): ")
        if user_input.lower() in ["exit", "quit"]: break
        
        messages.append({'role': 'user', 'content': user_input})
        
        print("Dr. Knows: ", end="", flush=True)
        stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
        
        full_response = ""
        for chunk in stream:
            part = chunk['message']['content']
            print(part, end="", flush=True)
            full_response += part
            
        messages.append({'role': 'assistant', 'content': full_response})
        print("\n")

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Load Raw File
    try:
        with open(RAW_DATA_FILE, "r") as f:
            raw_data = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Could not find {RAW_DATA_FILE}. Make sure it's in this folder.")
        exit()

    # 2. Extract (The "Thing")
    timeline_json = extract_clinical_timeline(raw_data)
    
    # Optional: Save it so you can show the client the "Clean" data
    with open("patient_timeline_clean.json", "w") as f:
        f.write(timeline_json)
    print("   (Saved clean timeline to 'patient_timeline_clean.json')")

    # 3. Chat
    start_chat(timeline_json)