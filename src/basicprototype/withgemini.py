import requests
import json
import os
import sys

# --- 1. CONFIGURATION ---
# ⚠️ PASTE YOUR REAL KEY HERE (The one starting with AIza...)
API_KEY = "AIzaSyARKSIvADELLSqq2eGqVOPi6Afpb-f-N_8"
DATA_FILE = "patient101.json"  # Ensure this matches your file name

if "AIza" not in API_KEY:
    print("❌ ERROR: It looks like you didn't paste your API key correctly.")
    sys.exit()

# --- 2. AUTOMATIC MODEL FINDER ---
def get_best_model():
    print("🔍 Auto-detecting available Gemini models...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # We prefer Flash (fast/cheap), then Pro, then anything else that works.
        preferred_order = ["flash", "pro", "gemini"]
        
        valid_models = []
        for model in data.get('models', []):
            # Must support 'generateContent'
            if "generateContent" in model.get("supportedGenerationMethods", []):
                valid_models.append(model['name'])

        # Find the best match
        for preference in preferred_order:
            for model_name in valid_models:
                if preference in model_name and "vision" not in model_name:
                    print(f"✅ Found working model: {model_name}")
                    return model_name
        
        # Fallback to the first one found
        if valid_models:
            print(f"⚠️ specific 'Flash' model not found. Using fallback: {valid_models[0]}")
            return valid_models[0]
            
    except Exception as e:
        print(f"❌ Could not list models. Error: {e}")
        return None
    
    print("❌ No valid text-generation models found for this API key.")
    return None

# # --- 3. THE CALLER ---
# def call_gemini(model_name, prompt):
#     # Construct URL dynamically based on the found model
#     # model_name comes in like "models/gemini-1.5-flash-001"
#     url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={API_KEY}"
    
#     headers = {'Content-Type': 'application/json'}
#     payload = { "contents": [{ "parts": [{"text": prompt}] }] }
    
#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         return response.json()['candidates'][0]['content']['parts'][0]['text']
#     except Exception as e:
#         print(f"❌ API Call Failed: {e}")
#         return None
# --- 3. THE CALLER (Updated to show Tokens) ---
def call_gemini(model_name, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={API_KEY}"
    
    headers = {'Content-Type': 'application/json'}
    payload = { "contents": [{ "parts": [{"text": prompt}] }] }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # 1. EXTRACT TOKEN COUNTS
        usage = result.get('usageMetadata', {})
        input_tokens = usage.get('promptTokenCount', 0)
        output_tokens = usage.get('candidatesTokenCount', 0)
        total_tokens = usage.get('totalTokenCount', 0)
        
        # 2. PRINT THEM (So you can see the cost)
        print(f"\n   [📊 TOKENS: Input={input_tokens} + Output={output_tokens} = {total_tokens} Total]")
        
        # 3. RETURN THE ANSWER
        return result['candidates'][0]['content']['parts'][0]['text']
        
    except Exception as e:
        print(f"❌ API Call Failed: {e}")
        return None

# --- 4. EXTRACTOR ---
def extract_data(model_name, raw_text):
    print("⚙️  Extracting Clinical Data...")
    prompt = f"""
    Act as a Clinical Coder.
    TASK: Extract Diagnosis, Meds, Labs, Symptoms from the text below.
    CRITICAL: Keep Dates. Map to SNOMED IDs.
    RAW DATA: {raw_text}
    OUTPUT: Valid JSON List.
    """
    res = call_gemini(model_name, prompt)
    if res:
        return res.replace("```json", "").replace("```", "").strip()
    return "[]"

# --- 5. CHATBOT ---
def start_chat(model_name, context):
    print("\n✅ System Ready. Type 'exit' to quit.")
    chat_history = f"SYSTEM: You are Dr. Knows. Context: {context}\n"
    
    while True:
        user_input = input("\nPatient (You): ")
        if user_input.lower() in ["exit", "quit"]: break
        
        print("Dr. Knows: ", end="", flush=True)
        full_prompt = f"{chat_history}\nUSER: {user_input}\nASSISTANT:"
        
        response = call_gemini(model_name, full_prompt)
        if response:
            print(response)
            chat_history += f"\nUSER: {user_input}\nASSISTANT: {response}"

# --- MAIN ---
if __name__ == "__main__":
    # 1. Load File
    try:
        with open(DATA_FILE, "r") as f: raw_data = f.read()
    except:
        print(f"❌ File {DATA_FILE} not found."); sys.exit()

    # 2. Find Model
    best_model = get_best_model()
    if not best_model: sys.exit()

    # 3. Run
    clean_data = extract_data(best_model, raw_data)
    if clean_data != "[]":
        start_chat(best_model, clean_data)