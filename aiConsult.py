import google.generativeai as genai
from streamlit import secrets

api_key = secrets.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.0-flash-exp",generation_config={"response_mime_type": "application/json"})

def gemini_pro_vision_api(image, prompt):
    # Simulate a response from the API
    if prompt != "":
        response = model.generate_content([prompt, image])
    else:
        response = model.generate_content(image)

    return response.text
