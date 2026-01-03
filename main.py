import os
import json
from flask import Flask, request, jsonify, render_template
from google import genai
from pydantic import BaseModel
import PyPDF2

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Replace with your actual API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Updated Schema to include Recommendations and ATS Score
class ResumeAnalysis(BaseModel):
    ats_score: int
    top_skills: list[str]
    strengths: list[str]
    improvements: list[str]
    job_recommendations: list[str]

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"PDF Error: {e}")
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No resume file uploaded"}), 400
        
        resume_file = request.files["resume"]
        path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
        resume_file.save(path)
        
        resume_text = extract_text_from_pdf(path)
        
        if not resume_text.strip():
            return jsonify({"error": "PDF is empty or unreadable"}), 400

        # Optimized prompt for career pathing
        prompt = f"""
        Act as a Senior Career Consultant. Analyze the following resume:
        1. Calculate an ATS score (0-100).
        2. Identify key strengths and specific areas for improvement.
        3. Recommend 3-5 specific job titles that align perfectly with these strengths.

        RESUME:
        {resume_text}
        """

        # Corrected model string for the 'google-genai' SDK
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': ResumeAnalysis,
            }
        )
        
        return response.text 

    except Exception as e:
        # This will help you see the EXACT error in your terminal
        print(f"CRITICAL ERROR: {str(e)}")
        return jsonify({"error": "AI Service Error: " + str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True, port=8080)