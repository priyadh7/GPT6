from flask import Flask, render_template, request, send_file
from pathlib import Path
from datetime import datetime
import requests
import json
import io

app = Flask(__name__)

MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def as_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def analyze_symptoms(patient):
    prompt = f"""
You are an educational health-information assistant, not a doctor.
Use cautious, non-diagnostic language. Never claim certainty or prescribe
medication. Do not invent patient details. Explain that symptoms can have
multiple causes and an in-person clinician is needed for diagnosis.

Patient information:
{json.dumps(patient, ensure_ascii=False)}

Return valid JSON with exactly these keys:
extracted_symptoms: array of strings
possible_conditions: array of strings
why_considered: string
follow_up_questions: array of strings
general_self_care: array of strings
medical_attention: string
summary: string

Give a small number of broad possible explanations, not a definitive diagnosis.
If information is insufficient, say so and ask relevant follow-up questions.
Do not provide medication dosages or suggest starting/stopping prescriptions.
For severe or potentially life-threatening symptoms, prioritize urgent medical care.
Keep the response clear and suitable for a general audience.
Respond with JSON only.
"""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2}
        },
        timeout=240
    )
    response.raise_for_status()
    data = json.loads(response.json()["response"])

    if not isinstance(data, dict):
        raise ValueError("Unexpected AI response. Please try again.")

    for key in (
        "extracted_symptoms", "possible_conditions",
        "follow_up_questions", "general_self_care"
    ):
        data[key] = as_list(data.get(key))

    return data


@app.route("/", methods=["GET", "POST"])
def home():
    patient = {
        "age": "",
        "sex": "",
        "medical_history": "",
        "medication": "",
        "allergies": "",
        "symptoms": "",
        "duration": "",
        "severity": "",
        "additional_info": ""
    }
    report = None
    error = ""

    if request.method == "POST":
        for key in patient:
            patient[key] = request.form.get(key, "").strip()

        if not patient["age"].isdigit() or not 0 <= int(patient["age"]) <= 120:
            error = "Please enter a valid age between 0 and 120."
        elif not patient["symptoms"]:
            error = "Please describe at least one symptom."
        elif not patient["duration"]:
            error = "Please select the symptom duration."
        elif not patient["severity"]:
            error = "Please select the severity."
        else:
            try:
                report = analyze_symptoms(patient)
                report["patient"] = patient
                report["generated_at"] = datetime.now().strftime(
                    "%d %b %Y, %I:%M %p"
                )
            except requests.RequestException:
                error = (
                    "Cannot connect to Ollama. Make sure Ollama is running "
                    "and the llama3.2 model is installed."
                )
            except (ValueError, json.JSONDecodeError, KeyError, TypeError):
                error = "The AI response could not be processed. Please try again."

    return render_template(
        "index.html", patient=patient, report=report, error=error
    )


@app.route("/download", methods=["POST"])
def download():
    try:
        data = json.loads(request.form.get("report_json", ""))
    except json.JSONDecodeError:
        return "Invalid report.", 400

    patient = data.get("patient", {})
    sections = [
        ("PATIENT CONTEXT", [
            f"Age: {patient.get('age', 'Not provided')}",
            f"Sex: {patient.get('sex') or 'Not provided'}",
            f"Symptoms: {patient.get('symptoms', '')}",
            f"Duration: {patient.get('duration', '')}",
            f"Severity: {patient.get('severity', '')}",
            f"Medical history: {patient.get('medical_history') or 'Not provided'}",
            f"Medication: {patient.get('medication') or 'Not provided'}",
            f"Allergies: {patient.get('allergies') or 'Not provided'}",
        ]),
        ("EXTRACTED SYMPTOMS", data.get("extracted_symptoms", [])),
        ("POSSIBLE EXPLANATIONS - NOT A DIAGNOSIS",
         data.get("possible_conditions", [])),
        ("WHY THESE WERE CONSIDERED", data.get("why_considered", "")),
        ("FOLLOW-UP QUESTIONS", data.get("follow_up_questions", [])),
        ("GENERAL SELF-CARE", data.get("general_self_care", [])),
        ("MEDICAL ATTENTION GUIDANCE", data.get("medical_attention", "")),
        ("SUMMARY", data.get("summary", "")),
    ]

    lines = [
        "AI-POWERED MEDICAL SYMPTOM ASSISTANT",
        "=" * 42,
        f"Generated: {data.get('generated_at', '')}",
        "",
        "IMPORTANT: Educational information only. Not a medical diagnosis.",
        "Do not delay emergency care to use this report.",
        ""
    ]

    for title, content in sections:
        lines.extend([title, "-" * len(title)])
        if isinstance(content, list):
            lines.extend([f"- {item}" for item in content] or ["- Not specified"])
        else:
            lines.append(str(content or "Not specified"))
        lines.append("")

    output = io.BytesIO("\n".join(lines).encode("utf-8"))
    output.seek(0)

    return send_file(
        output,
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name="medical_symptom_report.txt"
    )


if __name__ == "__main__":
    app.run(debug=True)
