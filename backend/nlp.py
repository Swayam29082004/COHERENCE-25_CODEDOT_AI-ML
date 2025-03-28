import pymupdf  # PyMuPDF
import spacy
import re
import json
from transformers import pipeline

# Load Transformer-based NER model
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Load fine-tuned spaCy model for Resume Parsing
nlp = spacy.load("en_core_web_trf")  # Use transformer-based model

# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = pymupdf.open(pdf_path)  # Open PDF
    for page in doc:
        text += page.get_text("text") + "\n"  # Extract structured text
    return text.strip()

# Function to extract key details using advanced NLP
def extract_resume_details(text):
    doc = nlp(text)

    # Extract Name using NER
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    # Extract Email
    email = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email.group(0) if email else None

    # Extract Phone Number
    phone = re.search(r"\+?\d[\d\s\-\(\)]{9,}\d", text)
    phone = phone.group(0) if phone else None

    # Extract Skills using BERT-based NER
    skills = []
    ner_results = ner_pipeline(text)
    for entity in ner_results:
        if entity["entity_group"] == "MISC":  # Skills often appear in 'MISC' category
            skills.append(entity["word"])

    # Extract Education
    education = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "WORK_OF_ART"]:  # College or University
            education.append(ent.text)

    # Extract Projects
    projects = []
    project_start = text.lower().find("project work")
    if project_start != -1:
        project_text = text[project_start:]
        project_lines = project_text.split('\n')
        for line in project_lines:
            if line.strip():
                projects.append(line.strip())

    # Extract Certifications
    certifications = []
    cert_start = text.lower().find("certificates")
    if cert_start != -1:
        cert_text = text[cert_start:]
        cert_lines = cert_text.split('\n')
        for line in cert_lines:
            if line.strip():
                certifications.append(line.strip())

    return {
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Skills": list(set(skills)),  # Remove duplicates
        "Education": list(set(education)),
        "Projects": projects,
        "Certifications": certifications
    }

# Usage: Process uploaded resume
pdf_path = r"C:\Users\Admin\OneDrive\Desktop\codedot\backend\media\resume\bariankit_btech.pdf"  # Change this to your uploaded file path
resume_text = extract_text_from_pdf(pdf_path)
resume_data = extract_resume_details(resume_text)

# Save extracted data as JSON
with open("resume_data.json", "w") as f:
    json.dump(resume_data, f, indent=4)

print(resume_data)
