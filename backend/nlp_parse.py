import fitz  # PyMuPDF
import spacy
import re
import psycopg2
from django.conf import settings

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Connect to PostgreSQL using Django settings
conn = psycopg2.connect(
    dbname=settings.DATABASES['default']['NAME'],
    user=settings.DATABASES['default']['USER'],
    password=settings.DATABASES['default']['PASSWORD'],
    host=settings.DATABASES['default']['HOST'],
    port=settings.DATABASES['default']['PORT']
)
cursor = conn.cursor()

def extract_text_from_pdf(pdf_path):
    """Extract text from a resume PDF"""
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return text

def extract_details(text):
    """Extract structured details using NLP"""
    doc = nlp(text)
    
    # Extract Name
    name = doc.ents[0].text if doc.ents else "Unknown"
    
    # Extract Email
    email_match = re.search(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text)
    email = email_match.group() if email_match else "Not Found"
    
    # Extract Phone
    phone_match = re.search(r"\+?\d{10,13}", text)
    phone = phone_match.group() if phone_match else "Not Found"
    
    # Extract Skills (Keyword matching)
    predefined_skills = ["Python", "Java", "C++", "Machine Learning", "AI", "Django", "Flask", "React"]
    skills = [skill for skill in predefined_skills if skill.lower() in text.lower()]
    
    # Extract Experience (Years of Experience)
    experience_match = re.search(r"(\d+)\s*(years|yrs|year|yr)\s*experience", text, re.IGNORECASE)
    experience = int(experience_match.group(1)) if experience_match else 0
    
    # Extract Education (Keyword-based approach)
    education_keywords = ["B.Tech", "M.Tech", "MBA", "PhD", "B.Sc", "M.Sc"]
    education = next((word for word in education_keywords if word in text), "Not Found")

    return name, email, phone, skills, experience, education

def store_in_db(name, email, phone, skills, experience, education):
    """Insert extracted details into PostgreSQL"""
    query = """
    INSERT INTO resumes (name, email, phone, skills, experience, education) 
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (name, email, phone, skills, experience, education))
    conn.commit()

def process_resume(pdf_path):
    """Extract details and store in DB"""
    resume_text = extract_text_from_pdf(pdf_path)
    parsed_data = extract_details(resume_text)
    
    # Store in Database
    store_in_db(*parsed_data)
    return {"message": f"Resume stored: {parsed_data[0]}"}
