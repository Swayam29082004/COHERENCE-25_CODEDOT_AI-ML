from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
from .models import Resume
from .nlp_parse import extract_text_from_pdf, extract_details, store_in_db, process_resume

from django.http import JsonResponse
from .job_sim import train_model, rank_candidates, store_results
from django.utils import timezone
from django.shortcuts import render

# ✅ Ensure CSRF Token Function Exists
def get_csrf_token(request):
    return JsonResponse({"csrfToken": get_token(request)})

@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        uploaded_file = request.FILES["resume"]

        # ✅ Ensure 'resume' directory exists
        resume_folder = os.path.join(settings.MEDIA_ROOT, "resume")
        os.makedirs(resume_folder, exist_ok=True)

        # ✅ Delete existing resume file (if any)
        existing_resumes = Resume.objects.all()
        for resume in existing_resumes:
            if resume.file:  # Remove the old file from storage
                old_file_path = os.path.join(settings.MEDIA_ROOT, str(resume.file))
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)

        existing_resumes.delete()  # Remove old database records

        # ✅ Save new file as 'resume.pdf'
        resume_path = os.path.join(resume_folder, "resume.pdf")
        with open(resume_path, "wb") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # ✅ Save file reference to database
        new_resume = Resume.objects.create(
            name="resume.pdf",
            file="resume/resume.pdf"  # Relative path from MEDIA_ROOT
        )

        return JsonResponse({
            "message": "Resume uploaded successfully",
            "resume_id": new_resume.id,
            "file_path": new_resume.file.url  # Return a proper URL
        })

    return JsonResponse({"error": "No file uploaded"}, status=400)

# ✅ Function to Get the Latest Resume Path
def get_latest_pdf_path():
    resume_path = os.path.join(settings.MEDIA_ROOT, "resume/resume.pdf")
    return resume_path if os.path.exists(resume_path) else None

# ✅ API to Get Latest PDF Path Dynamically
def get_latest_pdf(request):
    pdf_path = get_latest_pdf_path()

    if pdf_path:
        return JsonResponse({
            "latest_pdf_path": f"{settings.MEDIA_URL}resume/resume.pdf"
        }, status=200)

    return JsonResponse({"error": "No PDF files found"}, status=404)

# ✅ API to Extract and Store Resume Details
@csrf_exempt
def process_uploaded_resume(request):
    pdf_path = get_latest_pdf_path()
    
    if not pdf_path:
        return JsonResponse({"error": "No resume found to process"}, status=404)

    # Step 1: Extract text from the PDF
    print("\n### Extracting Text from PDF ###")
    resume_text = extract_text_from_pdf(pdf_path)
    print(resume_text)

    # Step 2: Extract structured details
    print("\n### Extracted Resume Details ###")
    parsed_data = extract_details(resume_text)
    extracted_data = {
        "name": parsed_data[0],
        "email": parsed_data[1],
        "phone": parsed_data[2],
        "skills": parsed_data[3],
        "experience": parsed_data[4],
        "education": parsed_data[5]
    }
    print(extracted_data)  # ✅ Print as dictionary before proceeding

    # Step 3: Store extracted details in DB
    print("\n### Storing Data in Database ###")
    store_in_db(*parsed_data)
    print("Data stored successfully.")

    # Return Response
    return JsonResponse({
        "message": f"Resume processed successfully for {parsed_data[0]}",
        "data": extracted_data
    })


from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Resume  # Ensure Resume model is correctly defined
from .job_sim import rank_candidates, train_model  # Ensure job_sim.py exists
from .serializers import ResumeSerializer

# @api_view(['POST'])
# def recommend_candidates(request):
#     print("\n--- Starting Candidate Recommendation ---")  # Log start

#     # Deserialize request data
#     serializer = ResumeSerializer(data=request.data)
#     if not serializer.is_valid():
#         print("! Error: Invalid request data")  # Log validation failure
#         print("Errors:", serializer.errors)    # Print errors for debugging
#         return Response({"error": "Invalid request data", "details": serializer.errors}, status=400)

#     print("✓ Request data validated successfully")
#     new_resume = serializer.validated_data
#     print("New Resume Data:", new_resume)  # Log extracted resume data

#     # Train model and vectorizer
#     print("Training model and vectorizer...")
#     try:
#         model, vectorizer = train_model()
#         if not model or not vectorizer:
#             raise ValueError("Model training failed due to insufficient or invalid data.")
#     except Exception as e:
#         print(f"! Error: {e}")  # Log exception
#         return Response({"message": "Model training failed", "error": str(e)}, status=400)

#     print("✓ Model and vectorizer trained successfully")

#     # Rank candidates
#     print("Ranking candidates...")
#     try:
#         ranked_candidates = rank_candidates(new_resume, model, vectorizer)
#         print(f"Total candidates ranked: {len(ranked_candidates)}")  # Log total candidates
#     except Exception as e:
#         print(f"! Error during ranking: {e}")  # Log exception
#         return Response({"message": "Error ranking candidates", "error": str(e)}, status=500)

#     # Extract top 5 candidates
#     if not ranked_candidates:
#         print("⚠️ No suitable candidates found")
#         return Response({"message": "No suitable candidates found"}, status=404)

#     top_candidates = ranked_candidates[:5]
#     print("\n--- Top 5 Recommended Candidates ---")  # Log header
#     for i, candidate in enumerate(top_candidates, 1):
#         print(f"{i}. {candidate}")  # Log each candidate

#     print("\n--- Recommendation Complete ---")  # Log completion
#     return Response(top_candidates)

from django.http import JsonResponse
from .models import Resume  # Replace with your actual model
from django.db import connection
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd  # Ensure this is imported at the top of your module
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor

@api_view(['GET'])
def get_stored_resumes(request):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, name, email, phone, skills, experience, education, gender
                FROM resumes
                ORDER BY experience DESC
                LIMIT 10;
            """)
            data = cursor.fetchall()

        resumes = [
            {
                "ID": row[0],
                "Name": row[1],
                "Email": row[2],
                "Phone": row[3],
                "Skills": row[4],
                "Experience": row[5],
                "Education": row[6],
                "Gender": row[7]
            }
            for row in data
        ]

        return Response({"resumes": resumes}, status=200)

    except Exception as e:
        return Response({"message": "Error fetching resumes", "error": str(e)}, status=500)



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


@csrf_exempt  # Disable CSRF for testing (remove in production)
def recommend_candidates(request):
    # if request.method == "POST":
    #     try:
    #         data = json.loads(request.body.decode('utf-8'))  # Parse JSON data
    #         job_domain = data.get("job_domain")
    #         skills = data.get("skills")
    #         experience = data.get("experience")
    #         education = data.get("education")

    #         # Dummy response for now
    #         return JsonResponse({"message": "Success", "data": data})

    #     except json.JSONDecodeError:
    #         return JsonResponse({"error": "Invalid JSON format"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)  # Ret



stored_text = ""  # Variable to store the received text

@api_view(['POST'])
def store_text(request):
    global stored_text
    stored_text = request.data.get("text", "").strip()
    print("Received Text:", stored_text)  # Print input

    if not stored_text:
        return Response({"message": "No text provided"}, status=400)

    # Extract job domain from the HR input text.
    job_domain = extract_job_domain(stored_text)
    
    # Try to train the model from the job_sim table.
    model, vectorizer = train_model(job_domain)
    if not model or not vectorizer:
        print(f"❌ No training data found for '{job_domain}'. Trying fallback model.")
        model, vectorizer = fallback_model(job_domain)
        if not model or not vectorizer:
            return Response({"message": "No candidate data available for fallback."}, status=400)

    # Convert the HR input into a resume-like dictionary.
    new_resume = extract_resume_details(stored_text)
    
    ranked_candidates = rank_candidates(new_resume, model, vectorizer, job_domain)
    print(f"Total candidates ranked: {len(ranked_candidates)}")
    
    store_results(ranked_candidates)
    
    return Response({
        "message": "Text processed successfully",
        "top_candidates": ranked_candidates[:5]
    })


def extract_job_domain(text):
    # Look for job-related keywords in the text.
    keywords = ["data science", "web developer", "backend", "frontend", "full stack", "machine learning"]
    for keyword in keywords:
        if keyword in text.lower():
            return keyword
    return "general"  # Default domain if no match


def extract_resume_details(text):
    # Very basic extraction; improve this with an NLP solution as needed.
    experience = extract_number_after_keyword(text, "experience")
    education = extract_number_after_keyword(text, "education")
    skills = extract_skills_from_text(text)
    return {
        "skills": skills,
        "experience": experience,
        "education": education
    }


def extract_number_after_keyword(text, keyword):
    import re
    match = re.search(rf"{keyword}\s*:\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def extract_skills_from_text(text):
    known_skills = ["Python", "Django", "JavaScript", "SQL", "Machine Learning", "Deep Learning", "Data Science", "HTML", "CSS"]
    return ", ".join([skill for skill in known_skills if skill.lower() in text.lower()])


def train_model(job_domain):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT skills, experience, education, job_fit_score 
                FROM job_sim 
                WHERE job_fit_score IS NOT NULL AND lower(array_to_string(skills, ' ')) LIKE %s
            """, (f"%{job_domain}%",))
            data = cursor.fetchall()

        if not data:
            print(f"❌ No training data found for '{job_domain}'. Trying all available training data.")
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT skills, experience, education, job_fit_score 
                    FROM job_sim 
                    WHERE job_fit_score IS NOT NULL
                """)
                data = cursor.fetchall()
            if not data:
                print("❌ No training data available at all.")
                return None, None  

        df = pd.DataFrame(data, columns=["skills", "experience", "education", "job_fit_score"])
        df.fillna({"skills": "", "experience": 0, "education": "0", "job_fit_score": 50}, inplace=True)
        df["education"] = pd.to_numeric(df["education"], errors="coerce").fillna(0)
        df["experience"] = pd.to_numeric(df["experience"], errors="coerce").fillna(0)
        df["skills"] = df["skills"].astype(str)

        vectorizer = TfidfVectorizer()
        skills_vectorized = vectorizer.fit_transform(df["skills"])
        
        X = np.hstack((skills_vectorized.toarray(), df[["experience", "education"]].values))
        y = df["job_fit_score"].values

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        return model, vectorizer

    except Exception as e:
        print(f"Error in training model: {e}")
        return None, None


def fallback_model(job_domain):
    """
    Build a fallback vectorizer and dummy model using candidate resumes.
    The dummy model always returns a default job fit score (e.g., 50).
    """
    try:
        # Query candidate data with the job domain filter first.
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT skills, experience, education 
                FROM resumes 
                WHERE lower(array_to_string(skills, ' ')) LIKE %s
            """, (f"%{job_domain}%",))
            data = cursor.fetchall()
        if not data:
            print("❌ No candidate data found in resumes for fallback with domain filter. Trying all resumes.")
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT skills, experience, education 
                    FROM resumes
                """)
                data = cursor.fetchall()
        if not data:
            print("❌ No candidate data found in resumes even without filtering.")
            return None, None

        df = pd.DataFrame(data, columns=["skills", "experience", "education"])
        df.fillna({"skills": "", "experience": 0, "education": "0"}, inplace=True)
        df["experience"] = pd.to_numeric(df["experience"], errors="coerce").fillna(0)
        df["education"] = pd.to_numeric(df["education"], errors="coerce").fillna(0)
        df["skills"] = df["skills"].astype(str)

        vectorizer = TfidfVectorizer()
        vectorizer.fit(df["skills"])

        # Create a dummy model that returns a constant job fit score.
        class DummyModel:
            def predict(self, X):
                return np.full((X.shape[0],), 50)
        dummy_model = DummyModel()
        print("✅ Fallback dummy model created.")
        return dummy_model, vectorizer

    except Exception as e:
        print(f"Error in fallback_model: {e}")
        return None, None


def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def rank_candidates(new_resume, model, vectorizer, job_domain):
    try:
        # Try to fetch candidates matching the job domain.
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT name, email, phone, skills, experience, education 
                FROM resumes 
                WHERE lower(array_to_string(skills, ' ')) LIKE %s
            """, (f"%{job_domain}%",))
            data = cursor.fetchall()

        # Fallback: if no candidates match the domain, fetch all resumes.
        if not data:
            print("No candidates matched the domain. Fetching all resumes.")
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT name, email, phone, skills, experience, education 
                    FROM resumes
                """)
                data = cursor.fetchall()

        candidates = []
        # Process HR input skills for vectorization.
        new_resume_skills = new_resume.get("skills", "")
        # Ensure new_resume_skills is a string.
        if not isinstance(new_resume_skills, str):
            new_resume_skills = str(new_resume_skills)
        new_resume_vector = vectorizer.transform([new_resume_skills])
        
        for resume in data:
            name, email, phone, skills, experience, education = resume
            # 'skills' should be a list (from text[]); if not, try converting.
            if isinstance(skills, list):
                skills_list = skills
                skills_str = ", ".join(skills)
            else:
                # If skills is a string (or None), split by commas
                skills_str = skills or ""
                skills_list = [s.strip() for s in skills_str.split(",") if s.strip()]
            
            experience = safe_float(experience)
            education = safe_float(education)

            # Use the comma-separated string for vectorization.
            candidate_vector = vectorizer.transform([skills_str])
            similarity = cosine_similarity(new_resume_vector, candidate_vector)[0][0] * 100
            candidate_features = np.hstack((candidate_vector.toarray(), [[experience, education]]))
            job_fit_score = model.predict(candidate_features)[0]

            # Calculate the combined score with your chosen weights.
            weight_fit = 0.7
            weight_sim = 0.3
            combined_score = weight_fit * job_fit_score + weight_sim * similarity

            candidates.append({
                "name": name,
                "email": email,
                "phone": phone,
                "skills": skills_list,  # Keep this as a list for proper DB insertion.
                "experience": experience,
                "education": education,
                "job_fit_score": round(job_fit_score, 2),
                "similarity_with_new": round(similarity, 2),
                "combined_score": round(combined_score, 2)
            })

        # Sort candidates by combined score (highest first) to get top matches.
        ranked_candidates = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        return ranked_candidates

    except Exception as e:
        print(f"Error ranking candidates: {e}")
        return []



def store_results(ranked_candidates):
    if not ranked_candidates:
        print("⚠️ No candidates to store.")
        return
    
    try:
        with connection.cursor() as cursor:
            for candidate in ranked_candidates:
                cursor.execute("""
                    INSERT INTO job_sim (name, email, phone, skills, experience, education, job_fit_score, similarity_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO UPDATE 
                    SET job_fit_score = EXCLUDED.job_fit_score, 
                        similarity_score = EXCLUDED.similarity_score
                """, (
                    candidate["name"], candidate["email"], candidate["phone"],
                    candidate["skills"], candidate["experience"], candidate["education"],
                    candidate["job_fit_score"], candidate["similarity_with_new"]
                ))
        connection.commit()
        print("✅ Results stored successfully.")
    
    except Exception as e:
        print(f"Error storing results: {e}")


from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame

def avg_combined_score(y_true, y_pred):
    return np.mean(y_pred)

def calculate_bias_metrics(top_candidates):
    # Convert candidate list into a DataFrame.
    df = pd.DataFrame(top_candidates)
    
    # Ensure a sensitive feature exists. Here we assume 'gender'.
    if 'gender' not in df.columns:
        df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
    
    # Create a dummy y_true array (required by MetricFrame).
    y_true = np.ones(len(df), dtype=float)
    # Ensure y_pred is numeric.
    y_pred = df["combined_score"].to_numpy(dtype=float)
    # Sensitive features array.
    sensitive_features = df["gender"].to_numpy()
    
    # Compute fairness metrics.
    metric_frame = MetricFrame(
        metrics={"average_combined_score": avg_combined_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Calculate group percentages.
    group_percentages = df["gender"].value_counts(normalize=True) * 100
    
    return metric_frame, group_percentages

@api_view(['POST'])
def get_top_candidates(request):
    """
    Returns a JSON response with top candidates and fairness metrics.
    """
    # For demonstration, we use a fixed list.
    top_candidates = [
        {"name": "Candidate1", "gender": "Male", "combined_score": 85.0, "job_fit_score": 88, "similarity_with_new": 80},
        {"name": "Candidate2", "gender": "Female", "combined_score": 82.0, "job_fit_score": 83, "similarity_with_new": 80},
        {"name": "Candidate3", "gender": "Male", "combined_score": 80.0, "job_fit_score": 80, "similarity_with_new": 80},
        {"name": "Candidate4", "gender": "Female", "combined_score": 78.0, "job_fit_score": 78, "similarity_with_new": 80},
        {"name": "Candidate5", "gender": "Male", "combined_score": 75.0, "job_fit_score": 75, "similarity_with_new": 80},
    ]
    
    # Compute fairness metrics.
    metric_frame, group_percentages = calculate_bias_metrics(top_candidates)
    fairness_metrics = {
        "by_group": metric_frame.by_group.to_dict(),
        "group_percentages": group_percentages.to_dict()
    }
    
    # Return a JSON response.
    return Response({
        "message": "Top candidates and fairness metrics computed successfully",
        "top_candidates": top_candidates,
        "fairness_metrics": fairness_metrics,
    })



import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import google.generativeai as genai
from dotenv import load_dotenv
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Load environment variables (ensure you have a .env file with GENAI_API_KEY)
genai.configure(api_key="AIzaSyC1l9rED1nJeliRvS3LtWD3IxfC_Goue0E")
# model = genai.GenerativeModel("gemini-1.5-flash-latest")

def extract_text_from_pdf(pdf_path):
    """
    Attempts to extract text directly from a PDF using pdfplumber.
    If that fails or returns empty, falls back to OCR using pytesseract.
    """
    text = ""
    try:
        # Try direct text extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    # Fallback to OCR for image-based PDFs
    print("Falling back to OCR for image-based PDF.")
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    except Exception as e:
        print(f"OCR failed: {e}")

    return text.strip()

def analyze_resume(resume_text, job_description=None):
    """
    Uses Google’s Gemini model to generate an analysis of the resume.
    Optionally compares the resume to a provided job description.
    """
    if not resume_text:
        return {"error": "Resume text is required for analysis."}
    
    # Choose the Gemini model (adjust the model name as needed)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    base_prompt = f"""
    You are an experienced HR professional with technical expertise in roles such as Data Scientist, Data Analyst, DevOPS, Machine Learning Engineer, Prompt Engineer, AI Engineer, Full Stack Web Developer, Big Data Engineer, Marketing Analyst, Human Resource Manager, or Software Developer.
    Your task is to review the provided resume.
    Please share your professional evaluation on whether the candidate's profile aligns with the role.
    Also, mention the skills the candidate already has, suggest additional skills to improve the resume,
    and recommend some courses that might help improve those skills. Highlight the strengths and weaknesses.
    
    Resume:
    {resume_text}
    """
    
    if job_description:
        base_prompt += f"""
        Additionally, compare this resume to the following job description:
        
        Job Description:
        {job_description}
        
        Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
        """
    
    response = model.generate_content(base_prompt)
    analysis = response.text.strip()
    return analysis

@api_view(['POST'])
def gemini_analysis(request):
    """
    Endpoint for Gemini analysis of a resume.
    The request should include:
    - "pdf_path": (optional) the path to the PDF file to analyze. If not provided, a default path is used.
    - "job_description": (optional) a job description to compare against.
    Returns a JSON response containing the extracted resume text and the Gemini analysis.
    """
    pdf_path = request.data.get(r"E:\Coding\codedot\backend\media\resume\resume.pdf")
    if not pdf_path:
        # Use a default PDF path. Adjust this as needed.
        pdf_path = os.path.join(os.getcwd(), "media", "resume", "resume.pdf")
    
    job_description = request.data.get("job_description")
    
    # Extract text from the PDF
    resume_text = extract_text_from_pdf(pdf_path)
    
    # Analyze the resume using Gemini
    analysis = analyze_resume(resume_text, job_description)
    
    return Response({
        "message": "Gemini analysis completed successfully",
        "resume_text": resume_text,
        "analysis": analysis
    })
