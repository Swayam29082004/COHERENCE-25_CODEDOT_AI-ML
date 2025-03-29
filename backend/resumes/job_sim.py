import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory
sys.path.append("E:/Coding/codedot/backend")  # Add the backend path explicitly

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

import django
from django.conf import settings
import psycopg2
import pandas as pd
import numpy as np
from django.db import connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from rest_framework.decorators import api_view
from rest_framework.response import Response
# from .models import Resume
# from .serializers import ResumeSerializer

django.setup()

@api_view(['POST'])
def recommend_candidates(request):
    print("\n--- Starting Candidate Recommendation ---")  
    job_domain = request.data.get("job_domain", "").strip().lower()
    
    if not job_domain:
        print("! Error: Job domain not provided")
        return Response({"message": "Job domain is required"}, status=400)

    serializer = ResumeSerializer(data=request.data)
    if not serializer.is_valid():
        print("! Error: Invalid request data", serializer.errors)
        return Response(serializer.errors, status=400)
    
    new_resume = serializer.validated_data
    print("✓ Request data validated successfully")
    
    model, vectorizer = train_model(job_domain)  
    if not model or not vectorizer:
        print("! Error: Model training failed")
        return Response({"message": "Model training failed"}, status=400)
    print("✓ Model trained successfully")
    
    ranked_candidates = rank_candidates(new_resume, model, vectorizer, job_domain)
    print(f"Total candidates ranked: {len(ranked_candidates)}")
    
    store_results(ranked_candidates)
    
    return Response(ranked_candidates[:5])

def train_model(job_domain):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT skills, experience, education, job_fit_score 
                FROM job_sim 
                WHERE job_fit_score IS NOT NULL AND lower(skills) LIKE %s
            """, (f"%{job_domain}%",))
            data = cursor.fetchall()

        if not data:
            print(f"❌ No training data found for '{job_domain}'. Falling back to default data.")
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


def rank_candidates(new_resume, model, vectorizer, job_domain):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT name, email, phone, skills, experience, education 
                FROM resumes 
                WHERE lower(skills) LIKE %s
            """, (f"%{job_domain}%",))
            data = cursor.fetchall()

        candidates = []
        new_resume_skills = str(new_resume.get("skills", ""))
        new_resume_vector = vectorizer.transform([new_resume_skills])
        
        for resume in data:
            name, email, phone, skills, experience, education = resume
            skills = str(skills) if skills else ""
            experience = float(experience or 0)
            education = float(education or 0)

            candidate_vector = vectorizer.transform([skills])
            similarity = cosine_similarity(new_resume_vector, candidate_vector)[0][0] * 100
            candidate_features = np.hstack((candidate_vector.toarray(), [[experience, education]]))
            job_fit_score = model.predict(candidate_features)[0]

            candidates.append({
                "name": name,
                "email": email,
                "phone": phone,
                "skills": skills,
                "experience": experience,
                "education": education,
                "job_fit_score": round(job_fit_score, 2),
                "similarity_with_new": round(similarity, 2)
            })

        return sorted(candidates, key=lambda x: (x["job_fit_score"], x["similarity_with_new"]), reverse=True)
    
    except Exception as e:
        print(f"Error ranking candidates: {e}")
        return []
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
            print(f"❌ No training data found for '{job_domain}'.")
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

def rank_candidates(new_resume, model, vectorizer, job_domain):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT name, email, phone, skills, experience, education 
                FROM resumes 
                WHERE lower(array_to_string(skills, ' ')) LIKE %s
            """, (f"%{job_domain}%",))
            data = cursor.fetchall()

        candidates = []
        new_resume_skills = str(new_resume.get("skills", ""))
        new_resume_vector = vectorizer.transform([new_resume_skills])
        
        for resume in data:
            name, email, phone, skills, experience, education = resume
            skills = str(skills) if skills else ""
            experience = float(experience or 0)
            education = float(education or 0)

            candidate_vector = vectorizer.transform([skills])
            similarity = cosine_similarity(new_resume_vector, candidate_vector)[0][0] * 100
            candidate_features = np.hstack((candidate_vector.toarray(), [[experience, education]]))
            job_fit_score = model.predict(candidate_features)[0]

            candidates.append({
                "name": name,
                "email": email,
                "phone": phone,
                "skills": skills,
                "experience": experience,
                "education": education,
                "job_fit_score": round(job_fit_score, 2),
                "similarity_with_new": round(similarity, 2)
            })

        return sorted(candidates, key=lambda x: (x["job_fit_score"], x["similarity_with_new"]), reverse=True)
    
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
    
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error storing results: {e}")
