import numpy as np
import pandas as pd

# Global predictions log (in production, use a database or persistent store)
PREDICTIONS_LOG = []  # Each entry is a dict: {'gender': <str>, 'prediction': <int>}

def log_prediction(gender, prediction):
    """
    Log a candidate's prediction along with the candidate's gender.
    
    Args:
        gender (str): The candidate's gender (or any other protected attribute).
        prediction (int): Binary prediction (e.g., 1 for a positive outcome, 0 otherwise).
    """
    entry = {
        'gender': gender,
        'prediction': prediction
    }
    PREDICTIONS_LOG.append(entry)

def calculate_disparate_impact(protected_attribute='gender', positive_label=1):
    """
    Calculate the disparate impact ratio based on logged predictions.
    
    Disparate impact ratio is defined as the ratio of the minimum positive
    outcome rate to the maximum positive outcome rate across groups.
    
    Args:
        protected_attribute (str): The key in the prediction log to group by (default: 'gender').
        positive_label (int): The label representing a positive outcome (default: 1).
    
    Returns:
        tuple: (ratio, group_rates) where:
            - ratio is the disparate impact ratio (float) or None if no data,
            - group_rates is a pandas Series containing the positive outcome rate per group.
    """
    if not PREDICTIONS_LOG:
        return None, None
    
    df = pd.DataFrame(PREDICTIONS_LOG)
    # Calculate the positive outcome rate for each group
    group_rates = df.groupby(protected_attribute)['prediction'].apply(
        lambda x: np.mean(x == positive_label)
    )
    
    # Avoid division by zero if max is 0
    if group_rates.empty or group_rates.max() == 0:
        return 0, group_rates

    ratio = group_rates.min() / group_rates.max()
    return ratio, group_rates




# at views.py file 

import os
import joblib
import numpy as np
import fitz  # PyMuPDF for PDF text extraction
import spacy
from django.conf import settings
from django.http import JsonResponse
from django.views import View
from django.core.files.storage import FileSystemStorage
from sklearn.feature_extraction.text import TfidfVectorizer

# Import our bias detection functions
from bias_detection import log_prediction, calculate_disparate_impact

# Load NLP and ML components (assumed to be pre-trained and saved)
nlp = spacy.load("en_core_web_trf")
model = joblib.load("job_fit_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

class ResumeUploadView(View):
    def post(self, request):
        # Step 1: Save the uploaded resume PDF file
        uploaded_file = request.FILES['resume']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)

        # Step 2: Extract text from the PDF using PyMuPDF
        resume_text = self.extract_text_from_pdf(file_path)
        
        # Step 3: Extract candidate info using NLP (customize this as needed)
        candidate_info = self.extract_candidate_info(resume_text)
        
        # For demonstration, assume candidate's gender is provided via the form
        candidate_gender = request.POST.get('gender', 'Unknown')
        candidate_info['gender'] = candidate_gender

        # Step 4: Combine candidate info with a job description (from the form)
        job_description = request.POST.get('job_description', '')
        combined_text = candidate_info.get('skills', '') + " " + job_description
        
        # Convert the combined text to numerical features using TF-IDF
        X_text = tfidf.transform([combined_text]).toarray()
        # Assume 'years_experience' is extracted; default to 0 if not available
        years_experience = candidate_info.get('years_experience', 0)
        X_candidate = np.hstack((X_text, [[years_experience]]))
        
        # Step 5: Predict candidate-job fit (binary outcome; threshold = 0.5)
        fit_proba = model.predict_proba(X_candidate)[:, 1][0]
        prediction = 1 if fit_proba >= 0.5 else 0
        
        # Log the prediction along with the candidate's gender for bias monitoring
        log_prediction(candidate_gender, prediction)
        
        # Step 6: Calculate bias metric (disparate impact ratio) over all predictions
        ratio, group_rates = calculate_disparate_impact(protected_attribute='gender', positive_label=1)
        bias_message = "Fair" if ratio is None or ratio >= 0.8 else "Potential Bias Detected"
        
        # Optionally, remove the file after processing
        fs.delete(filename)
        
        # Return results as JSON
        return JsonResponse({
            "candidate_info": candidate_info,
            "job_fit_probability": f"{fit_proba*100:.2f}%",
            "binary_prediction": prediction,
            "bias_metric": {
                "disparate_impact_ratio": ratio,
                "group_positive_rates": group_rates.to_dict() if group_rates is not None else {},
                "message": bias_message
            },
            "total_predictions_logged": len(PREDICTIONS_LOG)  # from bias_detection.py global
        })
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF using PyMuPDF."""
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    
    def extract_candidate_info(self, text):
        """
        Extract candidate details from resume text.
        This is a simplified example using spaCy.
        Customize this function to extract skills, experience, etc.
        """
        doc = nlp(text)
        skills = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]  # dummy extraction
        years_experience = 3  # Dummy value; implement proper extraction logic
        return {
            "skills": " ".join(skills) if skills else "N/A",
            "years_experience": years_experience
        }
