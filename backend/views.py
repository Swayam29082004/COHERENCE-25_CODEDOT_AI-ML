from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
from .models import Resume
from .nlp_parse import extract_text_from_pdf, extract_details, store_in_db, process_resume

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


