from django.urls import path
from .views import get_csrf_token, upload_resume, process_uploaded_resume, recommend_candidates, get_stored_resumes, store_text, get_top_candidates, gemini_analysis   # ✅ Ensure Correct Import

urlpatterns = [
    path("get-csrf-token/", get_csrf_token, name="get_csrf_token"),
    path("upload-resume/", upload_resume, name="upload_resume"),
    path("process_uploaded_resume/", process_uploaded_resume, name="process_uploaded_resume"),
    path('recommend_candidates/', recommend_candidates, name='recommend_candidates'),
    path('get-resumes/', get_stored_resumes, name='get_resumes'),
    path('store-text/', store_text,name="store_text"),
    path('api/get-top-candidates/', get_top_candidates, name='get_top_candidates'),
    path("gemini-analysis/", gemini_analysis, name="gemini_analysis"),
]
