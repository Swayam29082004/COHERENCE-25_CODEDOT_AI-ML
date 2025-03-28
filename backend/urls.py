from django.urls import path
from .views import get_csrf_token, upload_resume, process_uploaded_resume  # ✅ Ensure Correct Import

urlpatterns = [
    path("get-csrf-token/", get_csrf_token, name="get_csrf_token"),
    path("upload-resume/", upload_resume, name="upload_resume"),
    path("process_uploaded_resume/", process_uploaded_resume, name="process_uploaded_resume"),
]
