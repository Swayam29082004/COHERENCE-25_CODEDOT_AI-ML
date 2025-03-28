from django.db import models

class Resume(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="resumes/")  # Files saved in /media/resumes/
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Auto timestamp
