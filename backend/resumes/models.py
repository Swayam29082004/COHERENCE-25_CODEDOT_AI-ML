from django.db import models

class Resume(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="resumes/")  # Files saved in /media/resumes/
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Auto timestamp


# resumes/models.py

from django.db import models

class Candidate(models.Model):
    name = models.CharField(max_length=255)
    skills = models.JSONField()
    experience = models.IntegerField()
    education = models.IntegerField()

    def __str__(self):
        return self.name

from rest_framework import serializers

class ResumeSerializer(serializers.Serializer):
    skills = serializers.ListField(child=serializers.CharField())
    experience = serializers.FloatField()
    education = serializers.FloatField()
