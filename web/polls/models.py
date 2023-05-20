from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class VoiceRecording(models.Model):
    audio_file = models.FileField(upload_to="audio")
    uploaded_at = models.DateTimeField(default=timezone.now)
    gender = models.CharField(max_length=10)
    user = models.ForeignKey(
        "User",
        on_delete=models.CASCADE,
        related_name="voice_recordings",
        related_query_name="voice_recording",
    )
    emotion_result = models.OneToOneField(
        "EmotionResult", on_delete=models.CASCADE, null=True, blank=True
    )


class EmotionResult(models.Model):
    emotion = models.CharField(max_length=20)
    ratio = models.CharField(max_length=50)


class User(AbstractUser):
    GENDER_CHOICES = (
        ("M", "Male"),
        ("F", "Female"),
        ("O", "Other"),
    )

    name = models.CharField(max_length=255)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)
    occupation = models.CharField(max_length=255)
    age = models.PositiveIntegerField(null=True, blank=True)
