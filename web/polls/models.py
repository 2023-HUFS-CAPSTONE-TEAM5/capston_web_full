from django.db import models
from django.utils import timezone

# Create your models here.
# 음성 녹음 데이터를 저장할 모델을 정의

# 이 모델은 audio_file 필드와 uploaded_at 필드로 구성되어 있다.
# audio_file 필드는 FileField로, 파일을 저장할 경로를 upload_to 매개변수로 설정한다.
# uploaded_at 필드는 DateTimeField로, 오디오 파일이 업로드된 시간을 저장한다.


class VoiceRecording(models.Model):
    audio_file = models.FileField(upload_to="audio")
    uploaded_at = models.DateTimeField(default=timezone.now)
    gender = models.CharField(max_length=10)
    emotion_result = models.OneToOneField(
        "EmotionResult", on_delete=models.CASCADE, null=True, blank=True
    )


class EmotionResult(models.Model):
    emotion = models.CharField(max_length=20)
    ratio = models.CharField(max_length=50)
