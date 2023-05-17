from django.contrib import admin

# Register your models here.
from .models import VoiceRecording
from .models import EmotionResult

admin.site.register(VoiceRecording)
admin.site.register(EmotionResult)
