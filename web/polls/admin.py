from django.contrib import admin
from .models import VoiceRecording, EmotionResult, User

admin.site.register(VoiceRecording)
admin.site.register(EmotionResult)


class UserAdmin(admin.ModelAdmin):
    list_display = ["username", "first_name", "last_name", "email"]


admin.site.register(User, UserAdmin)
