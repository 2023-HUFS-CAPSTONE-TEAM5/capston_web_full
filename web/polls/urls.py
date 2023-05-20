from django.urls import path
from django.urls import re_path as url
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("", views.index, name="index"),
    path("analysis/", views.analysis, name="analysis"),
    path("login/", views.login, name="login"),
    path("mypage/", views.mypage, name="mypage"),
    path("recording/", views.recording, name="recording"),
    path("signUp/", views.signUp, name="signUp"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
