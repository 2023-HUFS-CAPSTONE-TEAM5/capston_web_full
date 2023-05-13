from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import VoiceRecording, EmotionResult
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
from datetime import datetime  # datetime 모듈 추가
from django.core.files.storage import FileSystemStorage
import logging

import librosa
import numpy as np
import pandas as pd
import warnings
import sys

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import os
from sklearn.preprocessing import minmax_scale
from collections import Counter
from pydub import AudioSegment

from torch.utils.data import DataLoader
import logging


# Create your views here.
def index(request):
    template = loader.get_template("polls/index.html")
    print("연습")
    return render(request, "polls/index.html")


def analysis(request):
    template = loader.get_template("polls/analysis.html")
    return render(request, "polls/analysis.html")


def login(request):
    template = loader.get_template("polls/login.html")
    return render(request, "polls/login.html")


def signUp(request):
    template = loader.get_template("polls/signUp.html")
    return render(request, "polls/signUP.html")


def mypage(request):
    if request.method == "POST":
        recording = VoiceRecording(
            audio_file=request.FILES["audio_file"], gender=request.POST.get("gender")
        )
        recording.save()

        return JsonResponse(
            {
                "id": recording.id,
                "uploaded_at": recording.uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
                "gender": recording.gender,
            }
        )

    else:
        recordings = VoiceRecording.objects.all()
        context = {"recordings": recordings}
        return render(request, "polls/mypage.html", context)


def recording(request):
    if request.method == "POST":
        audio_file = request.FILES.get("audio_file")
        gender = request.POST.get("gender")
        if audio_file:
            # 파일이 올바르게 첨부된 경우
            # 파일을 읽어들이고 데이터베이스에 저장
            file_name = default_storage.save(
                audio_file.name, ContentFile(audio_file.read())
            )
            recording = VoiceRecording(audio_file=file_name, gender=gender)
            recording.save()

            return JsonResponse(
                {
                    "id": recording.id,
                    "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "gender": recording.gender,
                }
            )

    return render(request, "polls/recording.html")

    # def temp(request):
    if request.method == "POST":
        recording = VoiceRecording(
            audio_file=request.FILES["audio_file"], gender=request.POST.get("gender")
        )
        recording.save()
        logger = logging.getLogger("mylogger")
        logger.debug("로그:" + str(recording.audio_file))
        print("test")

        # 감정 분석
        PKL_LOCATION = generate_pkl(recording.audio_file)
        test_set = Voice_dataset(pkl_location=PKL_LOCATION)
        test_loader = DataLoader(
            test_set, batch_size=len(test_set), shuffle=False, num_workers=8
        )

        MALE_PATH = "C:\\Users\\yttn0\\Desktop\\git\\capston_web_full\\web\\polls\\pth\\male_best_model_epoch_70.pth"
        FEMALE_PATH = "C:\\Users\\yttn0\\Desktop\\git\\capston_web_full\\web\\polls\\pth\\female_best_model_epoch_110.pth"

        # 초기 모델 선언 (모델 구조 저장)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNTransformer(num_emotions=8).to(device)

        # Test
        if recording.gender == "male":
            emotions_ratio, max_emotion = test(model, test_loader, path=MALE_PATH)
        elif recording.gender == "female":
            emotions_ratio, max_emotion = test(model, test_loader, path=FEMALE_PATH)

        emotions = EmotionResult(emotion=max_emotion, ratio=emotions_ratio)
        emotions.save()
        if not emotions.emotion:
            emotions.emotion = "아무것도 안들어감"
        return JsonResponse(
            {
                "id": recording.id,
                "uploaded_at": recording.uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
                "gender": recording.gender,
                "emotions_ratio": emotions.ratio,
                "max_emotion": emotions.emotion,
            }
        )
    else:
        recordings = VoiceRecording.objects.all()
        context = {"recordings": recordings}
        return render(request, "polls/test.html", context)
