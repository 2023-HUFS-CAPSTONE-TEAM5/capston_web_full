import json
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import get_user_model


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
import soundfile
import subprocess

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
import pydub

from torch.utils.data import DataLoader
import logging
import subprocess
from django.apps import apps
from django.utils import timezone

if not apps.ready:
    apps.populate(settings.INSTALLED_APPS)


# Create your views here.
def index(request):
    print("test")
    template = loader.get_template("polls/index.html")
    return render(request, "polls/index.html")


def analysis(request):
    template = loader.get_template("polls/analysis.html")
    return render(request, "polls/analysis.html")


from django.contrib.auth import authenticate, login as user_login


def login(request):
    template = loader.get_template("polls/login.html")

    if request.method == "POST":
        username = request.POST.get("userName")
        password = request.POST.get("userPassword")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            print("로그인 완료")
            user_login(request, user)
            return redirect("mypage")  # 로그인 성공 시 리디렉션할 페이지 URL 이름으로 'mypage'을 대체해주세요
        else:
            return render(request, "polls/login.html", {"error": "유효하지 않은 로그인 정보입니다."})

    return render(request, "polls/login.html")


User = get_user_model()


def signUp(request):
    if request.method == "POST":
        email = request.POST.get("userName")
        password = request.POST.get("userPassword")
        password_check = request.POST.get("userPasswordCheck")
        name = request.POST.get("name")
        gender = request.POST.get("gender")
        job = request.POST.get("job")
        age = request.POST.get("age")
        print(email, name, gender)
        if gender == "여성":
            gender = "F"
        else:
            gender = "M"
        if password == password_check:
            user = User.objects.create_user(
                username=email, email=email, password=password, gender=gender, age=age
            )
            user.name = name
            user.occupation = job
            user.save()

            # 추가 동작 또는 리디렉션 수행
            print("회원가입 완료")
            # 회원가입 성공 시 'signup_success' 변수를 전달하여 템플릿에서 사용할 수 있도록 함
            return render(request, "polls/signUP.html", {"signup_success": True})

        # 비밀번호 불일치 오류 처리

        return render(request, "polls/signUP.html", {"error": "비밀번호가 일치하지 않습니다."})

    return render(request, "polls/signUP.html")


from django.contrib.auth.decorators import login_required


@login_required
def mypage(request):
    from .models import VoiceRecording, EmotionResult

    if request.method == "POST":
        recording = VoiceRecording(
            audio_file=request.FILES["audio_file"],
            gender=request.POST.get("gender"),
            uploaded_at=timezone.now(),
        )
        recording.save()

        if not request.POST.get("max_emotion"):
            max_emotion = "감정 없음"
        else:
            max_emotion = request.POST.get("max_emotion")

        if not request.POST.get("emotions_ratio"):
            emotions_ratio = "비율 없음"
        else:
            emotions_ratio = request.POST.get("emotions_ratio")

        recording.emotion_result = EmotionResult(
            emotion=max_emotion, ratio=emotions_ratio
        )
        recording.emotion_result.save()
        recording.save()
        print(f"uploaded_at:{recording.uploaded_at}")
        # print(f"mypage = {emotions.ratio}")
        # print(f"mypage- emotion = {emotions.emotion}")

        return JsonResponse(
            {
                "id": recording.id,
                "uploaded_at": recording.uploaded_at.strftime("%Y/%m/%d %H:%M"),
                "gender": recording.gender,
                "emotions_ratio": recording.emotion_result.ratio,
                "max_emotion": recording.emotion_result.emotion,
            }
        )

    else:
        user = request.user
        recordings = VoiceRecording.objects.filter(user=user)
        context = {"recordings": recordings}
        return render(request, "polls/mypage.html", context)


class Data:
    def __init__(self, wav_file, gender):
        self.wav_file = wav_file
        self.gender = gender


# Data Pre-processing
def MELSpectrogram(signal, sample_rate):
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        fmax=sample_rate / 2,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def generate_pkl(INPUT_WAV_PATH):  # 입력된 wav 파일을 .pkl(입력 음성의 경로, 멜스펙트로그램 포함) 형식으로 변환
    DURATION = 3.0
    SAMPLE_RATE = librosa.get_samplerate(INPUT_WAV_PATH)
    audio, _ = librosa.load(
        INPUT_WAV_PATH, duration=DURATION, offset=10.0, sr=SAMPLE_RATE
    )

    df_path = pd.DataFrame(columns=["path"])
    df_mel = pd.DataFrame(columns=["feature"])

    print(f"generate_wav_path:{INPUT_WAV_PATH}")

    audio, _ = librosa.effects.trim(audio, top_db=60)  # 묵음 처리

    for i, p in enumerate(INPUT_WAV_PATH):
        SAMPLE_RATE = librosa.get_samplerate(INPUT_WAV_PATH)
        temp_audio = np.zeros(
            (
                int(
                    SAMPLE_RATE * DURATION,
                )
            )
        )
        temp_audio[: len(audio)] = audio
        mel = MELSpectrogram(temp_audio, sample_rate=SAMPLE_RATE)
        df_path.loc[i] = p
        df_mel.loc[i] = [mel]

    PKL_DIR = "media/audio/pkl/"
    # 디렉토리가 존재하지 않으면 생성합니다
    os.makedirs(PKL_DIR, exist_ok=True)

    df = pd.concat([df_path, df_mel], axis=1)
    df.to_pickle(PKL_DIR + "test.pkl")
    PKL_LOCATION = os.path.join(PKL_DIR, "test.pkl")
    return PKL_LOCATION


# test.pkl을 Pytorch의 Dataset 형태로 변환해주는 함수
class Voice_dataset(Dataset):
    def __init__(self, pkl_location):
        self.df = pd.read_pickle(pkl_location)
        print(self.df)

    def normalize(self, data):
        return minmax_scale(data, feature_range=(0, 1))

    def __len__(self):  # returns the length of the data set
        return len(self.df)

    def __getitem__(self, idx):
        voice = dict()
        voice["features"] = self.df.iloc[idx, 1]
        return voice


# 사용한 모델의 구조 (모델 불러오기 위해 필요)
class CNNTransformer(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # 2. conv block
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 3. conv block
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 4. conv block
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])
        transf_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=512, dropout=0.4, activation="relu"
        )
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)

        # Linear softmax layer
        self.out_linear = nn.Linear(320, num_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # conv embedding
        conv_embedding = self.conv2Dblock(x)  # (b,channel,freq,time)
        conv_embedding = torch.flatten(
            conv_embedding, start_dim=1
        )  # do not flatten batch dimension

        # transformer embedding
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(
            2, 0, 1
        )  # requires shape = (time,batch,embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)

        # concatenate
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)

        # final Linear
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_softmax


# Test
def print_test_result(predictions):
    total_count = {
        "neutral": predictions[0],
        "happy": predictions[1],
        "sad": predictions[2],
        "angry": predictions[3],
        "fearful": predictions[4],
        "disgust": predictions[5],
        "surprised": predictions[6],
    }

    emotion_ratio = {}
    for emotion in total_count.keys():
        emotion_ratio[emotion] = round((total_count[emotion]) * 100, 5)
        print(f"{emotion} : {total_count[emotion] * 100:.5f}%")

    max_emotion = max(total_count, key=total_count.get)
    print(f'가장 큰 비율을 차지하고 있는 감정은 "{max_emotion}" 입니다.')

    return emotion_ratio, max_emotion


# helper function for computing model accuracy
def test(model, loader, path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 성별에 따라 다르게 학습된 모델 load => 초기 모델에 학습된 모델의 가중치 덮어씌우기
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    with torch.no_grad():
        y_preds_emotions = list()
        for data in loader:
            features = data["features"].unsqueeze(1).float().to(device)
            predictions = model(features)
        predictions = predictions[0].tolist()
    return print_test_result(predictions)


def convert_webm_to_wav(webm_path, wav_dir):
    # 원본 파일의 이름과 확장자 추출
    file_name = os.path.basename(webm_path)
    file_name_without_extension = os.path.splitext(file_name)[0]

    # wav 파일의 저장 경로 생성
    wav_file_name = f"{file_name_without_extension}.wav"
    wav_file_path = os.path.join(wav_dir, wav_file_name)

    # FFmpeg를 사용하여 webm 파일을 wav로 변환
    command = [
        "ffmpeg",
        "-i",
        webm_path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        wav_file_path,
    ]
    subprocess.run(command, check=True)

    return wav_file_path


@login_required
def recording(request):
    from .models import VoiceRecording, EmotionResult

    if request.method == "POST":
        audio_file = request.FILES.get("audio_file")
        gender = request.POST.get("gender")
        if audio_file:
            ## webm -> wav로 변환
            file_name = os.path.splitext(audio_file.name)[0]
            wav_dir = os.path.join(settings.MEDIA_ROOT, "audio")
            os.makedirs(wav_dir, exist_ok=True)  # 폴더가 없을 경우 생성
            temp_path = os.path.join(wav_dir, file_name)
            with open(temp_path, "wb") as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)

            # 변환된 wav 파일을 저장할 경로
            wav_path = convert_webm_to_wav(temp_path, wav_dir)
            # print(f"recording_wav path:{wav_path}, temp_path :{temp_path}")
            # 변환 후에는 임시 파일 삭제
            os.remove(temp_path)

            # 파일이 올바르게 첨부된 경우
            # 파일을 읽어들이고 데이터베이스에 저장
            file_path = default_storage.save(wav_path, audio_file)

            user_id = request.user.id
            recording = VoiceRecording(
                audio_file=file_path,
                gender=gender,
                uploaded_at=timezone.now(),
                user_id=user_id,
            )
            recording.save()

            # 감정 분석
            PKL_LOCATION = generate_pkl(wav_path)
            print(f"pkl_:{PKL_LOCATION}")
            test_set = Voice_dataset(pkl_location=PKL_LOCATION)
            test_loader = DataLoader(
                test_set, batch_size=len(test_set), shuffle=False, num_workers=8
            )

            print("test_set길이: " + str(len(test_set)))

            MALE_PATH = "C:\\Users\\yttn0\\Desktop\\git\\capston_web_full\\web\\polls\\pth\\2nd_best_model_epoch_55.pth"
            FEMALE_PATH = "C:\\Users\\yttn0\\Desktop\\git\\capston_web_full\\web\\polls\\pth\\2nd_best_model_epoch_26_train2.pth"

            # 초기 모델 선언 (모델 구조 저장)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CNNTransformer(num_emotions=8).to(device)

            # Test
            if recording.gender == "male":
                emotions_ratio, max_emotion = test(model, test_loader, path=MALE_PATH)
            elif recording.gender == "female":
                emotions_ratio, max_emotion = test(model, test_loader, path=FEMALE_PATH)

            print(f"recording= emotion_ratio:{emotions_ratio}")

            result = EmotionResult(emotion=max_emotion, ratio=emotions_ratio)
            result.save()

            recording.emotion_result = result
            recording.save()

            print(result.emotion)
            print(f"recording-{result.ratio}")
            print(f"recording- time-{recording.uploaded_at}")
            return JsonResponse(
                {
                    "id": recording.id,
                    "uploaded_at": recording.uploaded_at.strftime("%Y/%m/%d %H:%M"),
                    "gender": recording.gender,
                    "emotions_ratio": result.ratio,
                    "max_emotion": result.emotion,
                }
            )
    return render(request, "polls/recording.html")
