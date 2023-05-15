from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render, redirect
from django.http import JsonResponse

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
import pydub

from torch.utils.data import DataLoader
import logging
import subprocess
from django.apps import apps

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


def login(request):
    template = loader.get_template("polls/login.html")
    return render(request, "polls/login.html")


def signUp(request):
    template = loader.get_template("polls/signUp.html")
    return render(request, "polls/signUP.html")


def mypage(request):
    from .models import VoiceRecording, EmotionResult

    if request.method == "POST":
        recording = VoiceRecording(
            audio_file=request.FILES["audio_file"], gender=request.POST.get("gender")
        )
        recording.save()

        emotions = EmotionResult(
            emotion=request.POST.get("max_emotion"),
            ratio=request.POST.get("emotions_ratio"),
        )
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
        context = {"recordings": recordings, "emotions": emotions}
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


def generate_pkl(INPUT_WAV_PATH):
    SAMPLE_RATE = 48000  # 1차 모델용 샘플 레이트
    DURATION = 3.0
    SPLIT_LENGTH = 3000  # 3초 단위 분할 길이

    df_path = pd.DataFrame(columns=["path"])
    df_mel = pd.DataFrame(columns=["feature"])

    audio, _ = librosa.load(
        INPUT_WAV_PATH, duration=DURATION, offset=0.0, sr=SAMPLE_RATE
    )
    audio, _ = librosa.effects.trim(audio, top_db=60)  # 묵음 처리

    # 3초 단위로 분할
    audio = (audio * 32767).astype(np.int16)  # 오디오를 16비트 정수로 변환

    # 분할된 파일들을 저장할 디렉토리 생성
    OUTPUT_DIR = os.path.join("media", "audio", "split")  # 3초 단위로 잘린 파일들이 저장될 폴더 경로

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(audio[::SPLIT_LENGTH])
    for i in range(0, len(audio), SPLIT_LENGTH):
        chunk = audio[i : i + SPLIT_LENGTH]
        print(chunk)
        OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"splited_audio{i}.wav")
        pydub.AudioSegment(
            chunk.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1
        ).export(OUTPUT_PATH, format="wav")
        df_path.loc[i] = OUTPUT_PATH

        temp_audio = np.zeros((int(SAMPLE_RATE * DURATION),))
        temp_audio[: len(chunk)] = librosa.load(OUTPUT_PATH, sr=SAMPLE_RATE)[
            0
        ]  # OUTPUT_PATH에서 음성 데이터 로드하여 temp_audio에 할당

        mel = MELSpectrogram(temp_audio, sample_rate=SAMPLE_RATE)
        df_mel.loc[i] = [mel]
    print(SPLIT_LENGTH)

    df = pd.concat([df_path, df_mel], axis=1)

    # .pkl(test 데이터) 저장할 경로 설정
    PKL_PATH = os.path.join("media", "audio", "test")

    if not os.path.exists(PKL_PATH):
        os.makedirs(PKL_PATH)

    df.to_pickle(os.path.join(PKL_PATH, "test.pkl"))
    PKL_LOCATION = os.path.join(PKL_PATH, "test.pkl")
    print("여기까진 작동 잘됨")
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
        voice_labels = self.df.iloc[idx, 0].split("\\")[-1].split(".")[0].split("-")
        print(f"voice_labels = {len(voice_labels)}")
        voice["emotion"] = int(voice_labels[2]) - 1
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
def print_test_result(emotions_dict):
    neutral = 0

    for i in range(8):
        emotion = i + 1
        if emotion not in emotions_dict:
            emotions_dict[emotion] = 0
        if emotion == 1 or emotion == 2:
            neutral += emotions_dict[emotion]
        elif emotion == 3:
            happy = emotions_dict[emotion]
        elif emotion == 4:
            sad = emotions_dict[emotion]
        elif emotion == 5:
            angry = emotions_dict[emotion]
        elif emotion == 6:
            fearful = emotions_dict[emotion]
        elif emotion == 7:
            disgust = emotions_dict[emotion]
        elif emotion == 8:
            surprised = emotions_dict[emotion]

    total_count = {
        "neutral": neutral,
        "happy": happy,
        "sad": sad,
        "angry": angry,
        "fearful": fearful,
        "disgust": disgust,
        "surprised": surprised,
    }
    total = sum(total_count.values())

    emotion_ratio = {}
    for emotion in total_count.keys():
        if total_count[emotion] == 0:
            print("print_test-----emotion 빈값")
        else:
            emotion_ratio[emotion] = round((total_count[emotion] / total) * 100, 2)
            print(f"{emotion} : {(total_count[emotion] / total) * 100:.2f}%")

    max_emotion = max(total_count, key=total_count.get)
    # print(f'가장 큰 비율을 차지하고 있는 감정은 "{max_emotion}" 입니다.')

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

            # predictions
            predictions = model(features)
            y_preds_emotions.append(torch.argmax(predictions, dim=1))

        if len(y_preds_emotions) > 0:
            Y_Preds_Emotions = torch.cat(y_preds_emotions, dim=0)
        else:
            # 처리할 데이터가 없는 경우에 대한 처리
            print("test함수--- 데이터가 없음")
            Y_Preds_Emotions = torch.tensor([])  # 빈 텐서 생성

        emotions = Y_Preds_Emotions.tolist()
        emotions_dict = dict(Counter(emotions))

    return print_test_result(emotions_dict)


def convert_webm_to_wav(webm_path, wav_dir):
    # 원본 파일의 이름과 확장자 추출
    file_name = os.path.basename(webm_path)
    file_name_without_extension = os.path.splitext(file_name)[0]

    # wav 파일의 저장 경로 생성
    wav_file_name = f"{file_name_without_extension}.wav"
    wav_file_path = os.path.join(wav_dir, wav_file_name)

    # ffmpeg를 사용하여 webm 파일을 wav로 변환
    command = f'ffmpeg -i "{webm_path}" -acodec pcm_s16le -ar 16000 "{wav_file_path}"'
    subprocess.call(command, shell=True)

    return wav_file_path


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

            # 변환 후에는 임시 파일 삭제
            os.remove(temp_path)

            print(wav_path)

            # 파일이 올바르게 첨부된 경우
            # 파일을 읽어들이고 데이터베이스에 저장
            file_name = os.path.splitext(audio_file.name)[0] + ".wav"
            file_path = os.path.join(wav_dir, file_name)
            file_name = default_storage.save(file_path, ContentFile(audio_file.read()))
            recording = VoiceRecording(audio_file=file_name, gender=gender)
            recording.save()

            # 감정 분석
            PKL_LOCATION = generate_pkl(wav_path)
            print(PKL_LOCATION)
            test_set = Voice_dataset(pkl_location=PKL_LOCATION)
            # 데이터셋의 길이
            dataset_length = len(test_set)

            print("test_set길이: " + str(len(test_set)))
            # 원하는 배치 크기
            desired_batch_size = 32

            # 실제 배치 크기 계산
            batch_size = min(desired_batch_size, dataset_length)

            # 배치 크기가 0보다 작을 경우, 기본값으로 1 설정
            if batch_size < 1:
                batch_size = 1

            # DataLoader 초기화
            test_loader = DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=8
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

            return JsonResponse(
                {
                    "id": recording.id,
                    "uploaded_at": recording.uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "gender": recording.gender,
                    "emotions_ratio": emotions_ratio,
                    "max_emotion": max_emotion,
                }
            )
    return render(request, "polls/recording.html")
