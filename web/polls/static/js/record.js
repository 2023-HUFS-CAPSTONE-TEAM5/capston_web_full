// class VoiceRecording {}

const $audioEl = document.getElementById("audio");
const $btn = document.getElementById("record");
const $stopBtn = document.getElementById("stop");
const $uploadBtn = document.getElementById("upload");
let isRecording = false;
const audioArray = [];
let mediaRecorder = null;

class VoiceRecording {
  constructor(id, audio_file, uploaded_at, gender) {
    this.id = id;
    this.audio_file = audio_file;
    this.uploaded_at = uploaded_at;
    this.gender = gender;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    console.log("audioArray:", audioArray);
    $uploadBtn.disabled = false;
  }
  // $audioEl에 재생할 오디오 데이터를 할당합니다.
  const blob = new Blob(audioArray, { type: "audio/webm; codecs=opus" });
  $audioEl.src = window.URL.createObjectURL(blob);
}

$btn.onclick = async function (event) {
  if (!isRecording) {
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.start();

    mediaRecorder.ondataavailable = addToAudioArray;
    mediaRecorder.onstop = stopRecording;

    isRecording = true;
    $btn.disabled = true;
    $stopBtn.disabled = false;
  } else {
    stopRecording();
  }
};

$stopBtn.onclick = function (event) {
  if (isRecording) {
    stopRecording();
    $btn.disabled = false;
    $stopBtn.disabled = true;
  }
};

function handleStop(event) {
  if (isRecording) {
    addToAudioArray(event);
    mediaRecorder.stream.getTracks().forEach((track) => track.stop());
    console.log(event);
    console.log("eventdata:");
    console.log(audioData);
    audioArray.push(audioData);
    console.log("audioArray:");
    console.log(audioArray);
    audioData = [];
  }
}

$uploadBtn.onclick = async function (event) {
  if (audioArray.length > 0) {
    const blob = new Blob(audioArray, { type: "audio/wav; codecs=opus" });
    const formData = new FormData();
    formData.append("audio_file", blob, Date.now().toString() + ".webm");

    const genderInputs = document.querySelectorAll(
      'input[name="gender"]:checked'
    );
    if (genderInputs.length > 0) {
      const gender = genderInputs[0].value;
      formData.append("gender", gender);
      console.log(gender);
    } else {
      console.error("Gender is not selected.");
      return;
    }
    try {
      const csrfToken = document.cookie.match(/csrftoken=([\w-]+)/)[1];
      const response = await fetch("/polls/recording/", {
        method: "POST",
        body: formData,
        headers: {
          "X-CSRFToken": csrfToken,
        },
      });

      if (response.ok) {
        console.log("File uploaded successfully!");
        const responseData = await response.json();
        const recording = new VoiceRecording(
          responseData.id,
          responseData.audio_file,
          response.uploaded_at,
          responseData.gender
        );
        // recordingList.push(recording);
      } else {
        console.error("File upload failed with status", response.status);
      }
    } catch (error) {
      console.error("Error uploading audio file:", error);
    }
    mediaRecorder = null;
    isRecording = false;
    audioArray.length = 0;
    $uploadBtn.disabled = true;
  }
};

function addToAudioArray(event) {
  if (event.data) {
    // event.data가 undefined가 아닐 경우에만 추가
    audioArray.push(event.data);
  }
}
