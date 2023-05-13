// 엘리먼트 취득
const $audioEl = document.querySelector("audio");
const $btn = document.getElementById("record");
const $uploadBtn = document.getElementById("upload");

// 녹음중 상태 변수
let isRecording = false;

// MediaRecorder 변수 생성
let mediaRecorder = null;

// 녹음 데이터 저장 배열
const audioArray = [];

$btn.onclick = async function (event) {
  if (!isRecording) {
    // 마이크 mediaStream 생성: Promise를 반환하므로 async/await 사용
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    // MediaRecorder 생성 (mimeType 지정)
    mediaRecorder = new MediaRecorder(mediaStream, {
      mimeType: "audio/webm;codecs=opus",
    });

    // 이벤트핸들러: 녹음 데이터 취득 처리
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioArray.push(event.data); // 오디오 데이터가 취득될 때마다 배열에 담아둔다.
        $uploadBtn.disabled = false; // 녹음이 되어 데이터가 있을 경우, 파일 업로드 버튼 활성화
      }
    };

    // 녹음 시작 (timeslice 매개변수를 전달하여 일정 간격으로 녹음 데이터를 저장)
    mediaRecorder.start(100);
    isRecording = true;
  } else {
    // 녹음 종료
    mediaRecorder.stop();
    isRecording = false;
    $uploadBtn.disabled = true; // 녹음 종료 후 파일 업로드 버튼 비활성화
  }
};

// 파일 업로드 버튼 클릭 이벤트 핸들러
$uploadBtn.onclick = async function (event) {
  const blob = new Blob(audioArray, { type: "audio/webm;codecs=opus" });

  // Blob 데이터에 접근할 수 있는 주소를 생성한다.
  const blobURL = window.URL.createObjectURL(blob);

  // 서버로 업로드
  const formData = new FormData();
  formData.append("audio", blob, "recorded_audio.webm");

  const genderInputs = document.querySelectorAll(
    'input[name="gender"]:checked'
  );
  if (genderInputs.length > 0) {
    const gender = genderInputs[0].value === "female" ? "female" : "male";
    formData.append("gender", gender);
  } else {
    formData.append("gender", ""); // 선택된 gender가 없을 경우 빈 문자열로 추가
  }

  try {
    const response = await fetch("/recording", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      console.log("Audio file uploaded successfully!");
    } else {
      console.error("Error uploading audio file:", response.status);
    }
  } catch (error) {
    console.error("Error uploading audio file:", error);
  }

  // MediaRecorder 초기화
  mediaRecorder = null;
  isRecording = false;
  audioArray.length = 0;
  $uploadBtn.disabled = true; // 파일 업로드 후 버튼 비활성화
};
