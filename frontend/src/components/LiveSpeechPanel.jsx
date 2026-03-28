import React, { useState, useRef } from "react";

const LiveSpeechPanel = () => {
  const [transcript, setTranscript] = useState("");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const videoRef = useRef(null);
  const recognitionRef = useRef(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraOn(true);
      }
    } catch (err) {
      console.error("Camera access denied:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraOn(false);
    }
  };

  const startListening = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true; // real time updates
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let liveTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        liveTranscript += event.results[i][0].transcript;
      }

      setTranscript(liveTranscript);
    };

    recognition.start();
    recognitionRef.current = recognition;
  };

  const speakText = () => {
    const speech = new SpeechSynthesisUtterance(transcript);
    window.speechSynthesis.speak(speech);
  };

  return (
    <div style={styles.container}>
      <h2>Live Speech Communication</h2>

      <div style={styles.panel}>
        {/* Video */}
        <div style={styles.videoBox}>
          <video ref={videoRef} autoPlay muted style={{...styles.video, display: isCameraOn ? 'block' : 'none'}} />
          {!isCameraOn && (
            <div style={{...styles.video, background: '#ccc', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '200px'}}>
              <p>Camera is off</p>
            </div>
          )}
        </div>

        {/* Live Text */}
        <div style={styles.textBox}>
          <h3>Live Subtitles</h3>
          <p style={styles.text}>{transcript || "Waiting for speech..."}</p>
        </div>
      </div>

      <div style={styles.buttons}>
        {!isCameraOn ? (
          <button onClick={startCamera}>📷 Turn On Camera</button>
        ) : (
          <button onClick={stopCamera}>🛑 Turn Off Camera</button>
        )}
        <button onClick={startListening}>🎤 Start Listening</button>
        <button onClick={speakText}>🔊 Speak</button>
      </div>
    </div>
  );
};

const styles = {
  container: {
    marginTop: "50px",
    padding: "30px",
    background: "#f0f4f8",
    borderRadius: "12px",
  },

  panel: {
    display: "flex",
    gap: "30px",
    marginTop: "20px",
  },

  videoBox: {
    flex: 1,
  },

  textBox: {
    flex: 1,
    background: "white",
    padding: "20px",
    borderRadius: "10px",
    minHeight: "200px",
  },

  video: {
    width: "100%",
    borderRadius: "10px",
  },

  text: {
    fontSize: "22px",
    lineHeight: "1.6",
  },

  buttons: {
    marginTop: "20px",
    display: "flex",
    gap: "20px",
  },
};

export default LiveSpeechPanel;