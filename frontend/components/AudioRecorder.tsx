'use client';

import { useState, useRef, useEffect } from 'react';

interface AudioRecorderProps {
  onRecordingComplete: (audioBlob: Blob) => void;
  maxDuration?: number;
  disabled?: boolean;
}

export default function AudioRecorder({
  onRecordingComplete,
  maxDuration = 3,
  disabled = false
}: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        setRecordedBlob(blob);
        onRecordingComplete(blob);

        // Create URL for playback
        if (audioUrl) {
          URL.revokeObjectURL(audioUrl);
        }
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);

        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          const newTime = Math.round((prev + 0.1) * 10) / 10;
          if (newTime >= maxDuration) {
            stopRecording();
          }
          return newTime;
        });
      }, 100);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Please allow microphone access to record.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const handleRerecord = () => {
    setRecordedBlob(null);
    setRecordingTime(0);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-6">
      {/* Recording Button */}
      <div className="relative">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={disabled || (!isRecording && !!recordedBlob)}
          className={`
            relative w-28 h-28 rounded-full transition-all duration-300
            ${isRecording
              ? 'bg-red-500 scale-110'
              : recordedBlob
                ? 'bg-green-500'
                : 'bg-blue-500 hover:bg-blue-600 hover:scale-105'
            }
            disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100
            flex items-center justify-center
            shadow-lg hover:shadow-2xl
          `}
        >
          {/* Pulsing ring animation for recording */}
          {isRecording && (
            <>
              <div className="absolute inset-0 rounded-full border-4 border-red-400 animate-ping" />
              <div className="absolute inset-0 rounded-full border-4 border-red-500 animate-pulse" />
            </>
          )}

          {/* Icon */}
          {recordedBlob && !isRecording ? (
            // Checkmark for completed recording
            <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
            </svg>
          ) : isRecording ? (
            // Stop icon
            <div className="w-10 h-10 bg-white rounded-sm" />
          ) : (
            // Microphone icon
            <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          )}
        </button>

        {/* Status indicator */}
        <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2">
          {isRecording && (
            <span className="flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
            </span>
          )}
        </div>
      </div>

      {/* Timer */}
      {isRecording && (
        <div className="text-center">
          <div className="text-3xl font-mono text-red-400 font-bold">
            {recordingTime.toFixed(1)}s
          </div>
          <div className="text-sm text-gray-400">/ {maxDuration}s</div>
        </div>
      )}

      {/* Preview & Re-record */}
      {recordedBlob && !isRecording && (
        <div className="flex flex-col items-center space-y-4 w-full">
          {/* Audio Player */}
          {audioUrl && (
            <div className="w-full max-w-xs">
              <audio
                controls
                src={audioUrl}
                className="w-full"
              />
            </div>
          )}

          {/* Re-record button */}
          <button
            onClick={handleRerecord}
            disabled={disabled}
            className="text-sm text-gray-400 hover:text-white transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Re-record
          </button>
        </div>
      )}

      {/* Instructions */}
      {!recordedBlob && !isRecording && (
        <p className="text-sm text-gray-400 text-center">
          Click to record your melody (max {maxDuration}s)
        </p>
      )}
    </div>
  );
}
