import { useCallback, useEffect, useRef, useState } from "react";

// Browser microphone capture for the AE voice journey.
//
// Records mono PCM via the Web Audio API and encodes it as a 16-bit WAV Blob.
// WAV is used deliberately: MediaRecorder's default audio/webm output is not
// accepted by the backend STT service, which supports .wav/.mp3/.ogg/.flac/
// .aac/.m4a. Recording auto-stops at `maxSeconds` (the STT limit is 60 s).

export const RECORDER_MAX_SECONDS = 60;

interface UseWavRecorderOptions {
  maxSeconds?: number;
  onStop?: (blob: Blob) => void;
}

interface UseWavRecorderResult {
  recording: boolean;
  error: string | null;
  start: () => Promise<boolean>;
  stop: () => void;
  dismissError: () => void;
}

function encodeWav(chunks: Float32Array[], sampleRate: number): Blob {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + totalLength * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + totalLength * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, totalLength * 2, true);

  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([view], { type: "audio/wav" });
}

function permissionMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "Microphone access was denied. Allow microphone access in your browser and try again.";
    }
    if (error.name === "NotFoundError" || error.name === "OverconstrainedError") {
      return "No microphone was found on this device.";
    }
  }
  return "Could not start microphone recording. Please check your input device.";
}

export function useWavRecorder({
  maxSeconds = RECORDER_MAX_SECONDS,
  onStop,
}: UseWavRecorderOptions = {}): UseWavRecorderResult {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const timerRef = useRef<number | null>(null);
  const activeRef = useRef(false);
  const onStopRef = useRef(onStop);

  onStopRef.current = onStop;

  const cleanup = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    processorRef.current?.disconnect();
    processorRef.current = null;
    gainRef.current?.disconnect();
    gainRef.current = null;
    contextRef.current?.close().catch(() => undefined);
    contextRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    activeRef.current = false;
  }, []);

  const finish = useCallback(() => {
    if (!activeRef.current) return;
    const chunks = chunksRef.current;
    const sampleRate = contextRef.current?.sampleRate ?? 48000;
    chunksRef.current = [];
    cleanup();
    setRecording(false);
    if (chunks.length > 0) {
      onStopRef.current?.(encodeWav(chunks, sampleRate));
    }
  }, [cleanup]);

  const start = useCallback(async (): Promise<boolean> => {
    if (activeRef.current) return true;
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      const AudioContextCtor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!AudioContextCtor) {
        stream.getTracks().forEach((track) => track.stop());
        setError("Audio recording is not supported in this browser.");
        return false;
      }
      const context = new AudioContextCtor();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      // Route through a muted gain node so the processor runs without
      // monitoring the microphone through the speakers.
      const gain = context.createGain();
      gain.gain.value = 0;

      chunksRef.current = [];
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));
      };

      source.connect(processor);
      processor.connect(gain);
      gain.connect(context.destination);

      contextRef.current = context;
      streamRef.current = stream;
      processorRef.current = processor;
      gainRef.current = gain;
      activeRef.current = true;
      setRecording(true);
      timerRef.current = window.setTimeout(finish, maxSeconds * 1000);
      return true;
    } catch (err) {
      cleanup();
      setError(permissionMessage(err));
      setRecording(false);
      return false;
    }
  }, [cleanup, finish, maxSeconds]);

  const stop = useCallback(() => {
    finish();
  }, [finish]);

  const dismissError = useCallback(() => setError(null), []);

  useEffect(() => cleanup, [cleanup]);

  return { recording, error, start, stop, dismissError };
}
