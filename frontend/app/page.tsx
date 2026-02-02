'use client';

import { useState } from 'react';
import AudioRecorder from '@/components/AudioRecorder';
import StyleSelector from '@/components/StyleSelector';
import AudioPlayer from '@/components/AudioPlayer';

interface StylePreset {
  id: string;
  name: string;
  description: string;
}

export default function Home() {
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [selectedStyle, setSelectedStyle] = useState('pink-floyd');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedTrack, setGeneratedTrack] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!audioBlob) {
      setError('Please record a melody first');
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setError(null);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('style', selectedStyle);

      // Send to backend
      const response = await fetch('/api/generate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Generation failed');
      }

      const data = await response.json();
      setGeneratedTrack(data.output_url);

      // Simulate progress during generation
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + 5;
        });
      }, 1000);

      // When done, set to 100%
      setProgress(100);
      clearInterval(progressInterval);

    } catch (err) {
      console.error('Generation error:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate track');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-950 via-blue-950 to-black text-white">
      <div className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-6xl md:text-7xl font-bold mb-4 gradient-text">
            Dream Architect
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 max-w-2xl mx-auto">
            Turn your hummed melodies into full songs with AI
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Optimized for fast generation (~60s)
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-2xl mx-auto space-y-6">
          {/* Step 1: Record */}
          <div className="glass rounded-2xl p-8">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-500 text-sm">1</span>
              Hum Your Melody
            </h2>
            <AudioRecorder
              onRecordingComplete={setAudioBlob}
              disabled={isGenerating}
            />
          </div>

          {/* Step 2: Choose Style */}
          <div className="glass rounded-2xl p-8">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-500 text-sm">2</span>
              Choose Style
            </h2>
            <StyleSelector
              value={selectedStyle}
              onChange={setSelectedStyle}
              disabled={isGenerating}
            />
          </div>

          {/* Step 3: Generate */}
          <div className="glass rounded-2xl p-8">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-500 text-sm">3</span>
              Generate Track
            </h2>

            <button
              onClick={handleGenerate}
              disabled={!audioBlob || isGenerating}
              className={`
                w-full py-4 px-6 rounded-lg font-semibold text-lg
                transition-all duration-300
                ${!audioBlob || isGenerating
                  ? 'bg-gray-700 cursor-not-allowed opacity-50'
                  : 'bg-gradient-to-r from-purple-500 to-pink-500 hover:scale-105 hover:shadow-lg hover:shadow-purple-500/25'
                }
              `}
            >
              {isGenerating ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Generating... (~60s)
                </span>
              ) : 'Generate Track'}
            </button>

            {/* Progress Bar */}
            {isGenerating && (
              <div className="mt-6">
                <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-center text-sm text-gray-400 mt-3">
                  {progress < 30 ? 'Analyzing melody...' :
                   progress < 50 ? 'Generating instrumental...' :
                   progress < 70 ? 'Creating vocals...' :
                   progress < 90 ? 'Mixing and mastering...' :
                   'Almost done!'}
                </p>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Generated Track */}
          {generatedTrack && !isGenerating && (
            <div className="glass rounded-2xl p-8 animate-in fade-in slide-in-from-bottom duration-500">
              <h2 className="text-2xl font-semibold mb-4">Your Track</h2>
              <AudioPlayer src={generatedTrack} />
              <button
                onClick={() => {
                  const a = document.createElement('a');
                  a.href = generatedTrack;
                  a.download = `dream-architect-${selectedStyle}-${Date.now()}.mp3`;
                  a.click();
                }}
                className="mt-4 w-full py-3 px-6 bg-green-600 hover:bg-green-700 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download MP3
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-500 text-sm">
          <p>Built with Next.js, FastAPI, and AI</p>
          <p className="mt-1">Optimized for 4GB VRAM • ~60s generation time</p>
        </footer>
      </div>
    </main>
  );
}
