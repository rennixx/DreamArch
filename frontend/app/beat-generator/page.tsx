'use client';

import { useState } from 'react';
import PhonkStyleSelector from '@/components/PhonkStyleSelector';
import AudioPlayer from '@/components/AudioPlayer';

export const dynamic = { forceDynamic: true };

export default function BeatGeneratorPage() {
  const [selectedStyle, setSelectedStyle] = useState('montagem-batidao');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedBeat, setGeneratedBeat] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setProgress(0);
    setError(null);

    try {
      const response = await fetch('/api/beats/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ style: selectedStyle }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Beat generation failed');
      }

      const data = await response.json();
      setGeneratedBeat(data.beat_url);
      setProgress(100);

    } catch (err) {
      console.error('Generation error:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate beat');
    } finally {
      setIsGenerating(false);
    }
  };

  const getProgressMessage = () => {
    if (progress < 20) return '🎵 Loading Bark models...';
    if (progress < 40) return '🎹 Generating chunk 1...';
    if (progress < 60) return '🔊 Generating chunk 2...';
    if (progress < 80) return '🎛️ Applying effects...';
    return '✨ Finalizing...';
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-950 via-red-950 to-black text-white">
      <div className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="inline-block mb-4">
            <span className="text-6xl">🇧🇷</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-yellow-400 via-red-500 to-pink-500 bg-clip-text text-transparent">
            Brazilian Phonk Generator
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 max-w-2xl mx-auto">
            Generate authentic Phonk Brasileiro beats directly - no recording needed
          </p>
          <p className="text-sm text-gray-500 mt-2">
            150-180 BPM • Cuica drums • Heavy 808 bass • Aggressive beats
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-2xl mx-auto space-y-6">
          {/* Style Selector */}
          <div className="glass rounded-2xl p-8 border border-yellow-500/20">
            <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
              <span className="text-2xl">⚡</span>
              Choose Your Style
            </h2>
            <PhonkStyleSelector
              value={selectedStyle}
              onChange={setSelectedStyle}
              disabled={isGenerating}
            />
          </div>

          {/* Generate Button */}
          <div className="glass rounded-2xl p-8">
            <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
              <span className="text-2xl">🎵</span>
              Generate Beat
            </h2>

            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className={`
                w-full py-5 px-8 rounded-xl font-bold text-lg
                transition-all duration-300
                ${isGenerating
                  ? 'bg-gray-700 cursor-not-allowed opacity-50'
                  : 'bg-gradient-to-r from-yellow-500 via-red-500 to-pink-500 hover:scale-105 hover:shadow-2xl hover:shadow-red-500/25'
                }
              `}
            >
              {isGenerating ? (
                <span className="flex items-center justify-center gap-3">
                  <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 0 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Generating...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <span className="text-2xl">🔥</span>
                  Generate Brazilian Phonk Beat
                </span>
              )}
            </button>

            {/* Progress Bar */}
            {isGenerating && (
              <div className="mt-6">
                <div className="w-full bg-gray-900 rounded-full h-4 overflow-hidden border border-white/10">
                  <div
                    className="bg-gradient-to-r from-yellow-500 via-red-500 to-pink-500 h-4 rounded-full transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-center text-sm text-gray-400 mt-4 font-mono">
                  {getProgressMessage()}
                </p>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}

            {/* Info Box */}
            {!isGenerating && !generatedBeat && (
              <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <p className="text-yellow-300 text-sm">
                  ⚡ <strong>First generation?</strong> Bark models will download (~2GB). This only happens once.
                </p>
              </div>
            )}
          </div>

          {/* Generated Beat */}
          {generatedBeat && !isGenerating && (
            <div className="glass rounded-2xl p-8 border border-green-500/30 animate-in fade-in slide-in-from-bottom duration-500">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold">🔥 Your Beat is Ready!</h2>
                <span className="text-sm text-green-400">✓ COMPLETE</span>
              </div>

              <AudioPlayer src={generatedBeat} />

              <div className="mt-6 flex gap-3">
                <button
                  onClick={() => {
                    const a = document.createElement('a');
                    a.href = generatedBeat;
                    a.download = `brazilian-phonk-${selectedStyle}-${Date.now()}.wav`;
                    a.click();
                  }}
                  className="flex-1 py-4 px-6 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 rounded-lg font-semibold transition-all flex items-center justify-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4 4m4 4V4" />
                  </svg>
                  Download WAV
                </button>

                <button
                  onClick={() => {
                    setGeneratedBeat(null);
                    setProgress(0);
                  }}
                  className="flex-1 py-4 px-6 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold transition-colors"
                >
                  Generate Another
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-500 text-sm">
          <p>Built with Bark AI • Brazilian Phonk / Phonk Brasileiro</p>
          <p className="mt-1">No recording needed • Instant beats • 60-120 seconds</p>
        </footer>
      </div>
    </main>
  );
}
