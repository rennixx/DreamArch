'use client';

import { useState, useEffect } from 'react';

interface PhonkPreset {
  id: string;
  name: string;
  description: string;
  bpm: number;
  energy: 'high' | 'extreme' | 'maximum';
}

interface PhonkStyleSelectorProps {
  value: string;
  onChange: (styleId: string) => void;
  disabled?: boolean;
}

// Brazilian Phonk presets
const PRESETS: PhonkPreset[] = [
  {
    id: 'montagem-batidao',
    name: 'Montagem Batidão',
    description: '🔥 160 BPM heavy phonk, aggressive cuica, pounding 808 bass, festival anthem',
    bpm: 160,
    energy: 'extreme'
  },
  {
    id: 'rave-lofi',
    name: 'Rave Lofi',
    description: '⚡ 170 BPM psy-phonk, lofi aesthetic, fast cowbell patterns, rave atmosphere',
    bpm: 170,
    energy: 'maximum'
  },
  {
    id: 'mega-slap',
    name: 'Mega Slap',
    description: '💥 180 BPM maximum slap, face-melting bass drops, cuica madness, pure energy',
    bpm: 180,
    energy: 'maximum'
  },
  {
    id: 'funk-eletronico',
    name: 'Funk Eletrônico',
    description: '🎹 150 BPM electronic funk, modern Brazilian Phonk, synth melodies, club-ready',
    bpm: 150,
    energy: 'high'
  }
];

export default function PhonkStyleSelector({
  value,
  onChange,
  disabled = false
}: PhonkStyleSelectorProps) {
  const [selectedPreset, setSelectedPreset] = useState<PhonkPreset>(
    PRESETS.find(p => p.id === value) || PRESETS[0]
  );

  useEffect(() => {
    const preset = PRESETS.find(p => p.id === value);
    if (preset) {
      setSelectedPreset(preset);
    }
  }, [value]);

  const getEnergyColor = (energy: string) => {
    switch (energy) {
      case 'high': return 'from-orange-500 to-red-500';
      case 'extreme': return 'from-red-500 to-pink-500';
      case 'maximum': return 'from-pink-500 to-purple-500';
      default: return 'from-purple-500 to-pink-500';
    }
  };

  const getEnergyLabel = (energy: string) => {
    switch (energy) {
      case 'high': return 'HIGH ENERGY';
      case 'extreme': return 'EXTREME';
      case 'maximum': return 'MAXIMUM';
      default: return energy.toUpperCase();
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-bold bg-gradient-to-r from-yellow-400 via-red-500 to-pink-500 bg-clip-text text-transparent">
          BRAZILIAN PHONK PRESETS
        </h3>
        <p className="text-sm text-gray-400 mt-2">
          Select your style • 150-180 BPM • Aggressive beats
        </p>
      </div>

      {/* Style Options Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => {
              setSelectedPreset(preset);
              onChange(preset.id);
            }}
            disabled={disabled}
            className={`
              relative p-5 rounded-xl text-left transition-all duration-200
              border-2
              ${selectedPreset.id === preset.id
                ? `bg-gradient-to-br ${getEnergyColor(preset.energy)} border-transparent scale-105 shadow-lg`
                : 'bg-black/40 border-white/10 hover:border-white/30 hover:bg-white/5'
              }
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {/* Energy Badge */}
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-lg">{preset.name}</span>
              <span className={`
                text-xs px-2 py-1 rounded-full font-bold
                ${preset.energy === 'maximum' ? 'bg-purple-600' :
                  preset.energy === 'extreme' ? 'bg-red-600' : 'bg-orange-600'}
              `}>
                {getEnergyLabel(preset.energy)}
              </span>
            </div>

            {/* BPM Badge */}
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xs px-2 py-1 bg-white/10 rounded">
                {preset.bpm} BPM
              </span>
            </div>

            {/* Description */}
            <p className="text-sm text-gray-300 leading-relaxed">
              {preset.description}
            </p>

            {/* Selected Indicator */}
            {selectedPreset.id === preset.id && (
              <div className="absolute top-2 right-2">
                <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Selected Style Info */}
      {selectedPreset && (
        <div className="mt-4 p-4 bg-gradient-to-r from-yellow-500/10 to-red-500/10 rounded-lg border border-yellow-500/30">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-yellow-400">⚡</span>
            <span className="font-medium">Selected: {selectedPreset.name}</span>
            <span className="text-xs text-gray-400 ml-auto">
              {selectedPreset.bpm} BPM • {getEnergyLabel(selectedPreset.energy)}
            </span>
          </div>
          <p className="text-sm text-gray-300">{selectedPreset.description}</p>
        </div>
      )}
    </div>
  );
}
