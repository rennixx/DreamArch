'use client';

import { useState, useEffect } from 'react';

interface StylePreset {
  id: string;
  name: string;
  description: string;
}

interface StyleSelectorProps {
  value: string;
  onChange: (styleId: string) => void;
  disabled?: boolean;
}

const PRESETS: StylePreset[] = [
  {
    id: 'pink-floyd',
    name: 'Pink Floyd',
    description: 'Psychedelic space rock with atmospheric guitars'
  },
  {
    id: 'daft-punk',
    name: 'Daft Punk',
    description: 'Electronic funk with vocoder and synths'
  },
  {
    id: 'billie-eilish',
    name: 'Billie Eilish',
    description: 'Dark pop with whispered vocals'
  },
  {
    id: 'frank-zappa',
    name: 'Frank Zappa',
    description: 'Experimental rock with complex arrangements'
  },
  {
    id: 'phonk',
    name: 'Phonk',
    description: '🔥 Drift phonk with cowbells and heavy 808 bass'
  },
  {
    id: 'phonk-instrumental',
    name: 'Phonk (Instrumental)',
    description: '🔥 Pure drift phonk beat - no vocals'
  }
];

export default function StyleSelector({
  value,
  onChange,
  disabled = false
}: StyleSelectorProps) {
  const [selectedPreset, setSelectedPreset] = useState<StylePreset>(
    PRESETS.find(p => p.id === value) || PRESETS[0]
  );

  useEffect(() => {
    const preset = PRESETS.find(p => p.id === value);
    if (preset) {
      setSelectedPreset(preset);
    }
  }, [value]);

  return (
    <div className="space-y-4">
      {/* Style Options Grid */}
      <div className="grid grid-cols-2 gap-3">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => {
              setSelectedPreset(preset);
              onChange(preset.id);
            }}
            disabled={disabled}
            className={`
              p-4 rounded-xl text-left transition-all duration-200
              ${selectedPreset.id === preset.id
                ? 'bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/25'
                : 'bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10'
              }
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            <div className="font-semibold text-sm md:text-base">{preset.name}</div>
            <div className="text-xs mt-1 opacity-80 line-clamp-2">{preset.description}</div>
          </button>
        ))}
      </div>

      {/* Selected Style Info */}
      {selectedPreset && (
        <div className="mt-4 p-4 bg-white/5 rounded-lg border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-green-400">✓</span>
            <span className="font-medium">Selected: {selectedPreset.name}</span>
          </div>
          <p className="text-sm text-gray-400">{selectedPreset.description}</p>
        </div>
      )}
    </div>
  );
}
