import { useState } from 'react'
import { useA11y } from '../context/AccessibilityContext'
import './A11yWidget.css'

const THEMES = [
  { id: 'dark',          label: 'Dark',          icon: '🌙' },
  { id: 'light',         label: 'Light',         icon: '☀️' },
  { id: 'high-contrast', label: 'High Contrast', icon: '⬛' },
]
const FONTS = [
  { id: 'default',  label: 'Default' },
  { id: 'dyslexic', label: 'Lexend' },
]
const SIZES = [
  { id: 'base',   label: 'A',  style: { fontSize: '0.85rem' } },
  { id: 'large',  label: 'A',  style: { fontSize: '1rem'   } },
  { id: 'xlarge', label: 'A',  style: { fontSize: '1.2rem' } },
]

export default function A11yWidget() {
  const { settings, update } = useA11y()
  const [open, setOpen] = useState(false)

  return (
    <div className={`a11y-widget ${open ? 'a11y-widget--open' : ''}`} role="complementary" aria-label="Accessibility Settings">
      <button
        id="a11y-toggle"
        className="a11y-widget__toggle"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        aria-label="Toggle accessibility settings"
        title="Accessibility"
      >
        ♿
      </button>

      {open && (
        <div className="a11y-widget__panel" role="dialog" aria-label="Accessibility panel">
          <h2 className="a11y-widget__heading">Accessibility</h2>

          {/* Theme */}
          <fieldset className="a11y-widget__group">
            <legend>Theme</legend>
            <div className="a11y-widget__row">
              {THEMES.map(t => (
                <button
                  key={t.id}
                  className={`a11y-widget__chip ${settings.theme === t.id ? 'active' : ''}`}
                  onClick={() => update('theme', t.id)}
                  aria-pressed={settings.theme === t.id}
                  title={t.label}
                >
                  <span>{t.icon}</span>
                  <span>{t.label}</span>
                </button>
              ))}
            </div>
          </fieldset>

          {/* Font */}
          <fieldset className="a11y-widget__group">
            <legend>Font</legend>
            <div className="a11y-widget__row">
              {FONTS.map(f => (
                <button
                  key={f.id}
                  className={`a11y-widget__chip ${settings.font === f.id ? 'active' : ''}`}
                  onClick={() => update('font', f.id)}
                  aria-pressed={settings.font === f.id}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </fieldset>

          {/* Text Size */}
          <fieldset className="a11y-widget__group">
            <legend>Text Size</legend>
            <div className="a11y-widget__row">
              {SIZES.map(s => (
                <button
                  key={s.id}
                  className={`a11y-widget__chip ${settings.textSize === s.id ? 'active' : ''}`}
                  onClick={() => update('textSize', s.id)}
                  style={s.style}
                  aria-pressed={settings.textSize === s.id}
                  aria-label={`Text size ${s.id}`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </fieldset>

          {/* TTS */}
          <fieldset className="a11y-widget__group">
            <legend>Text-to-Speech</legend>
            <button
              className={`a11y-widget__toggle-btn ${settings.tts ? 'on' : 'off'}`}
              onClick={() => update('tts', !settings.tts)}
              aria-pressed={settings.tts}
            >
              <span className="a11y-widget__toggle-knob" />
              <span className="a11y-widget__toggle-label">
                {settings.tts ? '🔊 On — hover text to hear it' : '🔇 Off'}
              </span>
            </button>
          </fieldset>
        </div>
      )}
    </div>
  )
}
