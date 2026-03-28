import { useState } from 'react'
import { useA11y } from '../context/AccessibilityContext'
import './CoursePage.css'

const LESSONS = [
  { id: 1, title: 'What is Web Accessibility?', content: 'Web accessibility (a11y) means making websites usable by everyone — including people with disabilities. Screen readers, keyboard navigation, and clear contrast help all users.', code: null },
  { id: 2, title: 'Semantic HTML', content: 'Use <header>, <nav>, <main>, <article>, <footer> instead of only <div>. Screen readers use these to understand your page structure.', code: '<main>\n  <article>\n    <h1>Title</h1>\n    <p>Content</p>\n  </article>\n</main>' },
  { id: 3, title: 'Alt Text for Images', content: 'Every <img> needs an alt attribute. Describe what the image shows so screen reader users understand it.', code: '<img src="photo.jpg" alt="A child signing the letter A in ISL">' },
  { id: 4, title: 'Keyboard Navigation', content: 'Make sure users can tab through all interactive elements. Use :focus-visible to show focus clearly.', code: ':focus-visible {\n  outline: 2px solid blue;\n  outline-offset: 2px;\n}' },
  { id: 5, title: 'Color Contrast', content: 'Text must have enough contrast against the background. Aim for at least 4.5:1 for normal text.', code: '/* Good: dark text on light bg */\ncolor: #0f172a;\nbackground: #fff;' },
]

export default function CodingAccessibility({ setPage }) {
  const { settings } = useA11y()
  const [current, setCurrent] = useState(0)
  const lesson = LESSONS[current]

  const speak = (text) => {
    if (!settings.tts || !window.speechSynthesis) return
    window.speechSynthesis.cancel()
    const utt = new SpeechSynthesisUtterance(text)
    utt.rate = 0.9
    window.speechSynthesis.speak(utt)
  }

  return (
    <main id="main-content" className="course-page">
      <div className="container">
        <div className="course-page__header fade-in-up">
          <div>
            <h1 className="course-page__title">💻 Coding with Accessibility</h1>
            <p className="course-page__subtitle">Build accessible web apps — learn coding and a11y principles</p>
          </div>
          <button className="btn btn-ghost" onClick={() => setPage('dashboard')}>← Back to Courses</button>
        </div>

        <div className="course-page__main course-page__main--single">
          <div className="course-page__content glass-card fade-in-up delay-1" style={{ maxWidth: '700px', margin: '0 auto' }}>
            <h2 className="course-page__exercise-title">{lesson.title}</h2>
            <p className="course-page__exercise-desc">{lesson.content}</p>
            {lesson.code && (
              <pre className="course-page__code"><code>{lesson.code}</code></pre>
            )}
            <div className="course-page__exercise-actions">
              <button className="btn btn-primary" onClick={() => speak(lesson.content)}>
                🔊 Hear Lesson
              </button>
              <div className="course-page__nav-btns">
                <button className="btn btn-ghost" disabled={current === 0} onClick={() => setCurrent(c => c - 1)}>← Previous</button>
                <button className="btn btn-primary" disabled={current === LESSONS.length - 1} onClick={() => setCurrent(c => c + 1)}>Next →</button>
              </div>
            </div>
          </div>
        </div>

        <div className="course-page__pills fade-in-up delay-2">
          {LESSONS.map((l, i) => (
            <button
              key={l.id}
              className={`btn btn-sm ${current === i ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setCurrent(i)}
            >
              {i + 1}. {l.title}
            </button>
          ))}
        </div>
      </div>
    </main>
  )
}
