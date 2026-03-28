import { useState } from "react";
import { A11yText } from "../context/AccessibilityContext";
import { useAuth } from "../context/AuthContext";
import "./Dashboard.css";

const COURSES = [
  {
    id: 1,
    emoji: "🤝",
    title: "ISL Alphabet Basics",
    category: "Sign Language",
    level: "Beginner",
    badgeCls: "badge-violet",
    progress: 92,
    lessons: 12,
    duration: "2h 30m",
    desc: "Learn the Indian Sign Language alphabet letters with real-time feedback.",
  },
  {
    id: 2,
    emoji: "🔢",
    title: "Numbers in Sign Language",
    category: "Sign Language",
    level: "Beginner",
    badgeCls: "badge-violet",
    progress: 85,
    lessons: 8,
    duration: "1h 45m",
    desc: "Master signing numbers 1–100 for everyday communication.",
  },
  {
    id: 3,
    emoji: "📖",
    title: "Braille Introduction",
    category: "Visually Impaired",
    level: "Beginner",
    badgeCls: "badge-sky",
    progress: 70,
    lessons: 10,
    duration: "3h 00m",
    desc: "Audio-first course teaching Braille reading through sound and tactile description.",
  },
  {
    id: 4,
    emoji: "🗣️",
    title: "Speech Therapy Exercises",
    category: "Communication",
    level: "Intermediate",
    badgeCls: "badge-green",
    progress: 95,
    lessons: 15,
    duration: "4h 15m",
    desc: "AI-guided exercises to improve pronunciation, fluency, and confidence.",
  },
  {
    id: 5,
    emoji: "🎨",
    title: "Creative Arts & Expression",
    category: "Arts",
    level: "All Levels",
    badgeCls: "badge-amber",
    progress: 80,
    lessons: 20,
    duration: "5h 00m",
    desc: "Fully accessible art course adapted for motor and sensory differences.",
  },
  {
    id: 6,
    emoji: "💻",
    title: "Coding with Accessibility",
    category: "Technology",
    level: "Intermediate",
    badgeCls: "badge-sky",
    progress: 75,
    lessons: 18,
    duration: "6h 30m",
    desc: "Build accessible web apps — learn coding and a11y principles side-by-side.",
  },
  {
    id: 7,
    emoji: "🎥",
    title: "Live Video Transcriber",
    category: "Accessible Tool",
    level: "All Levels",
    badgeCls: "badge-amber",
    progress: 60,
    lessons: 1,
    duration: "Live",
    desc: "Real-time subtitles for anything spoken into the camera.",
  },
  {
    id: 8,
    emoji: "🤟",
    title: "Sign Language Interpreter",
    category: "Accessible Tool",
    level: "All Levels",
    badgeCls: "badge-green",
    progress: 65,
    lessons: 1,
    duration: "Live",
    desc: "Use your camera to translate sign language gestures into real-time text and speech."
  }
];

const LEVELS = ["All", "Beginner", "Intermediate", "All Levels"];

const COURSE_PAGE_MAP = {
  1: "alphabet",
  2: "numbers",
  3: null,
  4: "speech",
  5: "arts",
  6: "coding",
  7: "transcriber",
  8: "signdetector",   // 👈 ADD THIS
};

export default function Dashboard({ setPage }) {
  const { currentUser, logout } = useAuth();
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");

  const handleLogout = async () => {
     await logout();
     setPage('home');
  };

  const visible = COURSES.filter((c) => {
    const matchLevel = filter === "All" || c.level === filter;
    const matchSearch =
      c.title.toLowerCase().includes(search.toLowerCase()) ||
      c.category.toLowerCase().includes(search.toLowerCase());
    return matchLevel && matchSearch;
  });

  const totalProgress = Math.round(
    COURSES.reduce((s, c) => s + c.progress, 0) / COURSES.length,
  );

  return (
    <main id="main-content" className="dashboard-root fade-in-up">
      {/* Student Hub Hero */}
      <section className="dashboard-hero container">
         <div className="hero-hub">
            <div className="hub-content">
            <div className="hub-header-row">
               <div className="hub-badge">Student Dashboard</div>
               <div className="hub-header-actions">
                  <button className="btn btn-primary btn-sm" onClick={() => setPage('virtualclass')}>🎥 Join Virtual Class</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => setPage('transcriber')}>🎥 Live Transcriber</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => setPage('signdetector')}>🤟 Sign Detector</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => setPage('home')}>🏠 Home</button>
                  <button className="btn btn-ghost btn-sm" onClick={handleLogout}>🚪 Logout</button>
               </div>
            </div>
               <A11yText as="h1" className="hub-title">
                  Welcome Back, <span className="gradient-text">{currentUser?.email?.split('@')[0] || 'Learner'}</span>! ✨
               </A11yText>
               <A11yText as="p" className="hub-subtitle">
                  You've completed <strong>{totalProgress}%</strong> of your current learning journey. Ready to pick up where you left off?
               </A11yText>
               <div className="hub-stats">
                  <div className="hub-stat">
                     <span className="stat-num">8</span>
                     <span className="stat-label">Active Courses</span>
                  </div>
                  <div className="hub-stat divider"></div>
                  <div className="hub-stat">
                     <span className="stat-num">124</span>
                     <span className="stat-label">Total Lessons</span>
                  </div>
               </div>
            </div>
            
            <div className="hub-progress-card glass-card">
               <div className="dashboard__ring">
                  <svg viewBox="0 0 64 64">
                     <circle cx="32" cy="32" r="26" fill="none" stroke="rgba(0,0,0,0.05)" strokeWidth="6" />
                     <circle cx="32" cy="32" r="26" fill="none" stroke="url(#hub-grad)" strokeWidth="6" strokeLinecap="round" strokeDasharray={`${2 * Math.PI * 26}`} strokeDashoffset={`${2 * Math.PI * 26 * (1 - totalProgress / 100)}`} transform="rotate(-90 32 32)" />
                     <defs>
                        <linearGradient id="hub-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                           <stop offset="0%" stopColor="#7c6af7" />
                           <stop offset="100%" stopColor="#38bdf8" />
                        </linearGradient>
                     </defs>
                  </svg>
                  <span className="dashboard__ring-label">{totalProgress}%</span>
               </div>
               <span className="hub-overall-label">Overall Mastery</span>
            </div>
         </div>
      </section>

      <div className="container dashboard-main">
        {/* Search + Filter */}
        <div className="dashboard__controls">
          <div className="dashboard__search-wrap">
            <span aria-hidden="true">🔍</span>
            <input
              id="course-search"
              type="search"
              className="dashboard__search"
              placeholder="Search lessons or tools…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search courses"
            />
          </div>
          <div className="dashboard__filters" role="group" aria-label="Filter by level">
            {LEVELS.map((l) => (
              <button
                key={l}
                className={`btn btn-sm btn-ghost ${filter === l ? "active-filter" : ""}`}
                onClick={() => setFilter(l)}
                aria-pressed={filter === l}
              >
                {l}
              </button>
            ))}
          </div>
        </div>

        {/* Course Grid */}
        <div className="dashboard__grid" role="list">
          {visible.map((c, i) => (
            <article key={c.id} className={`course-card glass-card fade-in-up delay-${(i % 5) + 1}`}>
              <div className="course-card__top">
                <span className="course-card__emoji">{c.emoji}</span>
                <span className={`badge ${c.badgeCls}`}>{c.category}</span>
              </div>
              <A11yText as="h2" className="course-card__title">{c.title}</A11yText>
              <p className="course-card__desc">{c.desc}</p>
              
              <div className="course-card__progress">
                <div className="course-card__progress-label">
                  <span className="gradient-text">{c.progress}% done</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-bar__fill" style={{ width: `${c.progress}%` }} />
                </div>
              </div>

              <button
                className={`btn course-card__btn ${COURSE_PAGE_MAP[c.id] ? "btn-primary" : "btn-ghost"}`}
                onClick={() => COURSE_PAGE_MAP[c.id] ? setPage(COURSE_PAGE_MAP[c.id]) : setPage("dashboard")}
                disabled={!COURSE_PAGE_MAP[c.id]}
              >
                {COURSE_PAGE_MAP[c.id] ? "Continue" : "Soon"}
              </button>
            </article>
          ))}
        </div>
      </div>
    </main>
  );
}
