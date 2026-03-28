import { useState } from "react";
import { useA11y } from "../context/AccessibilityContext";
import { useAuth } from "../context/AuthContext";
import "./Navbar.css";

const NAV_LINKS = [
  { id: "home", label: "Home", icon: "🏠" },
  { id: "dashboard", label: "Courses", icon: "📚" },
  { id: "signdetector", label: "Sign Detector", icon: "🤝" },
  { id: "transcriber", label: "Video Transcribe", icon: "📽️" },
];

export default function Navbar({ page, setPage }) {
  const { settings, toggleTts } = useA11y();
  const { currentUser, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);

  const handleLogout = async () => {
    await logout();
    setPage('home');
  };

  return (
    <nav className="navbar">
      <div className="container navbar__inner">
        {/* Logo */}
        <button className="navbar__logo" onClick={() => setPage("home")}>
          <span className="navbar__logo-icon">🎯</span>
          <span className="navbar__logo-text">CAST</span>
        </button>

        {/* Desktop Links */}
        <div className="navbar__links">
          {NAV_LINKS.map((link) => (
            <button
              key={link.id}
              className={`navbar__link ${page === link.id ? "active" : ""}`}
              onClick={() => setPage(link.id)}
            >
              <span className="navbar__link-icon">{link.icon}</span>
              <span className="navbar__link-label">{link.label}</span>
            </button>
          ))}
        </div>

        {/* Right Section */}
        <div className="navbar__right">
          <button 
            className="navbar__tts-toggle" 
            onClick={toggleTts}
            aria-label={`TTS is ${settings.tts ? 'on' : 'off'}`}
          >
            <span className={`tts-dot ${settings.tts ? 'on' : 'off'}`}></span>
            {settings.tts ? 'TTS On' : 'TTS Off'}
          </button>

          {currentUser && (
            <button className="navbar__link logout-btn" onClick={handleLogout}>
               🚪 Logout
            </button>
          )}

          <button
            className="navbar__hamburger"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle menu"
          >
            <span></span>
            <span></span>
            <span></span>
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="navbar__mobile-menu">
          {NAV_LINKS.map((link) => (
            <button
              key={link.id}
              className={`navbar__mobile-link ${page === link.id ? "active" : ""}`}
              onClick={() => {
                setPage(link.id);
                setMenuOpen(false);
              }}
            >
              {link.icon} {link.label}
            </button>
          ))}
          <button
            className="navbar__mobile-link"
            onClick={() => {
              toggleTts();
              setMenuOpen(false);
            }}
          >
            {settings.tts ? "🔊 TTS On" : "🔇 TTS Off"}
          </button>
          {currentUser && (
            <button className="navbar__mobile-link" onClick={handleLogout}>
              🚪 Logout
            </button>
          )}
        </div>
      )}
    </nav>
  );
}
