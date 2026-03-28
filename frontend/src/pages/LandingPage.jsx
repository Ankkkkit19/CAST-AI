import { A11yText } from "../context/AccessibilityContext";
import "./LandingPage.css";

export default function LandingPage({ setPage }) {
  return (
    <main className="landing-new">
      <div className="landing-new__container">
        <div className="landing-new__header">
          <A11yText as="h1" className="landing-new__title">
            CAST
          </A11yText>

          <A11yText as="p" className="landing-new__subtitle">
            Speak · Sign · Connect
          </A11yText>
        </div>

        <div className="landing-new__graphic">
          <div className="landing-new__center-circle">👦🏽</div>

          <div className="landing-new__float-icon icon-video">🎥</div>
          <div className="landing-new__float-icon icon-audio">🔊</div>
          <div className="landing-new__float-icon icon-mic">🎙️</div>

          <div className="landing-new__decoration-ring"></div>
        </div>

        <div className="landing-new__description">
          <A11yText as="p">
            Welcome to a fun and magical world where everyone can learn to
            communicate. Discover inclusive games, AI tools and learning
            experiences built to bridge communication gaps.
          </A11yText>
        </div>

        <button
          className="btn-get-started"
          onClick={() => setPage("roleselect")}
        >
          🚀 Get Started
        </button>
      </div>

      <div className="landing-new__bg-shape shape-pink"></div>
      <div className="landing-new__bg-shape shape-yellow"></div>
      <div className="landing-new__bg-shape shape-cyan"></div>
    </main>
  );
}
