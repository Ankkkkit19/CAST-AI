import React from 'react';
import './RoleSelection.css'; // Let's reuse some landing page styles if we can, or make it inline/new css
import { A11yText } from '../context/AccessibilityContext';

export default function RoleSelection({ onSelectRole }) {
  return (
    <div className="role-selection fade-in-up">
      <div className="role-selection__container glass-card">
        <A11yText as="h2" className="role-selection__title">
          Welcome to CAST
        </A11yText>
        <A11yText as="p" className="role-selection__subtitle">
          How would you like to use the platform today?
        </A11yText>

        <div className="role-selection__cards">
          <button 
            className="role-card" 
            onClick={() => onSelectRole('student')}
            aria-label="Join as a Learner"
          >
            <div className="role-card__emoji">🎒</div>
            <A11yText as="h3">I am a Learner</A11yText>
            <A11yText as="p">Explore courses, master ISL, and practice communication.</A11yText>
          </button>

          <button 
            className="role-card" 
            onClick={() => onSelectRole('teacher')}
            aria-label="Join as an Educator"
          >
            <div className="role-card__emoji">🎓</div>
            <A11yText as="h3">I am an Educator</A11yText>
            <A11yText as="p">Host accessible live classes, track progress, and teach.</A11yText>
          </button>
        </div>
      </div>
    </div>
  );
}
