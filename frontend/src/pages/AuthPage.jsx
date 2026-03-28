import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { A11yText } from '../context/AccessibilityContext';
import './AuthPage.css';

export default function AuthPage({ role, onBack, onAuthSuccess }) {
  const { login, signup } = useAuth();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) return;
    setLoading(true);
    setError(null);
    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await signup(email, password, role);
      }
      onAuthSuccess();
    } catch (err) {
      console.error(err);
      setError(err.message || 'Authentication failed. Make sure your Firebase/Mock config is correct.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page fade-in-up">
      <div className="auth-page__container glass-card">
        <button className="btn btn-ghost auth-page__back" onClick={onBack}>
          ← Go Back
        </button>

        <div className="auth-page__header">
          <A11yText as="h2" className="auth-page__title">
            {isLogin ? 'Welcome Back!' : 'Join CAST today'}
          </A11yText>
          <A11yText as="p" className="auth-page__subtitle">
            {role === 'student' ? '🎒 Continuing your Learning Journey' : '🎓 Ready to teach.'}
          </A11yText>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {error && <div className="auth-form__error">{error}</div>}
          
          <div className="auth-form__group">
            <label htmlFor="auth-email">Email Address</label>
            <input 
              id="auth-email" 
              type="email" 
              value={email} 
              onChange={e => setEmail(e.target.value)} 
              placeholder="hello@example.com" 
              required 
            />
          </div>

          <div className="auth-form__group">
            <label htmlFor="auth-password">Password</label>
            <input 
              id="auth-password" 
              type="password" 
              value={password} 
              onChange={e => setPassword(e.target.value)} 
              placeholder="••••••••" 
              required 
              minLength="6"
            />
          </div>

          <button className="btn btn-primary auth-submit" disabled={loading} type="submit">
            {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <div className="auth-page__toggle">
          <p>
            {isLogin ? "Don't have an account?" : "Already have an account?"}
            <button className="btn-link" onClick={() => setIsLogin(!isLogin)} type="button">
              {isLogin ? 'Sign Up' : 'Log In'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
