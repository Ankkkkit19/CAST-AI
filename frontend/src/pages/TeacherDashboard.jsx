import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { subscribeToAllProgress } from '../services/progressService';
import './TeacherDashboard.css';


export default function TeacherDashboard({ setPage }) {
  const { logout } = useAuth();
  const [students, setStudents] = useState([]);
  const [activeTab, setActiveTab] = useState('students'); // 'students' | 'meeting'
  // const roomName = "VardaanInclusiveClassroom_101";

  // Real-time Firestore listener for all student progress
  useEffect(() => {
    const unsub = subscribeToAllProgress((data) => {
      setStudents(data);
    });
    return () => unsub();
  }, []);

  const handleLogout = async () => {
    await logout();
    setPage('home');
  };

  const totalStudents = students.length;
  const activeStudents = students.filter(s =>
    s.alphabets?.lastUpdated &&
    (Date.now() - new Date(s.alphabets.lastUpdated).getTime()) < 30 * 60 * 1000
  ).length;

  return (
    <main className="teacher-root">
      {/* Sidebar */}
      <aside className="teacher-sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">🎯</span>
          <span className="logo-text">CAST</span>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`sidebar-btn ${activeTab === 'students' ? 'active' : ''}`}
            onClick={() => setActiveTab('students')}
          >
            <span>📊</span> Live Progress
          </button>
          <button
            className={`sidebar-btn ${activeTab === 'meeting' ? 'active' : ''}`}
            onClick={() => setActiveTab('meeting')}
          >
            <span>🎥</span> Virtual Class
          </button>
          <button className="sidebar-btn" onClick={() => setPage('home')}>
            <span>🏠</span> Home
          </button>
        </nav>

        <button className="sidebar-logout" onClick={handleLogout}>
          🚪 Log Out
        </button>
      </aside>

      {/* Main Content */}
      <div className="teacher-content">
        {activeTab === 'students' && (
          <div className="teacher-panel fade-in-up">
            {/* Stats Row */}
            <div className="teacher-stats-row">
              <div className="stat-chip">
                <span className="stat-num">{totalStudents}</span>
                <span className="stat-lbl">Total Students</span>
              </div>
              <div className="stat-chip green">
                <span className="stat-num">{activeStudents}</span>
                <span className="stat-lbl">Active (30 min)</span>
              </div>
              <div className="stat-chip blue">
                <span className="stat-num">2</span>
                <span className="stat-lbl">Courses Tracked</span>
              </div>
            </div>

            <h2 className="panel-title">📊 Real-Time Student Progress</h2>

            {students.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">👩‍🎓</div>
                <h3>Waiting for students...</h3>
                <p>Progress will appear here when students open their lessons.</p>
              </div>
            ) : (
              <div className="students-grid">
                {students.map((student) => {
                  const alphaPct = student.alphabets?.percent ?? 0;
                  const numPct = student.numbers?.percent ?? 0;
                  const name = student.email?.split('@')[0] || student.id.slice(0, 8);
                  const lastSeen = student.alphabets?.lastUpdated || student.numbers?.lastUpdated;
                  const isRecent = lastSeen && (Date.now() - new Date(lastSeen).getTime()) < 30 * 60 * 1000;

                  return (
                    <div key={student.id} className="student-card glass-card">
                      <div className="student-header">
                        <div className="student-avatar">
                          {name[0]?.toUpperCase()}
                        </div>
                        <div className="student-info">
                          <h3 className="student-name">{name}</h3>
                          <span className={`status-dot ${isRecent ? 'online' : 'offline'}`}>
                            {isRecent ? '● Active' : '○ Inactive'}
                          </span>
                        </div>
                      </div>

                      {/* ISL Alphabets Progress */}
                      <div className="progress-section">
                        <div className="progress-label-row">
                          <span>🔤 ISL Alphabets</span>
                          <span className="pct-badge">{alphaPct}%</span>
                        </div>
                        <div className="progress-bar">
                          <div className="progress-bar__fill" style={{ width: `${alphaPct}%` }} />
                        </div>
                        {student.alphabets?.currentLetter && (
                          <p className="progress-detail">
                            Currently on: <strong>{student.alphabets.currentLetter}</strong> · {student.alphabets.completed?.length ?? 0}/26 letters
                          </p>
                        )}
                      </div>

                      {/* Numbers Progress */}
                      <div className="progress-section">
                        <div className="progress-label-row">
                          <span>🔢 Numbers (1-20)</span>
                          <span className="pct-badge">{numPct}%</span>
                        </div>
                        <div className="progress-bar">
                          <div className="progress-bar__fill numbers" style={{ width: `${numPct}%` }} />
                        </div>
                        {student.numbers?.currentNumber && (
                          <p className="progress-detail">
                            Currently on: <strong>Number {student.numbers.currentNumber}</strong> · {student.numbers.completedCount ?? 0}/20 done
                          </p>
                        )}
                      </div>

                      {lastSeen && (
                        <p className="last-seen">
                          Last active: {new Date(lastSeen).toLocaleTimeString()}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {activeTab === 'meeting' && (
          <div className="teacher-panel fade-in-up">
            <h2 className="panel-title">🎥 Live Virtual Classroom</h2>
            <p className="panel-sub">Host accessible live classes with real-time sign broadcasting to students.</p>
            <div className="meeting-placeholder glass-card">
              <div className="meeting-icon">🎓</div>
              <h3>Ready to teach?</h3>
              <p>Click below to open the full virtual classroom where you can broadcast signs live to all connected students.</p>
              <button className="btn btn-primary btn-lg" onClick={() => setPage('virtualclass')}>
                🚀 Start Class Now
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
