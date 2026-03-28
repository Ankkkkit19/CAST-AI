import { useState } from "react";

import { A11yProvider } from "./context/AccessibilityContext";
import { AuthProvider, useAuth } from "./context/AuthContext";

import Navbar from "./components/Navbar";
import A11yWidget from "./components/A11yWidget";
import LiveSpeechPanel from "./components/LiveSpeechPanel";

import LandingPage from "./pages/LandingPage";
import Dashboard from "./pages/Dashboard";
import SignDetector from "./pages/SignDetector";
import AlphabetLearner from "./pages/AlphabetLearner";
import NumbersSignLanguage from "./pages/NumbersSignLanguage";
import SpeechTherapy from "./pages/SpeechTherapy";
import CreativeArts from "./pages/CreativeArts";
import CodingAccessibility from "./pages/CodingAccessibility";
import VideoTranscriber from "./pages/VideoTranscriber";
import VirtualClass from "./pages/VirtualClass";

import RoleSelection from "./pages/RoleSelection";
import AuthPage from "./pages/AuthPage";
import TeacherDashboard from "./pages/TeacherDashboard";

// Pages that are full-screen and manage their own layout (no shared Navbar)
const FULL_SCREEN_PAGES = new Set([
  "home", "dashboard", "transcriber", "teacherDashboard",
  "alphabet", "signdetector", "virtualclass",
]);

function MainRouter() {
  const [page, setPage] = useState("home");
  const [selectedRoleForAuth, setSelectedRoleForAuth] = useState("student");
  const { currentUser, userRole } = useAuth();

  // ── Auth flow ──────────────────────────────────────────────────────────────
  if (page === "roleselect") {
    return (
      <RoleSelection onSelectRole={(role) => {
        setSelectedRoleForAuth(role);
        setPage("auth");
      }} />
    );
  }

  if (page === "auth") {
    return (
      <AuthPage
        role={selectedRoleForAuth}
        onBack={() => setPage("roleselect")}
        onAuthSuccess={() => {
          if (selectedRoleForAuth === "teacher" || userRole === "teacher") {
            setPage("teacherDashboard");
          } else {
            setPage("dashboard");
          }
        }}
      />
    );
  }

  // ── Virtual class (full-screen, no Navbar/widgets) ─────────────────────────
  if (page === "virtualclass") {
    return (
      <VirtualClass
        setPage={setPage}
        onBack={() => setPage(userRole === "teacher" ? "teacherDashboard" : "dashboard")}
      />
    );
  }

  // ── Standard pages ─────────────────────────────────────────────────────────
  const PAGE_MAP = {
    home: LandingPage,
    dashboard: Dashboard,
    signdetector: SignDetector,
    alphabet: AlphabetLearner,
    numbers: NumbersSignLanguage,
    speech: SpeechTherapy,
    arts: CreativeArts,
    coding: CodingAccessibility,
    transcriber: VideoTranscriber,
    teacherDashboard: TeacherDashboard,
  };

  const PageComponent = PAGE_MAP[page] || LandingPage;
  const showNav = !FULL_SCREEN_PAGES.has(page);

  return (
    <>
      {showNav && <Navbar page={page} setPage={setPage} />}
      <PageComponent setPage={setPage} />
      <A11yWidget />
      <LiveSpeechPanel />
    </>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <A11yProvider>
        <MainRouter />
      </A11yProvider>
    </AuthProvider>
  );
}
