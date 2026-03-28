import LiveSpeechPanel from "../components/LiveSpeechPanel";

function Home() {
  return (
    <div>
      <Hero />
      <Features />

      {/* Real-time communication panel */}
      <LiveSpeechPanel />
    </div>
  );
}

export default Home;