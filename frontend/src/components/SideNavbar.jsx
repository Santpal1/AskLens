import React, { useState, useEffect, useRef, createContext, useContext } from "react";
import "./SideNavbar.css";

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return [h, m, s]
    .map((v) => v.toString().padStart(2, "0"))
    .join(":");
}

function getTodayDateStr() {
  const d = new Date();
  return d.toISOString().slice(0, 10);
}

// Timer context for sharing timer state
export const TimerContext = createContext();

const SideNavbar = ({ onSelectTab, selectedTab }) => {
  return (
    <div className="side-navbar">
      <button
        className={selectedTab === "home" ? "active" : ""}
        onClick={() => onSelectTab("home")}
      >
        🏠 Home
      </button>
      <button
        className={selectedTab === "main" ? "active" : ""}
        onClick={() => onSelectTab("main")}
      >
        📄 App
      </button>
    </div>
  );
};

const MOTIVATIONAL_QUOTES = [
  "Stay focused and never give up!",
  "Small steps every day lead to big results.",
  "Discipline is the bridge between goals and accomplishment.",
  "You are capable of amazing things.",
  "Push yourself, because no one else is going to do it for you.",
  "Success is the sum of small efforts repeated day in and day out.",
];

function getRandomQuote() {
  return MOTIVATIONAL_QUOTES[Math.floor(Math.random() * MOTIVATIONAL_QUOTES.length)];
}

export function HomePage() {
  const {
    studySeconds,
    timer,
    timerRunning,
    handleSetTimer,
    handleStop,
    handleResetStudy,
  } = useContext(TimerContext);

  // Weekly summary mock (replace with real data if available)
  const [weekData, setWeekData] = useState(() => {
    // Try to load from localStorage, else mock
    const saved = localStorage.getItem("weekStudyData");
    if (saved) return JSON.parse(saved);
    const todayIdx = new Date().getDay();
    const arr = Array(7).fill(0);
    arr[todayIdx] = studySeconds;
    return arr;
  });

  useEffect(() => {
    // Update today's value in weekData
    const idx = new Date().getDay();
    setWeekData((prev) => {
      const arr = [...prev];
      arr[idx] = studySeconds;
      localStorage.setItem("weekStudyData", JSON.stringify(arr));
      return arr;
    });
  }, [studySeconds]);

  // Progress bar for a daily goal (e.g., 2 hours)
  const dailyGoal = 2 * 60 * 60;
  const progress = Math.min(1, studySeconds / dailyGoal);

  // Quote
  const [quote] = useState(getRandomQuote());

  return (
    <div className="homepage">
      <h2>📅 Study Dashboard</h2>
      <div className="study-summary-section">
        <div className="study-time-today">
          <span className="summary-label">Today's Study Time</span>
          <div className="study-time-main">{formatTime(studySeconds)}</div>
          <div className="progress-bar-bg">
            <div
              className="progress-bar-fg"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
          <div className="progress-label">
            {Math.floor((studySeconds / dailyGoal) * 100)}% of daily goal ({formatTime(dailyGoal)})
          </div>
        </div>
        <button onClick={handleResetStudy} className="reset-study-btn">
          Reset Today's Study
        </button>
      </div>
      <div className="motivation-section">
        <div className="motivation-quote">
          <span role="img" aria-label="sparkles">✨</span> <em>{quote}</em>
        </div>
      </div>
      <hr />
      <div className="timer-section">
        <form onSubmit={handleSetTimer} style={{ marginTop: 10 }}>
          <label>
            Set Study Timer (minutes):{" "}
            <input type="number" name="minutes" min="1" required disabled={timerRunning} />
          </label>
          <button type="submit" disabled={timerRunning}>
            Start Timer
          </button>
        </form>
        {timer > 0 && (
          <div style={{ marginTop: 10 }}>
            <strong>Timer:</strong> {formatTime(timer)}
            <button onClick={handleStop} style={{ marginLeft: 10 }}>
              Stop
            </button>
          </div>
        )}
      </div>
      <hr />
      <div className="weekly-summary-section">
        <h3>📊 Weekly Study Summary</h3>
        <div className="weekly-bars">
          {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((d, i) => (
            <div key={d} className="weekly-bar-item">
              <div
                className="weekly-bar"
                style={{
                  height: `${Math.min(100, (weekData[i] / dailyGoal) * 100)}%`,
                  background: i === new Date().getDay() ? "#667eea" : "#c8d6f5",
                }}
                title={`${d}: ${formatTime(weekData[i] || 0)}`}
              />
              <div className="weekly-bar-label">{d}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// TimerProvider to wrap App and provide timer state
export function TimerProvider({ children }) {
  const [studySeconds, setStudySeconds] = useState(
    Number(localStorage.getItem("studySeconds") || 0)
  );
  const [timer, setTimer] = useState(0);
  const [timerRunning, setTimerRunning] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    localStorage.setItem("studySeconds", studySeconds);
  }, [studySeconds]);

  useEffect(() => {
    if (timerRunning && timer > 0) {
      intervalRef.current = setInterval(() => {
        setTimer((t) => {
          if (t <= 1) {
            setTimerRunning(false);
            clearInterval(intervalRef.current);
            setStudySeconds((s) => s + 1);
            return 0;
          }
          setStudySeconds((s) => s + 1);
          return t - 1;
        });
      }, 1000);
    }
    return () => clearInterval(intervalRef.current);
  }, [timerRunning, timer]);

  const handleSetTimer = (e) => {
    e.preventDefault();
    const mins = Number(e.target.minutes.value);
    if (mins > 0) {
      setTimer(mins * 60);
      setTimerRunning(true);
    }
  };

  const handleStop = () => {
    setTimerRunning(false);
    setTimer(0);
    clearInterval(intervalRef.current);
  };

  const handleResetStudy = () => {
    setStudySeconds(0);
    localStorage.setItem("studySeconds", 0);
  };

  return (
    <TimerContext.Provider
      value={{
        studySeconds,
        timer,
        timerRunning,
        handleSetTimer,
        handleStop,
        handleResetStudy,
      }}
    >
      {children}
    </TimerContext.Provider>
  );
}

export function TimerBar() {
  const { timer, timerRunning } = useContext(TimerContext);
  if (!timerRunning || timer <= 0) return null;
  return (
    <div className="timer-bar">
      ⏳ Study Timer: <strong>{formatTime(timer)}</strong>
    </div>
  );
}

export default SideNavbar;
