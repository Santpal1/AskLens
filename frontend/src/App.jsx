import { useState } from "react";
import AppMain from "./components/AppMain";
import SideNavbar, { HomePage, TimerProvider } from "./components/SideNavbar";
import "./App.css";

function App() {
  const [selectedTab, setSelectedTab] = useState("home");

  return (
    <TimerProvider>
      <div style={{ display: "flex", minHeight: "100vh", height: "100vh", width: "100vw", overflow: "hidden" }}>
        <SideNavbar onSelectTab={setSelectedTab} selectedTab={selectedTab} />
        <div style={{ flex: 1, overflowY: "auto", background: "none" }}>
          {selectedTab === "home" && <HomePage />}
          {selectedTab === "main" && <AppMain />}
        </div>
      </div>
    </TimerProvider>
  );
}

export default App;