import { useState } from "react";
import { Outlet } from "react-router-dom";
import { API_BASE_URL } from "../api/client";
import Sidebar from "./Sidebar";

export default function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="app-shell">
      <button
        aria-label="Close navigation"
        className={sidebarOpen ? "sidebar-backdrop visible" : "sidebar-backdrop"}
        type="button"
        onClick={() => setSidebarOpen(false)}
      />
      <Sidebar open={sidebarOpen} onNavigate={() => setSidebarOpen(false)} />
      <main className="main">
        <header className="topbar">
          <button
            aria-label="Open navigation"
            className="mobile-menu-button"
            type="button"
            onClick={() => setSidebarOpen(true)}
          >
            Menu
          </button>
          <div>
            <span>API Base URL</span>
            <strong>{API_BASE_URL}</strong>
          </div>
        </header>
        <section className="page">
          <Outlet />
        </section>
      </main>
    </div>
  );
}
