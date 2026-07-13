import { Outlet } from "react-router-dom";
import { API_BASE_URL } from "../api/client";
import Sidebar from "./Sidebar";

export default function AppLayout() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main">
        <header className="topbar">
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
