import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Home" },
  { to: "/chat", label: "Chat" },
  { to: "/agents", label: "Agents" },
  { to: "/knowledge", label: "Knowledge" },
  { to: "/history", label: "History" }
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">AI</div>
        <div>
          <strong>AI Workspace</strong>
          <span>User Portal</span>
        </div>
      </div>
      <nav className="nav-list" aria-label="Workspace navigation">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
