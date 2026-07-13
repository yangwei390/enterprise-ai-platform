import { Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "./components/AppLayout";
import AgentsPage from "./pages/AgentsPage";
import ChatPage from "./pages/ChatPage";
import HistoryPage from "./pages/HistoryPage";
import HomePage from "./pages/HomePage";
import KnowledgePage from "./pages/KnowledgePage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<HomePage />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="agents" element={<AgentsPage />} />
        <Route path="knowledge" element={<KnowledgePage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
