import { Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "./components/AppLayout";
import AgentChatPage from "./pages/AgentChatPage";
import AgentDetailPage from "./pages/AgentDetailPage";
import AgentsPage from "./pages/AgentsPage";
import ChatPage from "./pages/ChatPage";
import HistoryPage from "./pages/HistoryPage";
import KnowledgeDetailPage from "./pages/KnowledgeDetailPage";
import HomePage from "./pages/HomePage";
import KnowledgePage from "./pages/KnowledgePage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<HomePage />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="agents" element={<AgentsPage />} />
        <Route path="agents/:agentId" element={<AgentDetailPage />} />
        <Route path="agents/:agentId/chat" element={<AgentChatPage />} />
        <Route path="knowledge" element={<KnowledgePage />} />
        <Route path="knowledge/:knowledgeBaseId" element={<KnowledgeDetailPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
