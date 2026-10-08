import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ExoProvider } from './providers/ExoProvider';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { AnalyzePage } from './pages/AnalyzePage';
import { TwinPage } from './pages/TwinPage';
import { HistoryPage } from './pages/HistoryPage';
import { SettingsPage } from './pages/SettingsPage';

import { queryClient } from './lib/queryClient';
import './index.css';

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ExoProvider>
          <AppRoutes />
        </ExoProvider>
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="analyze" element={<AnalyzePage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="twin" element={<TwinPage />} />
        <Route path="twin/:analysisId" element={<TwinPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}

export default App;
