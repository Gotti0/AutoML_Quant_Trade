import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { ScreenerPage } from './pages/ScreenerPage';
import { LayoutDashboard, Activity } from 'lucide-react';

function App() {
  return (
    <BrowserRouter>
      {/* 공유 네비게이션 바 */}
      <nav className="bg-slate-950 border-b border-slate-800 px-6 py-2 flex items-center gap-2 sticky top-0 z-20">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `flex items-center gap-1.5 text-sm font-medium px-3 py-1.5 rounded transition-colors ${
              isActive
                ? 'text-indigo-400 bg-indigo-400/10'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`
          }
        >
          <LayoutDashboard className="w-4 h-4" />
          Dashboard
        </NavLink>
        <NavLink
          to="/screener"
          className={({ isActive }) =>
            `flex items-center gap-1.5 text-sm font-medium px-3 py-1.5 rounded transition-colors ${
              isActive
                ? 'text-indigo-400 bg-indigo-400/10'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`
          }
        >
          <Activity className="w-4 h-4" />
          Screener
        </NavLink>
      </nav>

      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/screener" element={<ScreenerPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
