import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, History, Settings, Menu, X, Orbit } from 'lucide-react';
import { useState } from 'react';
import { useExoStore } from '../store/useExoStore';
import { cn } from '../lib/utils';

export function Layout() {
  const { sidebarOpen, toggleSidebar } = useExoStore();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { path: '/', label: 'Analyze', icon: LayoutDashboard },
    { path: '/twin', label: '3D Twin', icon: Orbit },
    { path: '/history', label: 'History', icon: History },
    { path: '/settings', label: 'Settings', icon: Settings },
  ];

  const isActivePath = (path: string) =>
    location.pathname === path || (path !== '/' && location.pathname.startsWith(path));

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Mobile menu button */}
      <button
        className="fixed top-4 left-4 z-50 lg:hidden p-2 bg-white rounded-lg shadow-md border border-gray-200"
        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        aria-label="Toggle menu"
      >
        {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed lg:static inset-y-0 left-0 z-40 w-64 bg-white border-r border-gray-200',
          'transform transition-transform duration-300 ease-in-out flex flex-col',
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
          !sidebarOpen && 'lg:hidden'
        )}
      >
        <div className="flex h-16 items-center justify-between px-4 border-b border-gray-200">
          <h1 className="text-xl font-bold text-blue-600">ExoSphere</h1>
          <button
            className="lg:hidden p-2 rounded-lg hover:bg-gray-100"
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={20} />
          </button>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive: navActive }) =>
                cn(
                  'flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                  navActive || isActivePath(item.path)
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                )
              }
              onClick={() => {
                setMobileMenuOpen(false);
                if (!sidebarOpen) toggleSidebar();
              }}
            >
              <item.icon className="w-5 h-5 mr-3 flex-shrink-0" />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-200">
          <button
            className="w-full text-left text-xs text-gray-500 hover:text-gray-700"
            onClick={toggleSidebar}
          >
            {sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          </button>
          <div className="flex items-center justify-between text-xs text-gray-500 mt-2">
            <span>ExoSphere v0.1.0</span>
            <span className="text-blue-600 font-medium">Phase 8</span>
          </div>
        </div>
      </aside>

      {/* Mobile overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Main content */}
      <main className="flex-1 min-h-screen">
        <header className="sticky top-0 z-30 bg-white/80 backdrop-blur-sm border-b border-gray-200">
          <div className="flex items-center justify-between px-4 py-3">
            <h2 className="text-lg font-semibold text-gray-900">
              {{
                '/': 'Analyze',
                '/twin': '3D Digital Twin',
                '/history': 'History',
                '/settings': 'Settings',
              }[location.pathname] ||
                (location.pathname.startsWith('/twin') ? '3D Digital Twin' : 'ExoSphere')}
            </h2>
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex items-center gap-2 text-sm text-gray-500">
                <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium">
                  Phase 8
                </span>
              </div>
            </div>
          </div>
        </header>

        <div className="p-4 lg:p-6 overflow-auto pb-16">
          <Outlet />
        </div>

        {/* Global footer disclaimer */}
        <footer className="fixed bottom-0 left-0 right-0 bg-gray-50 border-t border-gray-200 px-4 py-2 text-center text-xs text-gray-500">
          Scientific results come from physics-based retrieval. ML scores are candidate indicators only.
        </footer>
      </main>
    </div>
  );
}

export default Layout;
