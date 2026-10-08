import { Link } from 'react-router-dom';
import { Sparkles, BarChart3, FileText } from 'lucide-react';
import { PlanetSearchForm } from '../components/PlanetSearchForm';

export function Dashboard() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Analyze</h1>
        <p className="text-gray-600 mt-1">Search for a planet and run atmospheric retrieval analysis</p>
      </div>

      <div className="card p-6 mb-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Find a Planet</h2>
        <PlanetSearchForm />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <QuickActionCard
          icon={<Sparkles className="w-6 h-6 text-purple-600" />}
          title="New Analysis"
          description="Create a new atmospheric retrieval analysis"
          href="/analyze"
        />
        <QuickActionCard
          icon={<BarChart3 className="w-6 h-6 text-blue-600" />}
          title="View History"
          description="Browse previous analyses and results"
          href="/history"
        />
        <QuickActionCard
          icon={<FileText className="w-6 h-6 text-green-600" />}
          title="3D Digital Twin"
          description="Orbit and atmosphere visualisation with provenance tags"
          href="/twin"
        />
      </div>

      <div className="card">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Recent Analyses</h2>
          <Link to="/history" className="text-sm text-blue-600 hover:underline">
            View all
          </Link>
        </div>
        <div className="p-6">
          <div className="text-center py-12 text-gray-500">
            <svg
              className="w-12 h-12 mx-auto text-gray-300 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V20a2 2 0 01-2 2z"
              />
            </svg>
            <p className="mt-2 text-gray-500">No analyses yet</p>
            <p className="text-sm text-gray-400 mt-1">Create your first analysis to get started</p>
            <Link to="/analyze" className="mt-4 inline-block btn-primary">
              Create Your First Analysis
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

function QuickActionCard({
  icon,
  title,
  description,
  href,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  href: string;
}) {
  return (
    <Link to={href} className="card p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-gray-100 flex items-center justify-center text-gray-600">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-900">{title}</h3>
          <p className="text-sm text-gray-500 mt-1">{description}</p>
        </div>
        <svg
          className="w-5 h-5 text-gray-400 flex-shrink-0"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
    </Link>
  );
}

export default Dashboard;
