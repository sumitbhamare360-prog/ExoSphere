export function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600 mt-1">Dashboard preferences.</p>
      </div>
      <div className="card p-6">
        <p className="text-sm text-gray-500">
          API base URL is configured via the <code>VITE_API_BASE_URL</code> environment variable.
          Set <code>VITE_USE_MOCK=true</code> to run the dashboard against bundled mock data.
        </p>
      </div>
    </div>
  );
}

export default SettingsPage;
