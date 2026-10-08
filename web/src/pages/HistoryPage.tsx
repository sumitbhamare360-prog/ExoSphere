export function HistoryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">History</h1>
        <p className="text-gray-600 mt-1">Past analyses will be listed here.</p>
      </div>
      <div className="card p-6">
        <p className="text-sm text-gray-500">
          No analyses yet. Run an analysis from the Analyze page, then reopen it here.
        </p>
      </div>
    </div>
  );
}

export default HistoryPage;
