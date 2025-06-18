import Header from "@/components/Header";
import TrainingControl from "@/components/TrainingControl";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-medium text-gray-900">Welcome to iiixr Dashboard</h2>
            <p className="mt-2 text-sm text-gray-600">
              This is your new dashboard application.
            </p>
          </div>
          <TrainingControl />
        </div>
      </main>
    </div>
  );
}
