import Link from 'next/link'
import Header from "@/components/Header"

export default function NotFound() {
  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Content Not Found</h2>
          <p className="text-gray-600 mb-6">
            The content you're looking for doesn't exist or has been moved.
          </p>
          <Link
            href="/"
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            ← Back to Dashboard
          </Link>
        </div>
      </main>
    </div>
  )
} 