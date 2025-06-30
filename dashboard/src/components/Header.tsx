import Link from 'next/link'

export default function Header() {
  return (
    <header className="bg-white shadow-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-center">
          <Link href="/" className="hover:opacity-80 transition-opacity">
            <h1 className="text-2xl font-bold text-gray-900 cursor-pointer">iiixr</h1>
          </Link>
        </div>
      </div>
    </header>
  );
} 