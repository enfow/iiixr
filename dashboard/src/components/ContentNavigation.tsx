'use client'

import Link from 'next/link'

interface ContentFile {
  id: string
  title: string
  description?: string
  type: string
}

interface ContentNavigationProps {
  files: ContentFile[]
  currentFile: string
  onFileChange?: (fileId: string) => void
}

export default function ContentNavigation({ files, currentFile, onFileChange }: ContentNavigationProps) {
  // Filter out files with type "main" and group remaining files by type
  const filteredFiles = files.filter(file => file.type !== 'main')
  const groupedFiles = filteredFiles.reduce((acc, file) => {
    acc[file.type] = acc[file.type] || []
    acc[file.type].push(file)
    return acc
  }, {} as Record<string, ContentFile[]>)
  
  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h3 className="text-xl font-bold text-gray-900 mb-4">Contents</h3>
      <div className="flex flex-wrap gap-3">
        {Object.entries(groupedFiles).map(([type, files]) => (
          <div key={type}>
            <h4 className="text-xl font-medium text-gray-900 mb-2">{type.charAt(0).toUpperCase() + type.slice(1)}</h4>
            {files.map((file) => (
              onFileChange ? (
                <button
                  key={file.id}
                  onClick={() => onFileChange(file.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors mr-2 mb-2 ${
                    currentFile === file.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {file.title}
                </button>
              ) : (
                <Link
                  key={file.id}
                  href={`/content/${file.id}`}
                  className={`inline-block px-4 py-2 rounded-lg text-sm font-medium transition-colors mr-2 mb-2 ${
                    currentFile === file.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {file.title}
                </Link>
              )
            ))}
          </div>
        ))}
      </div>
    </div>
  )
} 