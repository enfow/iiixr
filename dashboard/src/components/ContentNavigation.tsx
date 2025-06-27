'use client'

interface ContentFile {
  id: string
  title: string
  description?: string
}

interface ContentNavigationProps {
  files: ContentFile[]
  currentFile: string
  onFileChange: (fileId: string) => void
}

export default function ContentNavigation({ files, currentFile, onFileChange }: ContentNavigationProps) {
  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Available Content</h3>
      <div className="flex flex-wrap gap-3">
        {files.map((file) => (
          <button
            key={file.id}
            onClick={() => onFileChange(file.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              currentFile === file.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {file.title}
          </button>
        ))}
      </div>
    </div>
  )
} 