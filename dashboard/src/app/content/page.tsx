'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import Header from "@/components/Header"
import { getMDXFiles } from '@/lib/mdx-loader'

interface ContentFile {
  id: string
  title: string
  description?: string
  type: string
}

export default function ContentIndexPage() {
  const [contentFiles, setContentFiles] = useState<ContentFile[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadContent = async () => {
      try {
        const files = await getMDXFiles()
        const fileList = files.map(file => ({
          id: file.id,
          title: file.title,
          description: file.description,
          type: file.type
        }))
        setContentFiles(fileList)
      } catch (error) {
        console.error('Error loading MDX files:', error)
      } finally {
        setLoading(false)
      }
    }

    loadContent()
  }, [])

  // Group files by type
  const groupedFiles = contentFiles.reduce((acc, file) => {
    acc[file.type] = acc[file.type] || []
    acc[file.type].push(file)
    return acc
  }, {} as Record<string, ContentFile[]>)

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100">
        <Header />
        <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-center">
            <div className="text-lg">Loading content...</div>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Content Library</h1>
          <p className="text-gray-600">
            Browse all available content organized by category.
          </p>
        </div>

        <div className="space-y-8">
          {Object.entries(groupedFiles).map(([type, files]) => (
            <div key={type} className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </h2>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {files.map((file) => (
                  <Link
                    key={file.id}
                    href={`/content/${file.id}`}
                    className="block p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <h3 className="font-medium text-gray-900 mb-2">{file.title}</h3>
                    {file.description && (
                      <p className="text-sm text-gray-600 line-clamp-3">{file.description}</p>
                    )}
                    <div className="mt-2">
                      <span className="text-sm text-blue-600 hover:text-blue-800">
                        Read more →
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>

        {contentFiles.length === 0 && (
          <div className="bg-white rounded-lg shadow p-6 text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-2">No Content Available</h2>
            <p className="text-gray-600">
              No MDX content files were found. Add some .mdx files to the content directory to get started.
            </p>
          </div>
        )}
      </main>
    </div>
  )
} 