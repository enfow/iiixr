'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import Header from "@/components/Header";
import ContentNavigation from "@/components/ContentNavigation";
import MDXContent from "@/components/MDXContent";
import { getMDXFiles, getMDXFile, MDXPost } from '@/lib/mdx-loader'

export default function Home() {
  const [contentFiles, setContentFiles] = useState<Array<{id: string, title: string, description?: string, type: string}>>([])
  const [mainPageContent, setMainPageContent] = useState<MDXPost | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load content files and main page content on component mount
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
        
        // Load main page content
        const mainPage = await getMDXFile('mainPage')
        if (mainPage) {
          setMainPageContent(mainPage)
        }
      } catch (error) {
        console.error('Error loading MDX files:', error)
      } finally {
        setLoading(false)
      }
    }

    loadContent()
  }, [])

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
        <div className="grid gap-8">

          {/* Content Navigation */}
          {contentFiles.length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Content List</h2>
                <Link
                  href="/content"
                  className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  View All Content →
                </Link>
              </div>
              <ContentNavigation 
                files={contentFiles}
                currentFile=""
              />
            </div>
          )}
          {/* Main Page Content */}
          {mainPageContent && (
            <div className="bg-white rounded-lg shadow p-8">
              <MDXContent post={mainPageContent} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
