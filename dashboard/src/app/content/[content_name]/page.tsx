'use client'

import { useState, useEffect, use } from 'react'
import { notFound } from 'next/navigation'
import Header from "@/components/Header"
import MDXContent from "@/components/MDXContent"
import ContentNavigation from "@/components/ContentNavigation"
import { getMDXFiles, getMDXFile, MDXPost } from '@/lib/mdx-loader'

interface ContentPageProps {
  params: Promise<{
    content_name: string
  }>
}

export default function ContentPage({ params }: ContentPageProps) {
  const { content_name } = use(params)
  const [contentFiles, setContentFiles] = useState<Array<{id: string, title: string, description?: string, type: string}>>([])
  const [currentPost, setCurrentPost] = useState<MDXPost | null>(null)
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
        
        // Load the specific post based on the route parameter
        const post = await getMDXFile(content_name)
        if (!post) {
          notFound()
          return
        }
        setCurrentPost(post)
      } catch (error) {
        console.error('Error loading MDX files:', error)
        notFound()
      } finally {
        setLoading(false)
      }
    }

    loadContent()
  }, [content_name])

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

  if (!currentPost) {
    notFound()
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid gap-6">
          {contentFiles.length > 0 && (
            <ContentNavigation 
              files={contentFiles}
              currentFile={content_name}
            />
          )}
          <div className="bg-white rounded-lg shadow p-6">
            <MDXContent post={currentPost} />
          </div>
        </div>
      </main>
    </div>
  )
} 