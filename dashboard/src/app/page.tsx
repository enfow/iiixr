'use client'

import { useState, useEffect } from 'react'
import Header from "@/components/Header";
import TrainingControl from "@/components/TrainingControl";
import MDXContent from "@/components/MDXContent";
import ContentNavigation from "@/components/ContentNavigation";
import { getMDXFiles, getMDXFile, MDXPost } from '@/lib/mdx-loader'

export default function Home() {
  const [currentFile, setCurrentFile] = useState('example-post')
  const [contentFiles, setContentFiles] = useState<Array<{id: string, title: string, description?: string, type: string}>>([])
  const [currentPost, setCurrentPost] = useState<MDXPost | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load content files on component mount
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
        
        // Set initial post
        const initialPost = files.find(file => file.id === currentFile) || files[0]
        if (initialPost) {
          setCurrentPost(initialPost)
          setCurrentFile(initialPost.id)
        }
      } catch (error) {
        console.error('Error loading MDX files:', error)
      } finally {
        setLoading(false)
      }
    }

    loadContent()
  }, [])

  useEffect(() => {
    // Update current post when file changes
    const updatePost = async () => {
      try {
        const post = await getMDXFile(currentFile)
        setCurrentPost(post)
      } catch (error) {
        console.error('Error loading post:', error)
      }
    }

    if (currentFile) {
      updatePost()
    }
  }, [currentFile])

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
        <div className="grid gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-medium text-gray-900">Welcome to iiixr Dashboard</h2>
            <p className="mt-2 text-sm text-gray-600">
              This is your new dashboard application with MDX content support.
            </p>
          </div>
          {/* <TrainingControl /> */}
          {contentFiles.length > 0 && (
            <ContentNavigation 
              files={contentFiles}
              currentFile={currentFile}
              onFileChange={setCurrentFile}
            />
          )}
          {currentPost && (
            <div className="bg-white rounded-lg shadow p-6">
              <MDXContent post={currentPost} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
