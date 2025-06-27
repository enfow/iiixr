'use client'
import { useState, useEffect } from 'react'
import { MDXProvider } from '@mdx-js/react'
import { compile, run } from '@mdx-js/mdx'
import * as runtime from 'react/jsx-runtime'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { format } from 'date-fns'
import 'katex/dist/katex.min.css'

interface MDXContentProps {
  post: {
    title: string
    date: string
    description?: string
    tags?: string[]
    content: string // raw MDX string
    _raw: {
      flattenedPath: string
    }
  }
}

// Custom MDX components
const components = {
  // Handle images and GIFs
  img: ({ src, alt, ...props }: { src: string; alt: string; [key: string]: any }) => (
    <img 
      src={src} 
      alt={alt} 
      className="max-w-full h-auto rounded-lg mx-auto my-4" 
      {...props} 
    />
  ),
  
  // Custom headings
  h1: ({ children }: { children: React.ReactNode }) => (
    <h1 className="text-3xl font-bold mt-8 mb-4 text-gray-900">{children}</h1>
  ),
  h2: ({ children }: { children: React.ReactNode }) => (
    <h2 className="text-2xl font-semibold mt-6 mb-3 text-gray-800">{children}</h2>
  ),
  h3: ({ children }: { children: React.ReactNode }) => (
    <h3 className="text-xl font-medium mt-4 mb-2 text-gray-700">{children}</h3>
  ),
  
  // Custom paragraph
  p: ({ children }: { children: React.ReactNode }) => (
    <p className="mb-4 leading-relaxed text-gray-700">{children}</p>
  ),
  
  // Custom code blocks
  pre: ({ children }: { children: React.ReactNode }) => (
    <pre className="bg-gray-100 rounded-lg p-4 overflow-x-auto mb-4 text-sm">{children}</pre>
  ),
  
  // Inline code
  code: ({ children }: { children: React.ReactNode }) => (
    <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">{children}</code>
  ),
}

export default function MDXContent({ post }: MDXContentProps) {
  const [Content, setContent] = useState<React.ComponentType | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!post?.content) {
      setIsLoading(false)
      return
    }

    const compileAndRun = async () => {
      try {
        setIsLoading(true)
        setError(null)

        // Compile the MDX content
        const compiled = await compile(post.content, {
          outputFormat: 'function-body',
          development: false,
          remarkPlugins: [remarkMath, remarkGfm],
          rehypePlugins: [rehypeKatex, rehypeHighlight],
        })

        // Run the compiled content
        const { default: MDXContent } = await run(compiled, {
          ...runtime,
          baseUrl: import.meta.url,
        })

        setContent(() => MDXContent)
      } catch (err) {
        console.error('Error compiling MDX:', err)
        setError(err instanceof Error ? err.message : 'Unknown error occurred')
        setContent(() => () => (
          <div className="text-red-600 p-4 border border-red-300 rounded bg-red-50">
            <h3 className="font-semibold mb-2">Error loading content</h3>
            <p className="text-sm">{err instanceof Error ? err.message : 'Unknown error'}</p>
          </div>
        ))
      } finally {
        setIsLoading(false)
      }
    }

    compileAndRun()
  }, [post?.content])

  if (!post) return null

  return (
    <article className="max-w-4xl mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">{post.title}</h1>
        <div className="flex items-center text-gray-600 mb-4">
          <time dateTime={post.date} className="text-sm">
            {format(new Date(post.date), 'MMMM d, yyyy')}
          </time>
        </div>
        {post.description && (
          <p className="text-lg text-gray-600 mb-4">{post.description}</p>
        )}
        {post.tags && post.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-6">
            {post.tags.map((tag: string) => (
              <span
                key={tag}
                className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </header>
      <div className="prose prose-lg max-w-none">
        <MDXProvider components={components}>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-gray-600">Loading content...</div>
            </div>
          ) : error ? (
            <div className="text-red-600 p-4 border border-red-300 rounded bg-red-50">
              <h3 className="font-semibold mb-2">Error loading content</h3>
              <p className="text-sm">{error}</p>
            </div>
          ) : Content ? (
            <Content />
          ) : (
            <div className="text-gray-600">No content available</div>
          )}
        </MDXProvider>
      </div>
    </article>
  )
}