import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

export interface MDXPost {
  id: string
  title: string
  date: string
  description?: string
  tags?: string[]
  content: string
  body: {
    html: string
  }
  _raw: {
    flattenedPath: string
  }
}

function getMDXFiles(): MDXPost[] {
  const contentDir = path.join(process.cwd(), 'content')
  
  if (!fs.existsSync(contentDir)) {
    return []
  }

  const files = fs.readdirSync(contentDir)
  const mdxFiles = files.filter(file => file.endsWith('.mdx'))

  return mdxFiles.map(file => {
    const filePath = path.join(contentDir, file)
    const fileContent = fs.readFileSync(filePath, 'utf8')
    const { data, content } = matter(fileContent)
    
    const id = file.replace('.mdx', '')
    
    return {
      id,
      title: data.title || 'Untitled',
      date: data.date || new Date().toISOString(),
      description: data.description,
      tags: data.tags || [],
      content,
      body: {
        html: content // For now, we'll use the raw content
      },
      _raw: {
        flattenedPath: id
      }
    }
  })
}

export async function GET() {
  try {
    const files = getMDXFiles()
    return NextResponse.json(files)
  } catch (error) {
    console.error('Error reading MDX files:', error)
    return NextResponse.json({ error: 'Failed to load MDX files' }, { status: 500 })
  }
} 