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

export async function getMDXFiles(): Promise<MDXPost[]> {
  try {
    const response = await fetch('/api/mdx')
    if (!response.ok) {
      throw new Error('Failed to fetch MDX files')
    }
    return await response.json()
  } catch (error) {
    console.error('Error fetching MDX files:', error)
    return []
  }
}

export async function getMDXFile(id: string): Promise<MDXPost | null> {
  try {
    const files = await getMDXFiles()
    return files.find(file => file.id === id) || null
  } catch (error) {
    console.error('Error fetching MDX file:', error)
    return null
  }
} 