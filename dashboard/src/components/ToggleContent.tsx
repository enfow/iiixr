
'use client'

import { useState, ReactNode } from 'react'

interface ToggleContentProps {
  buttonText: string
  children: ReactNode
}

export default function ToggleContent({ buttonText, children }: ToggleContentProps) {
  const [isOpen, setIsOpen] = useState(false)

  const toggleVisibility = () => {
    setIsOpen(!isOpen)
  }

  return (
    <div className="my-4">
      <button
        onClick={toggleVisibility}
        className="px-4 py-2 text-sm font-semibold text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
      >
        {isOpen ? `Hide ${buttonText}` : `Show ${buttonText}`}
      </button>
      
      {/* Conditionally render the content with a smooth transition */}
      <div
        className={`mt-3 overflow-hidden transition-all duration-300 ease-in-out ${
          isOpen ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        {/* The children (your code block) will be rendered here */}
        {children}
      </div>
    </div>
  )
}