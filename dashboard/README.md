# iiixr Dashboard

A Next.js dashboard application with MDX content support for displaying reinforcement learning documentation and training results.

## Features

- **MDX Content Support**: Display rich content with math equations, code highlighting, and images
- **Training Control**: Interface for managing reinforcement learning training sessions
- **Responsive Design**: Modern UI built with Tailwind CSS
- **TypeScript**: Full type safety throughout the application

## MDX Content Features

The application supports:
- **Math Equations**: Using KaTeX for beautiful mathematical notation
- **Code Highlighting**: Syntax highlighting for code blocks
- **GitHub Flavored Markdown**: Tables, strikethrough, task lists, etc.
- **Images and GIFs**: Support for displaying training progress and results
- **Custom Components**: Extensible component system for rich content

## Setup

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Content Structure

MDX files are stored in the `content/` directory and automatically processed by contentlayer. Each MDX file should have frontmatter with:

```yaml
---
title: "Your Post Title"
date: "2024-01-15"
description: "Brief description of the post"
tags: ["tag1", "tag2"]
---
```

## Adding New Content

1. Create a new `.mdx` file in the `content/` directory
2. Add frontmatter with required metadata
3. Write your content using Markdown with support for:
   - Math equations: `$E = mc^2$` or `$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$`
   - Code blocks with syntax highlighting
   - Images and GIFs
   - Tables and other GitHub Flavored Markdown features

## Configuration

The application uses:
- **contentlayer2**: For MDX processing
- **remark-math/rehype-katex**: For math equation rendering
- **remark-gfm/rehype-highlight**: For GitHub Flavored Markdown and syntax highlighting
- **date-fns**: For date formatting
- **Tailwind CSS**: For styling

## Building for Production

```bash
npm run build
npm start
```

## Customization

You can customize the MDX rendering by modifying:
- `contentlayer.config.ts`: Content processing configuration
- `src/components/MDXContent.tsx`: Component styling and layout
- `src/app/globals.css`: Global styles for content elements

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
