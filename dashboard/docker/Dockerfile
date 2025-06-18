FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Set environment variables
ENV NODE_ENV=development
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Expose the port
EXPOSE 3000

# Start the development server
CMD ["npm", "run", "dev"] 