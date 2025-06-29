import { NextResponse } from 'next/server';
import Docker from 'dockerode';

const docker = new Docker();

export async function POST() {
  try {
    // List all containers (including stopped ones)
    const containers = await docker.listContainers({ all: true });
    
    // Find our training container
    const trainingContainer = containers.find(container => 
      container.Image === 'iiixr-2:latest' && 
      container.State === 'running'
    );

    if (!trainingContainer) {
      return NextResponse.json({
        message: 'No training container is currently running'
      });
    }

    // Get the container instance
    const container = docker.getContainer(trainingContainer.Id);
    
    // Stop the container
    await container.stop();
    
    // Remove the container
    await container.remove();

    return NextResponse.json({
      message: 'Training stopped successfully'
    });
  } catch (error: any) {
    console.error('Failed to stop training:', error);
    return NextResponse.json(
      { error: 'Failed to stop training', details: error.message },
      { status: 500 }
    );
  }
} 