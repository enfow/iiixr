import { NextResponse } from 'next/server';
import Docker from 'dockerode';
import path from 'path';

const docker = new Docker();
const PROJECT_ROOT = path.resolve(process.cwd(), '..');

export async function POST() {
  try {
    // Create and start the container
    const container = await docker.createContainer({
      Image: 'iiixr-2:latest',
      Cmd: ['python', 'src/train.py', '--algorithm', 'ppo', '--env', 'LunarLander-v3', '--n_episodes', '1000', '--max_steps', '1000'],
      Tty: true,
      AttachStdout: true,
      AttachStderr: true,
      HostConfig: {
        Binds: [
          // Mount models directory for saving trained models
          `${PROJECT_ROOT}/models:/app/models`,
          // Mount logs directory for training logs
          `${PROJECT_ROOT}/logs:/app/logs`,
        ],
      },
    });

    // Start the container
    await container.start();

    // Get container logs
    const stream = await container.logs({
      follow: true,
      stdout: true,
      stderr: true,
      timestamps: true
    });

    let logs = '';
    stream.on('data', (chunk: Buffer) => {
      logs += chunk.toString();
    });

    // Wait for container to finish
    await new Promise((resolve, reject) => {
      container.wait((err: Error | null, data: { StatusCode: number }) => {
        if (err) reject(err);
        else resolve(data);
      });
    });

    // Remove the container
    await container.remove();

    return NextResponse.json({
      message: 'Training completed successfully',
      logs: logs
    });
  } catch (error: any) {
    console.error('Failed to run training:', error);
    return NextResponse.json(
      { error: 'Failed to run training', details: error.message },
      { status: 500 }
    );
  }
} 