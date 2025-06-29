'use client';

import { useState, useEffect } from 'react';

export default function TrainingControl() {
  const [isTraining, setIsTraining] = useState(false);
  const [status, setStatus] = useState<string>('');
  const [logs, setLogs] = useState<string>('');

  const startTraining = async () => {
    try {
      setIsTraining(true);
      setStatus('Starting training...');
      setLogs('');
      
      const response = await fetch('/api/training/start', {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to start training');
      }
      
      const data = await response.json();
      setStatus(data.message || 'Training started successfully');
      setLogs(data.logs || '');
    } catch (error) {
      console.error('Training error:', error);
      setStatus('Failed to start training');
    } finally {
      setIsTraining(false);
    }
  };

  const stopTraining = async () => {
    try {
      setStatus('Stopping training...');
      
      const response = await fetch('/api/training/stop', {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to stop training');
      }
      
      const data = await response.json();
      setStatus(data.message || 'Training stopped successfully');
    } catch (error) {
      console.error('Stop training error:', error);
      setStatus('Failed to stop training');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-medium text-gray-900 mb-4">Training Control</h2>
      
      <div className="space-y-4">
        <div className="flex gap-4">
          <button
            onClick={startTraining}
            disabled={isTraining}
            className={`px-4 py-2 rounded-md text-white font-medium ${
              isTraining
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isTraining ? 'Starting...' : 'Start Training'}
          </button>
          
          <button
            onClick={stopTraining}
            disabled={!isTraining}
            className={`px-4 py-2 rounded-md text-white font-medium ${
              !isTraining
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700'
            }`}
          >
            Stop Training
          </button>
        </div>
        
        {status && (
          <div className="mt-4 p-3 rounded-md bg-gray-50">
            <p className="text-sm text-gray-700">{status}</p>
          </div>
        )}

        {logs && (
          <div className="mt-4 p-3 rounded-md bg-gray-900">
            <pre className="text-sm text-gray-100 whitespace-pre-wrap">{logs}</pre>
          </div>
        )}
      </div>
    </div>
  );
} 