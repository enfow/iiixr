// ./Chart.tsx

'use client'

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
  ArcElement,
} from 'chart.js'
import { Bar, Line, Pie } from 'react-chartjs-2'

// Register the necessary components from Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
)

interface ChartProps {
  type: 'bar' | 'line' | 'pie'
  data: any
  options?: any
}

export default function Chart({ type, data, options }: ChartProps) {
  return (
    <div className="my-6 p-4 border rounded-lg bg-white shadow-sm">
      {type === 'bar' && <Bar data={data} options={options} />}
      {type === 'line' && <Line data={data} options={options} />}
      {type === 'pie' && <Pie data={data} options={options} />}
    </div>
  )
}