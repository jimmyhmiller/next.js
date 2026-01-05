import type { Config } from 'tailwindcss'
import { customPlugin } from './lib/plugin'

export default {
  content: ['./pages/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {},
  },
  plugins: [customPlugin],
} satisfies Config
