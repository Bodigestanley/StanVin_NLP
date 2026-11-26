/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  safelist: ['flex', 'grid', 'block', 'hidden', 'w-full', 'text-sm'],
  theme: { extend: {} },
  plugins: [],
}
