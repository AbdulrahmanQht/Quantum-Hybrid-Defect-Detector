/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['selector', '.p-dark'], // Add this line here
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}