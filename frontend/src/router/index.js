import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const routes = [
  { path: '/', name: 'home', component: HomeView },
  { path: '/classify', name: 'classify', component: () => import('../views/ClassifyView.vue') },
  { path: '/benchmark', name: 'benchmark', component: () => import('../views/BenchmarkView.vue') },
  { path: '/about', name: 'about', component: () => import('../views/AboutView.vue') },
  { path: '/contact', name: 'contact', component: () => import('../views/ContactView.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router