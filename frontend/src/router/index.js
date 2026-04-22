import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const routes = [
  { path: '/', name: 'home', component: HomeView },
  { path: '/classify', name: 'classify', component: () => import('../views/ClassifyView.vue') },
  { path: '/benchmark', name: 'benchmark', component: () => import('../views/BenchmarkView.vue') },
  { path: '/quantum-advantage', name: 'quantum-advantage', component: () => import('../views/QuantumAdvantageView.vue') },
  { path: '/contact', name: 'contact', component: () => import('../views/ContactView.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    }

    return {
      top: 0,
      left: 0,
      behavior: 'smooth',
    }
  },
})

export default router