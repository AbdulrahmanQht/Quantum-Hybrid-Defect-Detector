import { createRouter, createWebHistory, START_LOCATION } from "vue-router";
import HomeView from "../views/HomeView.vue";

const routes = [
  { path: "/", name: "home", component: HomeView },
  {
    path: "/classify",
    name: "classify",
    component: () => import("../views/ClassifyView.vue"),
  },
  {
    path: "/benchmark",
    name: "benchmark",
    component: () => import("../views/BenchmarkView.vue"),
  },
  {
    path: "/quantum-advantage",
    name: "quantum-advantage",
    component: () => import("../views/QuantumAdvantageView.vue"),
  },
  {
    path: "/contact",
    name: "contact",
    component: () => import("../views/ContactView.vue"),
  },
  {
    path: "/:pathMatch(.*)*",
    name: "not-found",
    component: () => import("../views/NotFoundView.vue"),
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    // 1. Back/forward navigation uses the browser’s saved position
    if (savedPosition) {
      return savedPosition;
    }

    // 2. Hash navigation
    if (to.hash) {
      return { el: to.hash };
    }

    // 3. Initial page load (refresh) – try to restore from localStorage
    if (from === START_LOCATION) {
      // from.matched.length === 0 would also work
      const saved = localStorage.getItem("scrollRestore");
      if (saved) {
        const { path, top, left } = JSON.parse(saved);
        // Only restore if we’re still on the same page
        if (path === to.fullPath) {
          return { top, left };
        }
      }
      // If no saved position or different route, let it stay at top (or fall through)
      return false;
    }

    // 4. Normal in‑app navigation (links, router.push) – always scroll to top
    return { top: 0, left: 0 };
  },
});

export default router;
