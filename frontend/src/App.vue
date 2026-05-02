<script setup>
import { onMounted, onUnmounted, nextTick } from 'vue'
import NavBar from './components/NavBar.vue'
import Footer from './components/Footer.vue'
import { useRouter } from 'vue-router'

const router = useRouter()
let saveTimer

const handleScroll = () => {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    localStorage.setItem('scrollRestore', JSON.stringify({
      path: router.currentRoute.value.fullPath,
      top: window.scrollY,
      left: window.scrollX
    }))
  }, 100)
}

onMounted(() => {
  window.addEventListener('scroll', handleScroll, { passive: true })
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
  clearTimeout(saveTimer)
})
</script>

<template>
  <div class="app-shell min-h-screen">
    <div class="app-shell__bg" />
    <NavBar />
    <main class="relative z-[1]">
      <RouterView />
    </main>
    <Footer/>
  </div>
</template>

<style>
.app-shell {
  position: relative;
  min-height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(42, 184, 184, 0.12), transparent 28%),
    radial-gradient(circle at top right, rgba(42, 184, 184, 0.08), transparent 22%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(245, 250, 251, 0.98));
  color: var(--q-text);
}

.p-dark .app-shell {
  background:
    radial-gradient(circle at top left, rgba(42, 184, 184, 0.18), transparent 30%),
    radial-gradient(circle at top right, rgba(42, 184, 184, 0.10), transparent 24%),
    linear-gradient(180deg, #08131d 0%, #0d1f2d 100%);
}

.app-shell__bg {
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(120deg, transparent 0%, rgba(42, 184, 184, 0.04) 50%, transparent 100%);
  z-index: 0;
}
</style>
