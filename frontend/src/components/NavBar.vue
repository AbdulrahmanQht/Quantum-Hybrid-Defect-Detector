<script setup>
import { ref, computed, markRaw, onMounted, onUnmounted } from 'vue'
import Cookies from 'js-cookie'
import { useI18n } from 'vue-i18n'
import { Atom, Home, Search, ChartColumn, Mail, Languages, Sun, Moon, Menu, X } from 'lucide-vue-next'

const { t, locale } = useI18n()

const LANG_KEY = 'app_lang'
const THEME_KEY = 'theme'

const currentLang = computed(() => locale.value)
const isDark = ref(Cookies.get(THEME_KEY) === 'dark')
const menuOpen = ref(false)

const items = computed(() => [
  { label: t('navbar.home'), lucideIcon: markRaw(Home), to: '/' },
  { label: t('navbar.classify'), lucideIcon: markRaw(Search), to: '/classify' },
  { label: t('navbar.benchmark'), lucideIcon: markRaw(ChartColumn), to: '/benchmark' },
  { label: t('navbar.quantum_advantage'), lucideIcon: markRaw(Atom), to: '/quantum-advantage', isQuantum: true },
  { label: t('navbar.contact'), lucideIcon: markRaw(Mail), to: '/contact' },
])

const toggleLanguage = () => {
  const next = locale.value === 'EN' ? 'AR' : 'EN'
  locale.value = next
  Cookies.set(LANG_KEY, next, { expires: 365, path: '/' })

  document.documentElement.classList.toggle('lang-ar', next === 'AR')
  document.documentElement.lang = next === 'AR' ? 'ar' : 'en'
}

const toggleTheme = () => {
  isDark.value = !isDark.value
  document.documentElement.classList.toggle('p-dark', isDark.value)
  Cookies.set(THEME_KEY, isDark.value ? 'dark' : 'light', { expires: 365, path: '/' })
}

const closeMenu = () => {
  menuOpen.value = false
}
function onKeydown(e) {
  if (e.key === 'Escape' && menuOpen.value) closeMenu()
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <header class="qnn-bar">
    <div class="qnn-inner">
      <router-link to="/" class="qnn-logo" @click="closeMenu">
        <img src="/qnn_logo_final_no_text.svg" alt="QNN">
        <div class="qnn-logo-copy" lang="en" translate="no">
          <span class="qnn-logo-title" lang="en" translate="no">Quantum-Hybrid</span>
          <span class="qnn-logo-subtitle" lang="en" translate="no">Defect Detector</span>
        </div>
      </router-link>

      <nav class="qnn-nav" :class="{ 'qnn-nav--open': menuOpen }" id="qnn-nav" aria-label="Main navigation">
        <router-link v-for="item in items" :key="item.to" :to="item.to" class="qnn-link"
          :class="{ 'qnn-link--quantum': item.isQuantum }" aria-current-value="page" @click="closeMenu">
          <component :is="item.lucideIcon" class="qnn-link-icon" />
          <span>{{ item.label }}</span>
        </router-link>
      </nav>

      <div class="qnn-actions">
        <button class="qnn-lang-btn" @click="toggleLanguage">
          <Languages :size="14" />
          <span>{{ currentLang === 'EN' ? 'العربية' : 'English' }}</span>
        </button>

        <button class="qnn-icon-btn" :aria-label="isDark ? 'Light mode' : 'Dark mode'" @click="toggleTheme">
          <component :is="isDark ? markRaw(Sun) : markRaw(Moon)" :size="16" />
        </button>

        <button class="qnn-icon-btn qnn-burger" :aria-label="menuOpen ? 'Close menu' : 'Open menu'"
          :aria-expanded="menuOpen" aria-controls="qnn-nav" @click="menuOpen = !menuOpen">
          <component :is="menuOpen ? markRaw(X) : markRaw(Menu)" :size="19" />
        </button>
      </div>
    </div>

    <div class="qnn-accent-line" />
  </header>

  <div class="h-[78px] md:h-[88px]" />
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@500;600;700&family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --q-teal: #2ab8b8;
  --q-teal-soft: rgba(42, 184, 184, 0.12);
  --q-teal-glow: rgba(42, 184, 184, 0.35);
  --q-navy: #0d1f2d;
  --q-navy-mid: #152536;
  --q-bar-bg: rgba(255, 255, 255, 0.88);
  --q-bar-border: rgba(13, 31, 45, 0.08);
  --q-text: #0d1f2d;
  --q-muted: #4a6678;
  --q-h: 64px;
}

.p-dark {
  --q-bar-bg: rgba(9, 18, 28, 0.82);
  --q-bar-border: rgba(42, 184, 184, 0.12);
  --q-text: #cde8ec;
  --q-muted: #8db4bf;
}

.qnn-bar {
  position: fixed;
  inset: 0 0 auto 0;
  z-index: 999;
  background: var(--q-bar-bg);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border-bottom: 1px solid var(--q-bar-border);
  font-family: 'DM Sans', sans-serif;
}

.qnn-inner {
  display: flex;
  align-items: center;
  height: var(--q-h);
  padding: 0 1rem;
  max-width: 1400px;
  margin: 0 auto;
  position: relative;
}

.qnn-accent-line {
  height: 2px;
  background: linear-gradient(90deg, transparent 0%, var(--q-teal) 25%, #1eb0c8 55%, transparent 100%);
  opacity: 0.55;
}

.qnn-logo {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  flex-shrink: 0;
  z-index: 2;
  text-decoration: none;
}

.qnn-logo img {
  height: 44px;
  width: auto;
  object-fit: contain;
  transition: transform 0.25s ease, filter 0.25s ease;
  filter: drop-shadow(0 0 0px var(--q-teal));
}

.qnn-logo:hover img {
  transform: scale(1.05);
  filter: drop-shadow(0 0 8px var(--q-teal-glow));
}

.qnn-logo-copy {
  display: flex;
  flex-direction: column;
  line-height: 1;
}

.qnn-logo-title {
  font-family: 'Oxanium', sans-serif;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--q-text);
  direction: ltr !important;
  text-align: left !important;
}

.qnn-logo-subtitle {
  font-size: 0.72rem;
  color: var(--q-muted);
  margin-top: 0.22rem;
  direction: ltr !important;
  text-align: left !important;
}

.qnn-nav {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.3rem;
  border: 1px solid var(--q-bar-border);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.45);
}

.p-dark .qnn-nav {
  background: rgba(255, 255, 255, 0.03);
}

.qnn-link {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.58rem 0.95rem;
  border-radius: 999px;
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--q-muted);
  text-decoration: none;
  position: relative;
  white-space: nowrap;
  transition: color 0.2s, background 0.2s, transform 0.2s;
}

.qnn-link:hover,
.qnn-link.router-link-active {
  color: var(--q-text);
  background: var(--q-teal-soft);
}

.qnn-link.router-link-active {
  box-shadow: inset 0 0 0 1px rgba(42, 184, 184, 0.15);
}

.qnn-link--quantum {
  color: var(--q-teal);
  font-family: 'Oxanium', sans-serif;
  font-weight: 600;
}

.qnn-link-icon {
  width: 14px;
  height: 14px;
  opacity: 0.85;
  flex-shrink: 0;
}

.qnn-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  z-index: 2;
}

.qnn-lang-btn,
.qnn-icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.38rem;
  border: none;
  cursor: pointer;
  transition: color 0.18s, background 0.18s, transform 0.18s;
}

.qnn-lang-btn {
  padding: 0.55rem 0.8rem;
  border-radius: 999px;
  background: transparent;
  color: var(--q-muted);
  font-family: 'DM Sans', sans-serif;
  font-size: 0.82rem;
  font-weight: 600;
}

.qnn-icon-btn {
  width: 38px;
  height: 38px;
  border-radius: 999px;
  background: transparent;
  color: var(--q-muted);
}

.qnn-lang-btn:hover,
.qnn-icon-btn:hover {
  color: var(--q-teal);
  background: var(--q-teal-soft);
}

.qnn-burger {
  display: none;
}

@media (max-width: 1100px) {
  .qnn-logo-copy {
    display: none;
  }

  .qnn-nav {
    position: fixed;
    top: calc(var(--q-h) + 8px);
    left: 1rem;
    right: 1rem;
    transform: none;
    flex-direction: column;
    align-items: stretch;
    gap: 0.35rem;
    padding: 0.85rem;
    border-radius: 22px;
    background: var(--q-bar-bg);
    border: 1px solid var(--q-bar-border);
    opacity: 0;
    pointer-events: none;
    translate: 0 -8px;
    transition: opacity 0.22s ease, translate 0.22s ease;
  }

  .qnn-nav--open {
    opacity: 1;
    pointer-events: auto;
    translate: 0 0;
  }

  .qnn-link {
    justify-content: flex-start;
    border-radius: 14px;
  }

  .qnn-burger {
    display: flex;
  }
}

@media (max-width: 640px) {
  .qnn-inner {
    padding: 0 0.8rem;
  }

  .qnn-lang-btn span {
    display: none;
  }
}
</style>
