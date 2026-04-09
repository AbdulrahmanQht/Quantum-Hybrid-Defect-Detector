<script setup>
import { ref, computed, markRaw } from "vue";
import Cookies from 'js-cookie';
import { useI18n } from 'vue-i18n';
import { Atom, Home, Search, ChartColumn, Info, Mail, Languages, Sun, Moon, Menu, X } from 'lucide-vue-next';

const { t, locale } = useI18n();

const LANG_KEY = 'app_lang';
const THEME_KEY = 'theme';

const currentLang = computed(() => locale.value);
const isDark = ref(Cookies.get(THEME_KEY) === 'dark');
const menuOpen = ref(false);

const items = computed(() => [
  { label: t('navbar.home'),             lucideIcon: markRaw(Home),        to: '/' },
  { label: t('navbar.classify'),         lucideIcon: markRaw(Search),      to: '/classify' },
  { label: t('navbar.benchmark'),        lucideIcon: markRaw(ChartColumn), to: '/benchmark' },
  { label: t('navbar.quantum_advantage'),lucideIcon: markRaw(Atom),        to: '/quantum-advantage', isQuantum: true },
  { label: t('navbar.contact'),          lucideIcon: markRaw(Mail),        to: '/contact' },
]);

const toggleLanguage = () => {
  const next = locale.value === 'EN' ? 'AR' : 'EN';
  locale.value = next;
  Cookies.set(LANG_KEY, next, { expires: 365, path: '/' });
};

const toggleTheme = () => {
  isDark.value = !isDark.value;
  document.documentElement.classList.toggle('p-dark', isDark.value);
  Cookies.set(THEME_KEY, isDark.value ? 'dark' : 'light', { expires: 365, path: '/' });
};

const closeMenu = () => { menuOpen.value = false; };
</script>

<template>
  <header class="qnn-bar">
    <div class="qnn-inner">

      <!-- ── Logo ── -->
      <router-link to="/" class="qnn-logo" @click="closeMenu">
        <img src="/public/qnn_logo_final_no_text.svg" alt="QNN" />
      </router-link>

      <!-- ── Desktop nav ── -->
      <nav class="qnn-nav" :class="{ 'qnn-nav--open': menuOpen }">
        <router-link
          v-for="item in items"
          :key="item.to"
          :to="item.to"
          class="qnn-link"
          :class="{ 'qnn-link--quantum': item.isQuantum }"
          @click="closeMenu"
        >
          <component :is="item.lucideIcon" class="qnn-link-icon" />
          <span>{{ item.label }}</span>
        </router-link>
      </nav>

      <!-- ── Controls ── -->
      <div class="qnn-actions">
        <button class="qnn-lang-btn" @click="toggleLanguage">
          <Languages :size="14" />
          <span>{{ currentLang === 'EN' ? 'العربية' : 'English' }}</span>
        </button>

        <button class="qnn-icon-btn" @click="toggleTheme" :aria-label="isDark ? 'Light mode' : 'Dark mode'">
          <component :is="isDark ? markRaw(Sun) : markRaw(Moon)" :size="16" />
        </button>

        <button class="qnn-icon-btn qnn-burger" @click="menuOpen = !menuOpen" aria-label="Toggle menu">
          <component :is="menuOpen ? markRaw(X) : markRaw(Menu)" :size="19" />
        </button>
      </div>
    </div>

    <!-- Teal accent line -->
    <div class="qnn-accent-line" />
  </header>

  <!-- Page offset -->
  <div class="h-[85px]" />
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@500;600&family=DM+Sans:opsz,wght@9..40,400;9..40,500&display=swap');

/* ────────────────────────────────────────────
   Design tokens — logo palette
──────────────────────────────────────────── */
:root {
  --q-teal:        #2ab8b8;
  --q-teal-soft:   rgba(42, 184, 184, 0.12);
  --q-teal-glow:   rgba(42, 184, 184, 0.35);
  --q-navy:        #0d1f2d;
  --q-navy-mid:    #152536;

  --q-bar-bg:      rgba(255, 255, 255, 0.88);
  --q-bar-border:  rgba(13, 31, 45, 0.08);
  --q-text:        #0d1f2d;
  --q-muted:       #4a6678;
  --q-h:           58px;
}

.p-dark {
  --q-bar-bg:     rgba(9, 18, 28, 0.90);
  --q-bar-border: rgba(42, 184, 184, 0.08);
  --q-text:       #cde8ec;
  --q-muted:      #6a9aaa;
}

/* ────────────────────────────────────────────
   Bar shell
──────────────────────────────────────────── */
.qnn-bar {
  position: fixed;
  inset: 0 0 auto 0;
  z-index: 999;
  background: var(--q-bar-bg);
  backdrop-filter: blur(16px) saturate(180%);
  -webkit-backdrop-filter: blur(16px) saturate(180%);
  border-bottom: 1px solid var(--q-bar-border);
  font-family: 'DM Sans', sans-serif;
}

.qnn-inner {
  display: flex;
  align-items: center;
  height: var(--q-h);
  padding: 0 1.75rem;
  max-width: 1400px;
  margin: 0 auto;
  position: relative;
}

/* Teal gradient accent line at bottom */
.qnn-accent-line {
  height: 2px;
  background: linear-gradient(
    90deg,
    transparent      0%,
    var(--q-teal)   25%,
    #1eb0c8         55%,
    transparent    100%
  );
  opacity: 0.55;
}

.qnn-spacer {
  height: calc(var(--q-h) + 2px);
}

/* ────────────────────────────────────────────
   Logo
──────────────────────────────────────────── */
.qnn-logo {
  display: flex;
  align-items: center;
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
  transform: scale(1.06);
  filter: drop-shadow(0 0 8px var(--q-teal-glow));
}

/* ────────────────────────────────────────────
   Desktop nav (centered)
──────────────────────────────────────────── */
.qnn-nav {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 0.1rem;
}

.qnn-link {
  display: flex;
  align-items: center;
  gap: 0.38rem;
  padding: 0.42rem 0.8rem;
  border-radius: 7px;
  font-size: 0.855rem;
  font-weight: 500;
  color: var(--q-muted);
  text-decoration: none;
  position: relative;
  white-space: nowrap;
  transition: color 0.18s, background 0.18s;
}

/* Sliding teal underline */
.qnn-link::after {
  content: '';
  position: absolute;
  bottom: 3px;
  left: 50%;
  width: 55%;
  height: 1.5px;
  background: var(--q-teal);
  border-radius: 2px;
  transform: translateX(-50%) scaleX(0);
  transform-origin: center;
  transition: transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.qnn-link:hover,
.qnn-link.router-link-active {
  color: var(--q-text);
  background: var(--q-teal-soft);
}

.qnn-link:hover::after,
.qnn-link.router-link-active::after {
  transform: translateX(-50%) scaleX(1);
}

/* Quantum Advantage — special teal treatment */
.qnn-link--quantum {
  color: var(--q-teal);
  font-family: 'Oxanium', sans-serif;
  font-weight: 600;
  font-size: 0.84rem;
  letter-spacing: 0.015em;
}

.qnn-link--quantum::after {
  width: 75%;
  background: var(--q-teal);
}

.qnn-link--quantum:hover {
  color: var(--q-teal);
  background: var(--q-teal-soft);
}

.qnn-link-icon {
  width: 14px;
  height: 14px;
  opacity: 0.65;
  flex-shrink: 0;
  transition: opacity 0.18s;
}

.qnn-link:hover .qnn-link-icon,
.qnn-link.router-link-active .qnn-link-icon {
  opacity: 1;
}

.qnn-link--quantum .qnn-link-icon {
  opacity: 1;
  color: var(--q-teal);
}

/* ────────────────────────────────────────────
   Controls (right side)
──────────────────────────────────────────── */
.qnn-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  z-index: 2;
}

.qnn-lang-btn {
  display: flex;
  align-items: center;
  gap: 0.38rem;
  padding: 0.38rem 0.7rem;
  border: none;
  border-radius: 7px;
  background: transparent;
  color: var(--q-muted);
  font-family: 'DM Sans', sans-serif;
  font-size: 0.82rem;
  font-weight: 500;
  cursor: pointer;
  transition: color 0.18s, background 0.18s;
}

.qnn-icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: var(--q-muted);
  cursor: pointer;
  transition: color 0.18s, background 0.18s;
}

.qnn-lang-btn:hover,
.qnn-icon-btn:hover {
  color: var(--q-teal);
  background: var(--q-teal-soft);
}

/* Hamburger — hidden on desktop */
.qnn-burger {
  display: none;
  border-radius: 7px;
  width: 36px;
  height: 36px;
}

/* ────────────────────────────────────────────
   Mobile / Zoom breakpoint
──────────────────────────────────────────── */
@media (max-width: 960px) {
  .qnn-burger {
    display: flex;
  }

  .qnn-nav {
    /* Reset desktop centering */
    position: fixed;
    top: calc(var(--q-h) + 2px);
    left: 0;
    right: 0;
    transform: none;
    flex-direction: column;
    align-items: stretch;
    gap: 0.25rem;
    padding: 0.75rem 1rem 1.25rem;
    background: var(--q-bar-bg);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border-bottom: 1px solid var(--q-bar-border);
    z-index: 998;

    /* Hidden by default */
    opacity: 0;
    pointer-events: none;
    transform: translateY(-6px);
    transition: opacity 0.2s ease, transform 0.2s ease;
  }

  .qnn-nav--open {
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0);
  }

  .qnn-link {
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: 0.93rem;
  }

  .qnn-link::after {
    display: none;
  }

  .qnn-link:hover,
  .qnn-link.router-link-active {
    background: var(--q-teal-soft);
  }
}
</style>