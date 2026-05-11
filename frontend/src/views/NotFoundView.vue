<script setup>
import { markRaw } from 'vue'
import { RouterLink } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { Search, ChartColumn, Atom, Mail } from 'lucide-vue-next'
import Card from 'primevue/card'
import Button from 'primevue/button'
import Divider from 'primevue/divider'

const { t, locale } = useI18n()

const navLinks = [
    { to: '/classify', labelKey: 'navbar.classify', icon: markRaw(Search) },
    { to: '/benchmark', labelKey: 'navbar.benchmark', icon: markRaw(ChartColumn) },
    { to: '/quantum-advantage', labelKey: 'navbar.quantum_advantage', icon: markRaw(Atom) },
    { to: '/contact', labelKey: 'navbar.contact', icon: markRaw(Mail) },
]
</script>

<template>
    <div class="nf-page" :class="{ 'lang-ar': locale === 'ar' }">
        <!-- Animated teal grid -->
        <div class="nf-grid" aria-hidden="true" />

        <!-- Floating orbs -->
        <div class="nf-orbs" aria-hidden="true">
            <span class="nf-orb nf-orb--1" />
            <span class="nf-orb nf-orb--2" />
            <span class="nf-orb nf-orb--3" />
        </div>

        <!-- Scanline overlay -->
        <div class="nf-scanlines" aria-hidden="true" />

        <div class="nf-wrapper">

            <!-- Glitchy 404 -->
            <div class="nf-code" aria-hidden="true">
                <span class="nf-code__text glitch" data-text="404">404</span>
            </div>

            <!-- Main card -->
            <Card class="nf-card q-glass">
                <template #content>
                    <div class="nf-card__body" :dir="locale === 'AR' ? 'rtl' : 'ltr'">

                        <!-- Icon badge -->
                        <div class="nf-badge">
                            <i class="pi pi-wifi text-2xl" style="color: var(--q-teal)" />
                        </div>

                        <!-- Text block -->
                        <div class="nf-text-block">
                            <h1 class="nf-title">{{ t('notFound.title') }}</h1>
                            <p class="nf-subtitle">{{ t('notFound.subtitle') }}</p>
                        </div>

                        <Divider />

                        <!-- Nav suggestions -->
                        <div class="nf-nav-block">
                            <p class="nf-nav-label">{{ t('notFound.whereLabel') }}</p>
                            <div class="nf-nav-grid">
                                <RouterLink v-for="link in navLinks" :key="link.to" :to="link.to" class="nf-nav-link">
                                    <Button :label="t(link.labelKey)" variant="outlined" class="nf-nav-btn w-full"
                                        size="small">
                                        <template #icon>
                                            <component :is="link.icon" :size="13" class="nf-nav-icon" />
                                        </template>
                                    </Button>
                                </RouterLink>
                            </div>
                        </div>

                        <Divider />

                        <!-- Home CTA -->
                        <RouterLink to="/" class="nf-home-link">
                            <Button :label="t('notFound.cta')" icon="pi pi-home" class="nf-home-btn" size="large" />
                        </RouterLink>

                    </div>
                </template>
            </Card>

            <!-- Footer note -->
            <p class="nf-footer-note">
                <i class="pi pi-info-circle me-1" />
                {{ t('notFound.footerNote') }}
                <RouterLink to="/contact" class="nf-footer-link">
                    {{ t('notFound.footerLink') }}
                </RouterLink>
            </p>

        </div>
    </div>
</template>

<style scoped>
/* ────────────────────────────────────────────
   Page shell
──────────────────────────────────────────── */
.nf-page {
    position: relative;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    padding: 2rem 1rem;

    /* Light: clean white-teal gradient */
    background:
        radial-gradient(circle at 20% 20%, rgba(42, 184, 184, 0.14) 0%, transparent 40%),
        radial-gradient(circle at 80% 80%, rgba(42, 184, 184, 0.09) 0%, transparent 35%),
        linear-gradient(180deg, #f5fafb, #ffffff);
    color: var(--q-text);
    font-family: var(--q-font-body);
}

/* Dark override — uses .p-dark on <html> */
:global(.p-dark) .nf-page {
    background:
        radial-gradient(circle at 20% 20%, rgba(42, 184, 184, 0.18) 0%, transparent 40%),
        radial-gradient(circle at 80% 80%, rgba(42, 184, 184, 0.10) 0%, transparent 35%),
        linear-gradient(180deg, #08131d 0%, #0d1f2d 100%);
}

/* ────────────────────────────────────────────
   Animated grid
──────────────────────────────────────────── */
.nf-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(var(--q-teal) 1px, transparent 1px),
        linear-gradient(90deg, var(--q-teal) 1px, transparent 1px);
    background-size: 52px 52px;
    opacity: 0.04;
    animation: grid-drift 24s linear infinite;
}

@keyframes grid-drift {
    from {
        background-position: 0 0;
    }

    to {
        background-position: 52px 52px;
    }
}

/* ────────────────────────────────────────────
   Ambient orbs
──────────────────────────────────────────── */
.nf-orbs {
    position: absolute;
    inset: 0;
    pointer-events: none;
}

.nf-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(60px);
    opacity: 0.18;
    background: var(--q-teal);
    animation: orb-pulse 8s ease-in-out infinite alternate;
}

.nf-orb--1 {
    width: 340px;
    height: 340px;
    top: -80px;
    left: -100px;
    animation-delay: 0s;
}

.nf-orb--2 {
    width: 220px;
    height: 220px;
    bottom: -60px;
    right: -60px;
    animation-delay: 3s;
}

.nf-orb--3 {
    width: 160px;
    height: 160px;
    top: 40%;
    right: 10%;
    animation-delay: 6s;
    opacity: 0.10;
}

@keyframes orb-pulse {
    from {
        transform: scale(1);
        opacity: 0.14;
    }

    to {
        transform: scale(1.18);
        opacity: 0.22;
    }
}

/* ────────────────────────────────────────────
   Scanlines (subtle CRT texture)
──────────────────────────────────────────── */
.nf-scanlines {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(0deg,
            transparent,
            transparent 3px,
            rgba(0, 0, 0, 0.015) 3px,
            rgba(0, 0, 0, 0.015) 4px);
}

/* ────────────────────────────────────────────
   Content wrapper
──────────────────────────────────────────── */
.nf-wrapper {
    position: relative;
    z-index: 10;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    width: 100%;
    max-width: 460px;
}

/* ────────────────────────────────────────────
   Glitchy 404
──────────────────────────────────────────── */
.nf-code {
    line-height: 1;
    margin-bottom: -1.75rem;
    user-select: none;
}

.nf-code__text {
    display: block;
    font-family: var(--q-font-display);
    /* Oxanium */
    font-size: clamp(5.5rem, 22vw, 8.5rem);
    font-weight: 700;
    letter-spacing: -0.04em;
    color: var(--q-teal);
    opacity: 0.13;
}

/* Glitch pseudo-elements */
.glitch {
    position: relative;
}

.glitch::before,
.glitch::after {
    content: attr(data-text);
    position: absolute;
    top: 0;
    left: 0;
    font-size: inherit;
    font-weight: inherit;
    letter-spacing: inherit;
    font-family: inherit;
    color: var(--q-teal);
    opacity: 0.12;
}

.glitch::before {
    clip-path: polygon(0 0, 100% 0, 100% 38%, 0 38%);
    transform: translateX(-4px);
    color: #2ab8b8;
    animation: glitch-top 3.5s infinite steps(2);
}

.glitch::after {
    clip-path: polygon(0 62%, 100% 62%, 100% 100%, 0 100%);
    transform: translateX(4px);
    color: #08b5a0;
    animation: glitch-bot 3.5s infinite steps(2);
}

@keyframes glitch-top {

    0%,
    92% {
        transform: translateX(-4px);
    }

    93% {
        transform: translateX(6px) skewX(-12deg);
    }

    95% {
        transform: translateX(-6px) skewX(10deg);
    }

    100% {
        transform: translateX(-4px);
    }
}

@keyframes glitch-bot {

    0%,
    92% {
        transform: translateX(4px);
    }

    93% {
        transform: translateX(-6px) skewX(12deg);
    }

    95% {
        transform: translateX(6px) skewX(-10deg);
    }

    100% {
        transform: translateX(4px);
    }
}

/* ────────────────────────────────────────────
   Card
──────────────────────────────────────────── */
.nf-card {
    width: 100%;
    border-radius: var(--q-radius-md) !important;
    border: 1.5px solid var(--q-bar-border) !important;
    background: var(--q-surface-strong) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow:
        0 0 0 1px var(--q-teal-soft),
        0 24px 60px -12px rgba(13, 31, 45, 0.10);
}

:global(.p-dark) .nf-card {
    box-shadow:
        0 0 0 1px var(--q-teal-soft),
        0 24px 60px -12px rgba(0, 0, 0, 0.35);
}

/* Remove default PrimeVue card padding override */
.nf-card :deep(.p-card-body) {
    padding: 1.75rem;
}

.nf-card :deep(.p-card-content) {
    padding: 0;
}

.nf-card__body {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.25rem;
    text-align: center;
}

/* ────────────────────────────────────────────
   Badge icon
──────────────────────────────────────────── */
.nf-badge {
    width: 3.25rem;
    height: 3.25rem;
    border-radius: 50%;
    background: var(--q-teal-soft);
    border: 1.5px solid var(--q-teal-glow);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 16px var(--q-teal-soft);
}

/* ────────────────────────────────────────────
   Typography
──────────────────────────────────────────── */
.nf-title {
    font-family: var(--q-font-display);
    font-size: 1.375rem;
    font-weight: 700;
    color: var(--q-text);
    margin: 0;
    letter-spacing: -0.02em;
    text-align: center;
}

.nf-subtitle {
    font-size: 0.8125rem;
    color: var(--q-muted);
    margin: 0.35rem 0 0;
    line-height: 1.65;
    text-align: center;
}

.nf-text-block {
    display: flex;
    flex-direction: column;
}

/* ────────────────────────────────────────────
   Nav suggestions
──────────────────────────────────────────── */
.nf-nav-block {
    width: 100%;
}

.nf-nav-label {
    font-size: 0.6875rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--q-muted);
    margin: 0 0 0.625rem;
    text-align: center;
}

.nf-nav-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
}

.nf-nav-link {
    text-decoration: none;
}

.nf-nav-btn {
    justify-content: flex-start !important;
    font-size: 0.8125rem !important;
    font-family: var(--q-font-body) !important;
    border-color: var(--q-bar-border) !important;
    color: var(--q-text) !important;
    transition: border-color 0.2s, background 0.2s !important;
}

.nf-nav-btn:hover {
    border-color: var(--q-teal) !important;
    background: var(--q-teal-soft) !important;
    color: var(--q-teal) !important;
}

/* ────────────────────────────────────────────
   Home CTA
──────────────────────────────────────────── */
.nf-home-link {
    text-decoration: none;
}

.nf-home-btn {
    font-family: var(--q-font-display) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    background: var(--q-teal) !important;
    border-color: var(--q-teal) !important;
    box-shadow: 0 4px 20px var(--q-teal-glow);
    transition: box-shadow 0.2s, transform 0.15s !important;
}

.nf-home-btn:hover {
    box-shadow: 0 6px 28px var(--q-teal-glow) !important;
    transform: translateY(-1px);
}

/* ────────────────────────────────────────────
   Footer note
──────────────────────────────────────────── */
.nf-footer-note {
    font-size: 0.75rem;
    color: var(--q-muted);
    margin-top: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    flex-wrap: wrap;
    justify-content: center;
}

.nf-footer-link {
    color: var(--q-teal);
    text-decoration: none;
    font-weight: 500;
}

.nf-footer-link:hover {
    text-decoration: underline;
}

/* ────────────────────────────────────────────
   RTL adjustments
──────────────────────────────────────────── */

/* Direction on the card body for general text flow */
.lang-ar .nf-card__body {
    direction: rtl;
}

/* Title, subtitle, whereLabel always centered regardless of language */
.lang-ar .nf-title,
.lang-ar .nf-subtitle,
.lang-ar .nf-nav-label {
    text-align: center !important;
    direction: ltr;

}

.lang-ar .nf-nav-btn {
    direction: rtl !important;
}

.lang-ar .nf-nav-btn :deep(.p-button-label) {
    text-align: right;
    flex: 1;
}

/* Reset the icon's margin so gap appears on the correct (left) side */
.lang-ar .nf-nav-btn :deep(.p-button-icon) {
    margin-right: 0 !important;
    margin-left: var(--p-button-icon-gap, 0.5rem) !important;
}

.lang-ar .nf-footer-note {
    direction: rtl;
    text-align: center;
}
</style>