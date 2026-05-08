<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { ArrowRight, Github, Linkedin, Microscope, ShieldCheck, Zap, Users } from 'lucide-vue-next'
import heroLight from '../assets/homeimagelight.png'
import heroDark from '../assets/homepageimage.png'
const { t, tm, locale } = useI18n({ useScope: 'global' })

const currentIndex = ref(0)

const slides = computed(() => {
  const data = tm('home.slider')
  return Array.isArray(data) ? data : []
})

const highlights = computed(() => {
  const data = tm('home.highlights')
  return Array.isArray(data) ? data : []
})

const supervisors = computed(() => {
  const data = tm('home.team.supervisors')
  return Array.isArray(data) ? data : []
})

const researchers = computed(() => {
  const data = tm('home.team.researchers')
  return Array.isArray(data) ? data : []
})

const stats = computed(() => {
  const data = tm('home.stats')
  return Array.isArray(data) ? data : []
})

const isArabic = computed(() => locale.value === 'AR')

const nextCard = () => {
  if (slides.value.length > 0) {
    currentIndex.value = (currentIndex.value + 1) % slides.value.length
  }
}

let interval = null

onMounted(() => {
  interval = setInterval(() => {
    if (!document.hidden) nextCard()
  }, 5000)
})

onUnmounted(() => {
  if (interval) clearInterval(interval)
})
</script>

<template>
  <div class="home-page px-4 pb-20 pt-6 md:px-8 xl:px-14" :class="{ 'home-page--ar': isArabic }">
    <section class="hero-shell">
      <div class="hero-copy">
        <Tag :value="t('home.hero.badge')" rounded class="hero-badge" />
        <h1 class="hero-title">
          {{ t('home.hero.title') }}
        </h1>
        <h2 class="hero-subtitle">
          {{ t('home.hero.subtitle') }}
        </h2>
        <p class="hero-description">
          {{ t('home.hero.description') }}
        </p>

        <div class="hero-actions">
          <RouterLink to="/classify">
            <Button :label="t('home.hero.button')" class="hero-primary-btn" size="large">
              <template #icon>
                <ArrowRight :class="['h-4 w-4', { 'hero-arrow--rtl': isArabic }]" />
              </template>
            </Button>
          </RouterLink>

          <RouterLink to="/benchmark">
            <Button :label="t('home.hero.secondaryButton')" severity="secondary" outlined size="large" />
          </RouterLink>

          <RouterLink to="/quantum-advantage">
            <Button :label="t('home.hero.tertiaryButton')" severity="contrast" text size="large"
              class="hero-tertiary-btn" />
          </RouterLink>
        </div>
      </div>

      <div class="hero-visual">
        <div class="hero-visual-card">
          <div class="hero-image-wrap">
            <img :src="heroLight" class="block dark:hidden hero-image" alt="Hero Light">
            <img :src="heroDark" class="hidden dark:block hero-image" alt="Hero Dark">
          </div>

          <div class="hero-stat-grid">
            <div v-for="item in stats" :key="item.label" class="hero-stat-card">
              <span class="hero-stat-value">{{ item.value }}</span>
              <span class="hero-stat-label">{{ item.label }}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section-grid">
      <Card class="glass-card intro-card">
        <template #content>
          <div class="section-heading">
            <div>
              <span class="eyebrow">{{ t('home.sections.overviewEyebrow') }}</span>
              <h3 class="section-title">
                {{ t('home.sections.overviewTitle') }}
              </h3>
            </div>
            <Microscope class="section-icon" />
          </div>
          <p class="section-text">
            {{ t('home.sections.overviewText') }}
          </p>
        </template>
      </Card>

      <transition name="slide-fade" mode="out-in">
        <Card v-if="slides.length" :key="currentIndex" class="glass-card slider-card" @click="nextCard">
          <template #content>
            <div class="section-heading">
              <div>
                <span class="eyebrow">{{ t('home.sections.sliderEyebrow') }}</span>
                <h3 class="section-title">
                  {{ slides[currentIndex].title }}
                </h3>
              </div>
              <Zap class="section-icon" />
            </div>
            <p class="section-text">
              {{ slides[currentIndex].text }}
            </p>
          </template>
        </Card>
      </transition>
    </section>

    <section class="highlights-section">
      <div class="section-heading section-heading--highlights mb-6" :class="{ 'section-heading--rtl': isArabic }">
        <div class="section-heading__text" :dir="isArabic ? 'rtl' : 'ltr'">
          <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">
            {{ t('home.sections.highlightsEyebrow') }}
          </span>
          <h3 class="section-title">
            {{ t('home.sections.highlightsTitle') }}
          </h3>
        </div>
        <ShieldCheck class="section-icon" />
      </div>

      <div class="highlights-grid">
        <Card v-for="item in highlights" :key="item.title" class="feature-card">
          <template #content>
            <h4 class="feature-title">
              {{ item.title }}
            </h4>
            <p class="feature-text">
              {{ item.text }}
            </p>
          </template>
        </Card>
      </div>
    </section>

    <section class="team-section">
      <div class="section-heading section-heading--team mb-6" :class="{ 'section-heading--rtl': isArabic }">
        <div class="section-heading__text" :dir="isArabic ? 'rtl' : 'ltr'">
          <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">
            {{ t('home.team.eyebrow') }}
          </span>
          <h3 class="section-title">
            {{ t('home.team.title') }}
          </h3>
        </div>
        <Users class="section-icon" />
      </div>

      <div class="team-group">
        <h4 class="team-group-title">
          {{ t('home.team.supervisorsTitle') }}
        </h4>
        <div class="team-grid team-grid--supervisors">
          <Card v-for="person in supervisors" :key="person.name" class="team-card">
            <template #content>
              <div class="team-card-top">
                <div class="team-avatar">
                  {{ person.name.charAt(4) }}
                </div>
                <div>
                  <h5 class="team-name">
                    {{ person.name }}
                  </h5>
                  <p class="team-role">
                    {{ person.role }}
                  </p>
                </div>
              </div>
            </template>
          </Card>
        </div>
      </div>

      <div class="team-group">
        <h4 class="team-group-title">
          {{ t('home.team.researchersTitle') }}
        </h4>
        <div class="team-grid">
          <Card v-for="person in researchers" :key="person.name" class="team-card">
            <template #content>
              <div class="team-card-top">
                <div class="team-avatar">
                  {{ person.name.charAt(0) }}
                </div>
                <div>
                  <h5 class="team-name">
                    {{ person.name }}
                  </h5>
                  <p class="team-role">
                    {{ person.role }}
                  </p>
                </div>
              </div>

              <div class="team-links">
                <a :href="person.github" target="_blank" rel="noreferrer" class="team-link">
                  <Github class="h-4 w-4" />
                  <span>{{ t('home.team.github') }}</span>
                </a>
                <a :href="person.linkedin" target="_blank" rel="noreferrer" class="team-link">
                  <Linkedin class="h-4 w-4" />
                  <span>{{ t('home.team.linkedin') }}</span>
                </a>
              </div>
            </template>
          </Card>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.home-page {
  color: var(--q-text);
}

.hero-shell {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(420px, 0.95fr);
  gap: 2.5rem;
  align-items: center;
  min-height: calc(100vh - 110px);
  padding: 2rem 0 3rem;
}

.home-page--ar .hero-visual {
  order: -1;
}

.hero-copy {
  max-width: 720px;
}

.hero-badge {
  margin-bottom: 1rem;
  background: var(--q-teal-soft);
  color: var(--q-teal);
  border: 1px solid rgba(42, 184, 184, 0.18);
}

.hero-title {
  font-size: clamp(2.8rem, 6vw, 5rem);
  line-height: 0.95;
  font-weight: 800;
  color: var(--q-text);
  letter-spacing: -0.04em;
  margin-bottom: 1rem;
}

.hero-subtitle {
  font-size: clamp(1.1rem, 2vw, 1.55rem);
  font-weight: 700;
  color: var(--q-teal);
  margin-bottom: 1rem;
}

.hero-description {
  font-size: 1.05rem;
  line-height: 1.9;
  color: var(--q-muted);
  max-width: 640px;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.9rem;
  margin-top: 2rem;
}

.hero-primary-btn {
  background: var(--q-teal) !important;
  border-color: var(--q-teal) !important;
  color: white !important;
}


.hero-visual-card,
.glass-card,
.feature-card,
.team-card,
.closing-panel {
  border: 1px solid var(--q-bar-border);
  background: rgba(255, 255, 255, 0.64);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: 0 20px 60px rgba(13, 31, 45, 0.08);
}

.hero-tertiary-btn {
  color: var(--q-teal) !important;
}

.p-dark .hero-visual-card,
.p-dark .glass-card,
.p-dark .feature-card,
.p-dark .team-card,
.p-dark .closing-panel {
  background: rgba(14, 28, 41, 0.74);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
}

.hero-visual-card {
  padding: 1rem;
  border-radius: 32px;
}

.hero-image-wrap {
  overflow: hidden;
  border-radius: 24px;
  min-height: 340px;
}

.hero-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.hero-stat-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.85rem;
  margin-top: 1rem;
}

.hero-stat-card {
  padding: 1rem;
  border-radius: 18px;
  background: rgba(42, 184, 184, 0.08);
  text-align: center;
}

.hero-stat-value {
  display: block;
  font-size: 1.15rem;
  font-weight: 800;
  color: var(--q-text);
}

.hero-stat-label {
  display: block;
  margin-top: 0.35rem;
  font-size: 0.86rem;
  color: var(--q-muted);
}

.section-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}

.glass-card,
.feature-card,
.team-card {
  border-radius: 28px;
}

.section-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}

.eyebrow {
  display: inline-block;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--q-teal);
  margin-bottom: 0.5rem;
}

.section-title {
  font-size: 1.6rem;
  font-weight: 800;
  color: var(--q-text);
}

.section-icon {
  width: 20px;
  height: 20px;
  color: var(--q-teal);
  flex-shrink: 0;
}

.section-text {
  margin-top: 1rem;
  line-height: 1.9;
  color: var(--q-muted);
}

.slider-card {
  cursor: pointer;
}

.highlights-section,
.team-section {
  margin-top: 4.5rem;
}

.highlights-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
}

.feature-title {
  font-size: 1.1rem;
  font-weight: 800;
  color: var(--q-text);
  margin-bottom: 0.7rem;
}

.feature-text {
  line-height: 1.8;
  color: var(--q-muted);
}

.team-group {
  margin-top: 1.5rem;
}

.team-group-title {
  font-size: 1.1rem;
  font-weight: 800;
  color: var(--q-teal);
  margin-bottom: 1rem;
}

.team-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
}

.team-grid--supervisors {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.team-card-top {
  display: flex;
  align-items: center;
  gap: 0.9rem;
}

.team-avatar {
  display: grid;
  place-items: center;
  width: 52px;
  height: 52px;
  border-radius: 16px;
  background: linear-gradient(135deg, var(--q-teal), rgba(42, 184, 184, 0.55));
  color: white;
  font-weight: 800;
  font-size: 1.1rem;
}

.team-name {
  font-size: 1rem;
  font-weight: 800;
  color: var(--q-text);
}

.team-role {
  font-size: 0.92rem;
  color: var(--q-muted);
  margin-top: 0.2rem;
}

.team-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 1rem;
}

.team-link {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.6rem 0.9rem;
  border-radius: 999px;
  text-decoration: none;
  color: var(--q-text);
  background: var(--q-teal-soft);
  transition: transform 0.18s ease, opacity 0.18s ease;
}

.team-link:hover {
  transform: translateY(-2px);
  opacity: 0.92;
}

.closing-panel {
  max-width: 980px;
  margin: 4.5rem auto 0;
  padding: 1.5rem;
  border-radius: 26px;
  text-align: center;
  color: var(--q-muted);
  line-height: 1.9;
}

.slide-fade-enter-active,
.slide-fade-leave-active {
  transition: all 0.35s ease;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  opacity: 0;
  transform: translateY(12px);
}


@media (max-width: 1200px) {

  .highlights-grid,
  .team-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 960px) {

  .hero-shell,
  .section-grid {
    grid-template-columns: 1fr;
  }

  .hero-shell {
    min-height: auto;
    padding-top: 1rem;
  }

  .team-grid--supervisors,
  .hero-stat-grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 640px) {

  .highlights-grid,
  .team-grid,
  .team-grid--supervisors,
  .hero-stat-grid {
    grid-template-columns: 1fr;
  }

  .hero-title {
    font-size: 2.7rem;
  }

  .hero-description,
  .section-text,
  .feature-text {
    line-height: 1.75;
  }
}

.home-page--ar .hero-copy {
  direction: rtl;
  text-align: right;
}

.home-page--ar .hero-actions {
  direction: rtl;
  justify-content: flex-start;
}

.home-page--ar .hero-actions a {
  direction: rtl;
}

.section-heading--rtl {
  flex-direction: row-reverse;
}

.section-heading__text[dir='rtl'] {
  text-align: right;
}

.eyebrow--ar {
  letter-spacing: 0;
  text-transform: none;
}

.hero-arrow--rtl {
  transform: scaleX(-1);
}
</style>
