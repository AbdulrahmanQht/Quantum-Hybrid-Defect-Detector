<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { Github, Linkedin, Mail } from 'lucide-vue-next'

const { t, tm, locale } = useI18n({ useScope: 'global' })

const teamLinks = tm('footer.teamLinks')
const isArabic = computed(() => locale.value === 'AR')
</script>

<template>
  <footer class="site-footer" :class="{ 'site-footer--ar': isArabic }" :dir="isArabic ? 'rtl' : 'ltr'">
    <div class="site-footer__top">
      <div class="site-footer__grid">
        <div class="site-footer__col" :dir="isArabic ? 'rtl' : 'ltr'">
          <h4 class="site-footer__heading">{{ t('footer.navigationTitle') }}</h4>
          <RouterLink to="/" class="site-footer__link">{{ t('navbar.home') }}</RouterLink>
          <RouterLink to="/classify" class="site-footer__link">{{ t('navbar.classify') }}</RouterLink>
          <RouterLink to="/benchmark" class="site-footer__link">{{ t('navbar.benchmark') }}</RouterLink>
          <RouterLink to="/quantum-advantage" class="site-footer__link">{{ t('navbar.quantum_advantage') }}</RouterLink>
          <RouterLink to="/contact" class="site-footer__link">{{ t('navbar.contact') }}</RouterLink>
        </div>

        <div class="site-footer__col" :dir="isArabic ? 'rtl' : 'ltr'">
          <h4 class="site-footer__heading">{{ t('footer.teamTitle') }}</h4>
          <a
            v-for="person in teamLinks"
            :key="person.name"
            :href="person.linkedin"
            target="_blank"
            rel="noreferrer"
            class="site-footer__link"
          >
            <Linkedin :size="14" />
            <span>{{ person.name }}</span>
          </a>
        </div>

        <div class="site-footer__col" :dir="isArabic ? 'rtl' : 'ltr'">
          <h4 class="site-footer__heading">{{ t('footer.contactTitle') }}</h4>
          <RouterLink to="/contact" class="site-footer__link">
            <Mail :size="14" />
            <span>{{ t('footer.emailLabel') }}</span>
          </RouterLink>
          <a :href="t('footer.githubHref')" target="_blank" rel="noreferrer" class="site-footer__link">
            <Github :size="14" />
            <span>{{ t('footer.githubLabel') }}</span>
          </a>
        </div>
      </div>
    </div>

    <div class="site-footer__bottom">
      <div class="site-footer__bottom-inner">
        <span>{{ t('footer.copy') }}</span>
      </div>
    </div>
  </footer>
</template>

<style scoped>
.site-footer {
  margin-top: 5rem;
  border-top: 1px solid var(--q-bar-border);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(245, 250, 251, 0.95));
}

.p-dark .site-footer {
  background:
    linear-gradient(180deg, rgba(8, 19, 29, 0.5), rgba(13, 31, 45, 0.92));
}

.site-footer__top {
  width: 100%;
  padding: 3rem 1.25rem 2rem;
}

.site-footer__grid {
  /* Reducing this from 1400px keeps the columns from spreading too far */
  max-width: 1100px; 
  margin: 0 auto;
  display: grid;
  /* Using specific fractions to give the middle column (Research Team) more room */
  grid-template-columns: 1fr 1.5fr 1fr;
  /* Reverting to space-between now that the container is narrower */
  justify-content: space-between; 
  gap: 2rem;
}

.site-footer__col {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
  /* Aligns to start (Right for AR / Left for EN) */
  align-items: flex-start;
  text-align: start;
}
.site-footer:not([dir="rtl"]) .site-footer__link:hover {
  transform: translateX(4px);
}

.site-footer[dir="rtl"] .site-footer__link:hover {
  transform: translateX(-4px);
}

.site-footer__heading {
  font-size: 0.95rem;
  font-weight: 800;
  color: var(--q-teal);
  margin-bottom: 0.35rem;
  text-align: start; 
}

.site-footer__link {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  width: fit-content;
  text-decoration: none;
  color: var(--q-muted);
  transition: all 0.18s ease;
}

/* Logical hover effect: works for both EN and AR */
.site-footer__link:hover {
  color: var(--q-text);
  margin-inline-start: 4px; 
}

/* 4. REMOVE all the .site-footer--ar blocks at the bottom of your file */
/* They are no longer needed because :dir on the parent handles it all! */


.site-footer__bottom {
  border-top: 1px solid var(--q-bar-border);
  padding: 1rem 1.25rem;
}

.site-footer__bottom-inner {
  max-width: 1400px;
  margin: 0 auto;
  color: var(--q-muted);
  font-size: 0.9rem;
  /* Aligns the copyright text to the right in Arabic, left in English */
  text-align: start;
}

@media (max-width: 960px) {
  .site-footer__grid {
    grid-template-columns: 1fr;
    gap: 2rem;
  }
}
.site-footer--ar .site-footer__heading {
  direction: rtl;
  text-align: right;
  unicode-bidi: plaintext;
}

.site-footer--ar .site-footer__col {
  align-items: flex-start;
}

.site-footer--ar .site-footer__heading {
  width: 100%;
  direction: rtl;
  text-align: right;
  unicode-bidi: plaintext;
  align-self: stretch;
}

.site-footer--ar .site-footer__bottom-inner {
  direction: rtl;
  text-align: right;
  unicode-bidi: plaintext;
}
.site-footer--ar .site-footer__bottom-inner {
  direction: rtl;
  text-align: right;
  unicode-bidi: plaintext;
}
</style>
