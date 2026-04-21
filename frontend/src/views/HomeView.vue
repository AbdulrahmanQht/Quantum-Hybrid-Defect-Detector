<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'

const { tm, rt, locale } = useI18n({ useScope: 'global' })

const currentIndex = ref(0)

// 1. Use 'tm' (translate message) to get the raw array/object from i18n
// 2. Wrap it in a computed property so it reacts automatically to locale changes
const cards = computed(() => {
  const data = tm('home.slider')
  return Array.isArray(data) ? data : []
})

const nextCard = () => {
  if (cards.value.length > 0) {
    currentIndex.value = (currentIndex.value + 1) % cards.value.length
  }
}

let interval = null
onMounted(() => {
  interval = setInterval(nextCard, 5000)
})

onUnmounted(() => {
  if (interval) clearInterval(interval)
})
</script>

<template>
  <div class="min-h-screen px-4 md:px-16 py-16 bg-[var(--q-bar-bg)] dark:bg-[var(--q-navy)] text-[var(--q-text)] dark:text-[var(--q-bar-bg)] space-y-24">

    <section class="grid md:grid-cols-2 gap-16 items-center">
      <div class="flex flex-col justify-center space-y-6">
        <h1 class="text-5xl font-bold text-[var(--q-teal)]">{{ rt($t('home.hero.title')) }}</h1>
        <h2 class="text-2xl font-semibold text-[var(--q-muted)]">{{ rt($t('home.hero.subtitle')) }}</h2>
        <p class="text-lg text-[var(--q-muted)] leading-relaxed">{{ rt($t('home.hero.description')) }}</p>
        <div class="flex gap-4 flex-wrap mt-8">
          <a href="/classify">
            <button class="px-8 py-4 bg-[var(--q-teal)] text-[var(--q-bar-bg)] font-bold rounded-xl shadow-lg hover:shadow-[0_0_20px_var(--q-teal-glow)] hover:scale-105 transition transform duration-300">
              {{ rt($t('home.hero.button')) }}
            </button>
          </a>
        </div>
      </div>
      <div class="relative rounded-xl overflow-hidden h-64 w-full shadow">
        <img src="/homeimagelight.png" class="block dark:hidden w-full h-full object-cover" alt="Hero Light"/>
        <img src="/homepageimage.png" class="hidden dark:block w-full h-full object-cover" alt="Hero Dark"/>
      </div>
    </section>

    <section class="flex justify-center">
      <transition name="slide-fade" mode="out-in">
        <div
          v-if="cards.length > 0"
          :key="currentIndex"
          @click="nextCard"
          class="cursor-pointer w-full md:w-3/4 lg:w-1/2 bg-white dark:bg-[var(--q-navy-mid)] rounded-xl p-10 shadow-lg hover:shadow-2xl transition-shadow duration-300"
        >
          <h3 class="text-2xl font-semibold mb-4 text-[var(--q-teal)]">
            {{ rt(cards[currentIndex].title) }}
          </h3>
          <p class="text-[var(--q-muted)] text-lg leading-relaxed">
            {{ rt(cards[currentIndex].text) }}
          </p>
        </div>
      </transition>
    </section>

    <section class="text-center text-[var(--q-muted)] text-sm leading-relaxed max-w-3xl mx-auto space-y-4">
      <p>{{ rt($t('home.credits.value')) }}</p>
      <p>{{ rt($t('home.credits.team')) }}</p>
    </section>

  </div>
</template>