<script setup>
import { ref, computed, markRaw } from "vue";
import Cookies from 'js-cookie';
import { useI18n } from 'vue-i18n';
import { Atom, Home, Search, ChartColumn, Info, Mail, Languages, Sun, Moon } from 'lucide-vue-next';

const { t, locale } = useI18n();

const LANG_KEY = 'app_lang';
const THEME_KEY = 'theme';

const currentLang = computed(() => locale.value);
const isDark = ref(Cookies.get(THEME_KEY) === 'dark');

// 2. Updated items array to use Component references
const items = computed(() => [
  { label: t('navbar.home'), lucideIcon: markRaw(Home), to: '/' },
  { label: t('navbar.classify'), lucideIcon: markRaw(Search), to: '/classify' },
  { label: t('navbar.benchmark'), lucideIcon: markRaw(ChartColumn), to: '/benchmark' },
  { 
    label: t('navbar.quantum_advantage'), 
    lucideIcon: markRaw(Atom), 
    to: '/quantum-advantage',
    isQuantum: true 
  },
  { label: t('navbar.about'), lucideIcon: markRaw(Info), to: '/about' },
  { label: t('navbar.contact'), lucideIcon: markRaw(Mail), to: '/contact' }
]);

const toggleLanguage = () => {
  const nextLang = locale.value === 'EN' ? 'AR' : 'EN';
  locale.value = nextLang;
  Cookies.set(LANG_KEY, nextLang, { expires: 365, path: '/' });
};

const toggleTheme = () => {
  isDark.value = !isDark.value;
  const themeValue = isDark.value ? 'dark' : 'light';
  document.documentElement.classList.toggle('p-dark', isDark.value);
  Cookies.set(THEME_KEY, themeValue, { expires: 365, path: '/' });
};
</script>

<template>
  <div class="card">
    <MenuBar :model="items" class="px-6 border-none rounded-none shadow-md">
      <template #item="{ item, props }">
        <router-link v-if="item.to" :to="item.to" v-bind="props.action" class="flex items-center p-3">
          
          <component 
            v-if="item.lucideIcon"
            :is="item.lucideIcon" 
            :class="[
              'mr-2 w-4 h-4', 
              item.isQuantum ? 'text-primary' : 'text-surface-600 dark:text-surface-400'
            ]" 
          />

          <span class="font-medium">{{ item.label }}</span>
        </router-link>
      </template>

      <template #end>
        <div class="flex items-center gap-3">
          <Button
            @click="toggleLanguage"
            :label="currentLang === 'EN' ? 'العربية' : 'English'"
            text
            severity="secondary"
          >
            <template #icon>
              <Languages class="w-4 h-4 mr-2" />
            </template>
          </Button>

          <Button
            @click="toggleTheme"
            rounded
            text
            severity="secondary"
          >
            <template #icon>
              <component :is="isDark ? markRaw(Sun) : markRaw(Moon)" class="w-5 h-5" />
            </template>
          </Button>
        </div>
      </template>
    </MenuBar>
  </div>
</template>