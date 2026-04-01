<script setup>
import { ref, computed } from "vue";
import Cookies from 'js-cookie';
import { useI18n } from 'vue-i18n';

// Access global i18n instance to get the current locale
const { t, locale } = useI18n();

const LANG_KEY = 'app_lang';
const THEME_KEY = 'theme';

// Reactive across ALL components
const currentLang = computed(() => locale.value);

const isDark = ref(Cookies.get(THEME_KEY) === 'dark');

const items = computed(() => [
  { label: t('navbar.home'), icon: 'pi pi-home', to: '/' },
  { label: t('navbar.classify'), icon: 'pi pi-search-plus', to: '/classify' },
  { label: t('navbar.benchmark'), icon: 'pi pi-chart-line', to: '/benchmark' },
  { label: t('navbar.about'), icon: 'pi pi-info-circle', to: '/about' },
  { label: t('navbar.contact'), icon: 'pi pi-envelope', to: '/contact' }
]);

const toggleLanguage = () => {
  const nextLang = locale.value === 'EN' ? 'AR' : 'EN';

  // 1. Update the global state (This fixes the "all pages" problem)
  locale.value = nextLang;

  // 2. Persist for next visit
  Cookies.set(LANG_KEY, nextLang, { expires: 365, path: '/' });

  // Clean up old keys
  Cookies.remove('langen');
  Cookies.remove('lang');
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
          <span :class="item.icon" class="mr-2" />
          <span class="font-medium">{{ item.label }}</span>
        </router-link>
      </template>

      <template #end>
        <div class="flex items-center gap-3">
          <Button
            @click="toggleLanguage"
            :label="currentLang === 'EN' ? 'العربية' : 'English'"
            icon="pi pi-language"
            text
            severity="secondary"
          />
          <Button
            @click="toggleTheme"
            :icon="isDark ? 'pi pi-sun' : 'pi pi-moon'"
            rounded
            text
            severity="secondary"
          />
        </div>
      </template>
    </MenuBar>
  </div>
</template>