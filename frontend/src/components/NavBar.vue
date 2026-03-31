<script setup>
import { ref, computed } from "vue";
import Cookies from 'js-cookie';

const translations = {
  EN: { home: 'Home', classify: 'Classify', benchmark: 'Benchmark', about: 'About', contact: 'Contact' },
  AR: { home: 'الرئيسية', classify: 'تصنيف', benchmark: 'مقارنة', about: 'من نحن', contact: 'اتصل بنا' }
};

const LANG_KEY = 'app_lang';
const THEME_KEY = 'theme';

const currentLang = ref(Cookies.get(LANG_KEY) || 'EN');
const isDark = ref(Cookies.get(THEME_KEY) === 'dark');

const items = computed(() => [
  { label: translations[currentLang.value].home, icon: 'pi pi-home', to: '/' },
  { label: translations[currentLang.value].classify, icon: 'pi pi-search-plus', to: '/classify' },
  { label: translations[currentLang.value].benchmark, icon: 'pi pi-chart-line', to: '/benchmark' },
  { label: translations[currentLang.value].about, icon: 'pi pi-info-circle', to: '/about' },
  { label: translations[currentLang.value].contact, icon: 'pi pi-envelope', to: '/contact' }
]);

const toggleLanguage = () => {
  const nextLang = currentLang.value === 'EN' ? 'AR' : 'EN';
  currentLang.value = nextLang;

  // Save with the new specific key
  Cookies.set(LANG_KEY, nextLang, { expires: 365, path: '/' });

  // Clean up the old 'langen' cookie
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
    <MenuBar :model="items" class="px-6 shadow-md border-none rounded-none">
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