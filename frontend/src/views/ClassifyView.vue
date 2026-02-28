<script setup>
import { ref } from 'vue';
import ClassifyImages from '../components/ClassifyImages.vue';
import { useI18n } from 'vue-i18n'

const { t, locale } = useI18n()
</script>

<template>
  <div class="min-h-screen p-8 bg-gray-100">
    <ClassifyImages />
  </div>
</template>

<style scoped>

</style>