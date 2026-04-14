<script setup>
import { useI18n } from 'vue-i18n'

const { t, locale } = useI18n()
</script>

<template>
 
  <div class="min-h-screen p-8 bg-gray-100">
    <h1 class="mb-4 text-3xl font-bold text-green-600">
      Benchmark View
    </h1>
    <Timeline> </Timeline>
     
    <div class="mb-4 p-2 bg-white rounded shadow-sm  dark:bg-gray-800">
      This is the benchmark page.
      {{ t('benchmark.title') }}
      {{ t('benchmark.subtitle') }}
 
    </div>
  </div>
</template>

<style scoped>

</style>