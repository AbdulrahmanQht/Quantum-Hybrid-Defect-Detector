<script setup>
import { ref } from 'vue';
import FileUpload from '../components/MyFileUpload.vue';
</script>

<template>
  <div class="min-h-screen p-8 bg-gray-100">
    <h1 class="mb-4 text-3xl font-bold text-green-600">
      Classify
    </h1>
    <FileUpload />
    <div class="mb-4 p-2 bg-white rounded shadow-sm">
      This is the classify page.
    </div>
  </div>
</template>

<style scoped>

</style>