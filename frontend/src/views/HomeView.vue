<script setup>
import {onMounted, ref} from "vue";

const msg = ref("");

const api = async () => {
  try {
    const response = await fetch('http://127.0.0.1:8000/api/health');
    const data = await response.json();
    msg.value = data.message;
  } catch (error) {
    console.error('Error fetching data:', error);
  }
}

onMounted(api);
</script>

<template>
  <div class="min-h-screen p-8 bg-gray-100">
    <h1 class="mb-4 text-3xl font-bold text-green-600">
      Home Page
    </h1>

    <div v-if="msg" class="mb-4 p-2 bg-white rounded shadow-sm">
       Backend Status: {{ msg }}
    </div>


  </div>
</template>

<style scoped>

</style>