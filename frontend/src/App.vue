<script setup>
import { onMounted, ref } from 'vue';

const msg = ref("");
const api = async () => {
  try {
    const response = await fetch('http://127.0.0.1:8000/');
    const data = await response.json();
    msg.value = data.message;
    console.log(data);
  } catch (error) {
    console.error('Error fetching data:', error);
  }
}
onMounted(api);
</script>

<template>
  <div class="min-h-screen p-8 bg-gray-100">
    <h1 class="mb-4 text-3xl font-bold text-green-600">
      Quantum Defect Detector 5
    </h1>
    
    <button class="px-4 py-2 font-bold text-white bg-green-500 rounded hover:bg-green-700">
      Click Me i am azzam
    </button>
  </div>
</template>

<style scoped>
  h1{
    color: green;
  }
</style>
