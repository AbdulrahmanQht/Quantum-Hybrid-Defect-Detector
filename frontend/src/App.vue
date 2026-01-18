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
  <div>
    <h1>
      {{ msg }}
    </h1>
  </div>
</template>

<style scoped>
  h1{
    color: green;
  }
</style>
