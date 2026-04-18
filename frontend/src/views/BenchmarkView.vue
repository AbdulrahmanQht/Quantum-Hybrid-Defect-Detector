<script setup>
import { ref, onMounted } from 'vue';

const benchmarkData = ref(null);
const isLoading = ref(false);
const error = ref(null);

const fetchBenchmarkData = async () => {
  isLoading.value = true;
  error.value = null;
  try {
    // Relative path assumes your Vite proxy is configured to route /api to FastAPI
    const response = await fetch('/api/v1/benchmark');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    benchmarkData.value = await response.json();
  } catch (err) {
    console.error("Failed to fetch benchmark results:", err);
    error.value = "Failed to load benchmark data. Ensure the backend is running and benchmark_results.json exists.";
  } finally {
    isLoading.value = false;
  }
};

onMounted(() => {
  fetchBenchmarkData();
});
</script>

<template>
  <div class="min-h-screen p-8 bg-gray-100">
    <h1 class="mb-4 text-3xl font-bold text-green-600">
      Benchmark View
    </h1>

    <div v-if="isLoading" class="p-4 bg-blue-100 text-blue-700 rounded mb-4">
      Fetching benchmark results...
    </div>

    <div v-if="error" class="p-4 bg-red-100 text-red-700 rounded mb-4">
      {{ error }}
    </div>

    <div v-if="benchmarkData" class="bg-white p-6 rounded shadow-lg">
      <h2 class="text-xl font-semibold mb-2 text-gray-800">Raw JSON Output:</h2>
      <pre class="overflow-x-auto p-4 bg-gray-900 text-green-400 rounded text-sm">{{ JSON.stringify(benchmarkData, null, 2) }}</pre>
    </div>
    
    <div v-else-if="!isLoading && !error" class="text-gray-500">
      No data available.
    </div>
  </div>
</template>

<style scoped>
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>