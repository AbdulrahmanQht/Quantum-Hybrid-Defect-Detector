<script setup>
import { ref, computed } from 'vue'

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref(null)
const loading = ref(false)
const error = ref(null)
const results = ref(null)
const exportSuccess = ref(null)

const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png']
const MAX_SIZE_MB = 5

// --- Chart Configuration ---
const chartOptions = {
  plugins: { legend: { display: false } },
  scales: { y: { beginAtZero: true } }
}

const pieOptions = {
  plugins: { legend: { position: 'bottom' } }
}

function onFileChange(event) {
  error.value = null
  results.value = null
  exportSuccess.value = null

  const file = event.target.files?.[0]
  if (!file) return

  if (!ALLOWED_TYPES.includes(file.type)) {
    error.value = 'Invalid file format! Please upload a PNG, JPG, or JPEG image.'
    return
  }

  if (file.size > MAX_SIZE_MB * 1024 * 1024) {
    error.value = `File size exceeds ${MAX_SIZE_MB}MB! Please upload a smaller image.`
    return
  }

  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
}

async function uploadImage() {
  if (!selectedFile.value) return

  loading.value = true
  error.value = null
  results.value = null
  exportSuccess.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const res = await fetch('http://127.0.0.1:8000/api/classify', {
      method: 'POST',
      body: formData
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err?.detail || `Server error: ${res.status}`)
    }

    const data = await res.json()

    // Map the backend JSON to our frontend array structure
    results.value = [
      {
        modelName: 'Classical CNN',
        prediction: data.Classical_CNN.predicted_class,
        confidence: data.Classical_CNN.confidence * 100,
        latency: data.Classical_CNN.inference_latency_ms
      },
      {
        modelName: 'Hybrid QNN',
        prediction: data.Hybrid_QNN.predicted_class,
        confidence: data.Hybrid_QNN.confidence * 100,
        latency: data.Hybrid_QNN.inference_latency_ms
      },
      {
        modelName: 'GPU-Hybrid',
        prediction: data.GPU_Hybrid.predicted_class,
        confidence: data.GPU_Hybrid.confidence * 100,
        latency: data.GPU_Hybrid.inference_latency_ms
      }
    ]

  } catch (err) {
    error.value = err.message || 'Upload failed.'
  } finally {
    loading.value = false
  }
}

// --- Computed Properties for Dashboard ---
const topResult = computed(() => {
  if (!results.value) return null
  return results.value.reduce((prev, current) =>
    (prev.confidence > current.confidence) ? prev : current
  )
})

const confidenceChartData = computed(() => {
  if (!results.value) return null
  return {
    labels: results.value.map(r => r.modelName),
    datasets: [{
      label: 'Confidence Score (%)',
      data: results.value.map(r => r.confidence.toFixed(1)),
      backgroundColor: ['#14B8A6', '#22C55E', '#0D9488'],
      borderRadius: 6
    }]
  }
})

const latencyChartData = computed(() => {
  if (!results.value) return null
  return {
    labels: results.value.map(r => r.modelName),
    datasets: [{
      label: 'Inference Latency (ms)',
      data: results.value.map(r => r.latency.toFixed(1)),
      backgroundColor: ['#14B8A6', '#22C55E', '#0D9488'],
      borderRadius: 6
    }]
  }
})

// --- Export Functions ---
function exportToCSV() {
  if (!results.value) return
  const headers = ['Model Name', 'Prediction Class', 'Confidence Score (%)', 'Inference Latency (ms)']
  const csvContent = [
    headers.join(','),
    ...results.value.map(r => `${r.modelName},${r.prediction},${r.confidence.toFixed(1)},${r.latency.toFixed(1)}`)
  ].join('\n')

  const blob = new Blob([csvContent], { type: 'text/csv' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `defect-detection-results-${Date.now()}.csv`
  a.click()
  window.URL.revokeObjectURL(url)
  exportSuccess.value = 'Results exported to CSV successfully!'
}

function exportToJSON() {
  if (!results.value) return
  const jsonData = {
    timestamp: new Date().toISOString(),
    fileName: selectedFile.value?.name,
    topPrediction: topResult.value,
    allResults: results.value
  }
  const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `defect-detection-results-${Date.now()}.json`
  a.click()
  window.URL.revokeObjectURL(url)
  exportSuccess.value = 'Results exported to JSON successfully!'
}
</script>

<template>
  <div class="max-w-6xl mx-auto px-4 py-8 sm:px-6 lg:px-8">

    <div class="text-center mb-12">
      <h1 class="text-3xl font-bold text-slate-900 mb-3">Hybrid Quantum-Classical Defect Detector</h1>
      <p class="text-slate-600">Upload an image to compare model performance</p>
    </div>

    <Card class="mb-8 border-2 border-dashed border-slate-300 hover:border-teal-500 transition-colors shadow-none text-center">
      <template #content>
        <div class="p-8">
          <input
            ref="fileInput"
            type="file"
            accept=".jpg,.jpeg,.png"
            class="hidden"
            @change="onFileChange"
          />
          <i class="pi pi-cloud-upload text-5xl text-slate-400 mb-4 block"></i>

          <Button
            label="Choose File"
            icon="pi pi-folder-open"
            severity="success"
            class="mb-3"
            @click="fileInput.click()"
          />
          <p class="text-slate-500 text-sm">Max File Size is 5MB</p>
          <p class="text-slate-500 text-sm">Accepted file types: .jpg, .jpeg, .png</p>
          <p v-if="selectedFile" class="text-slate-800 font-semibold mt-3">
            Selected: {{ selectedFile.name }}
          </p>
        </div>
      </template>
    </Card>

    <Message v-if="error" severity="error" @close="error = null">{{ error }}</Message>

    <Card v-if="previewUrl" class="mb-8 text-center shadow-sm border border-slate-200">
      <template #title>Image Preview</template>
      <template #content>
        <img :src="previewUrl" alt="Preview" class="max-h-64 rounded-lg border border-slate-300 mx-auto mb-6" />
        <Button
          label="Run Classification Models"
          icon="pi pi-play"
          size="large"
          :loading="loading"
          @click="uploadImage"
        />
      </template>
    </Card>

    <div v-if="results && topResult" class="space-y-8 animate-fadein">

      <Card class="bg-gradient-to-br from-teal-50 to-blue-50 border border-teal-200 shadow-sm">
        <template #content>
          <div class="flex items-center gap-3 mb-4">
            <i :class="topResult.prediction === 'No Defect' ? 'pi pi-check-circle text-green-500' : 'pi pi-exclamation-triangle text-red-500'" class="text-3xl"></i>
            <h2 class="text-2xl font-bold text-slate-800">Top Prediction</h2>
          </div>
          <div class="text-lg mb-2">
            <span class="text-slate-600">Defect Type: </span>
            <Tag :severity="topResult.prediction === 'No Defect' ? 'success' : 'danger'" :value="topResult.prediction" class="text-base" />
          </div>
          <div class="text-lg">
            <span class="text-slate-600">Highest Confidence: </span>
            <span class="font-bold text-slate-900">{{ topResult.confidence.toFixed(1) }}%</span>
          </div>
        </template>
      </Card>

      <Card class="shadow-sm border border-slate-200">
        <template #title>Model Comparison</template>
        <template #content>
          <DataTable :value="results" responsiveLayout="scroll" stripedRows class="p-datatable-sm">
            <Column field="modelName" header="Model Name" class="font-medium"></Column>
            <Column field="prediction" header="Prediction Class">
              <template #body="slotProps">
                <Tag
                  :severity="slotProps.data.prediction === 'No Defect' ? 'success' : 'danger'"
                  :value="slotProps.data.prediction"
                />
              </template>
            </Column>
            <Column field="confidence" header="Confidence Score (%)">
              <template #body="slotProps">
                {{ slotProps.data.confidence.toFixed(1) }}%
              </template>
            </Column>
            <Column field="latency" header="Inference Latency (ms)">
              <template #body="slotProps">
                {{ slotProps.data.latency.toFixed(1) }} ms
              </template>
            </Column>
          </DataTable>
        </template>
      </Card>

      <Card class="shadow-sm border border-slate-200">
        <template #title>
          <div class="flex items-center gap-2">
            <i class="pi pi-chart-bar text-teal-600"></i>
            <span>Performance Comparison Charts</span>
          </div>
        </template>
        <template #content>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div>
              <h4 class="text-center font-semibold text-slate-700 mb-4">Confidence Score Comparison</h4>
              <Chart type="bar" :data="confidenceChartData" :options="chartOptions" class="h-64" />
            </div>
            <div>
              <h4 class="text-center font-semibold text-slate-700 mb-4">Inference Latency Comparison</h4>
              <Chart type="bar" :data="latencyChartData" :options="chartOptions" class="h-64" />
            </div>
          </div>
          <div class="w-full md:w-1/2 mx-auto">
            <h4 class="text-center font-semibold text-slate-700 mb-4">Confidence Distribution</h4>
            <Chart type="pie" :data="confidenceChartData" :options="pieOptions" class="h-64" />
          </div>
        </template>
      </Card>

      <Card class="shadow-sm border border-slate-200">
        <template #content>
          <div class="flex flex-col md:flex-row justify-between items-center gap-4">
            <div>
              <h3 class="font-bold text-slate-800 text-lg">Export Classification Results</h3>
              <p class="text-slate-600 text-sm">Download your results as CSV or JSON format for further analysis</p>
            </div>
            <div class="flex gap-3">
              <Button label="Export CSV" icon="pi pi-file-excel" severity="success" outlined @click="exportToCSV" />
              <Button label="Export JSON" icon="pi pi-file" severity="info" outlined @click="exportToJSON" />
            </div>
          </div>
        </template>
      </Card>

      <Message v-if="exportSuccess" severity="success" @close="exportSuccess = null">{{ exportSuccess }}</Message>

    </div>
  </div>
</template>