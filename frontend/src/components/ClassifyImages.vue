<script setup>
import { ref, computed, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'

const { t, locale } = useI18n()
const dir = computed(() => locale.value === 'AR' ? 'rtl' : 'ltr')

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref(null)
const loading = ref(false)
const validating = ref(false)
const error = ref(null)
const results = ref(null)
const exportSuccess = ref(null)

const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
const MAX_SIZE_MB = 5
const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

// --- Chart Configuration ---
const chartOptions = computed(() => ({
  plugins: { legend: { display: false } },
  scales: {
    y: {
      position: locale.value === 'AR' ? 'right' : 'left',
      beginAtZero: true,
      grid: { color: 'rgba(128,128,128,0.1)' },
      ticks: { color: '#94a3b8' }
    },
    x: {
      grid: { display: false },
      ticks: { color: '#94a3b8' }
    }
  }
}))

const pieOptions = computed(() => ({
  plugins: {
    legend: {
      position: 'bottom',
      labels: { color: '#94a3b8', padding: 20, font: { size: 12 } }
    }
  }
}))

// --- Magic bytes validation ---
async function validateImageSignature(file) {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onloadend = function (e) {
      const arr = new Uint8Array(e.target.result).subarray(0, 4)
      let header = ''
      for (let i = 0; i < arr.length; i++) header += arr[i].toString(16).padStart(2, '0')
      resolve(header.startsWith('ffd8') || header.startsWith('89504e47') || header.startsWith('52494646'))
    }
    reader.readAsArrayBuffer(file.slice(0, 4))
  })
}

// --- File selection from native input or drop ---
async function handleFile(file) {
  error.value = null
  results.value = null
  exportSuccess.value = null
  if (!file) return

  validating.value = true
  try {
    if (!ALLOWED_TYPES.includes(file.type)) {
      error.value = t('classify.err_format')
      return
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      error.value = t('classify.err_size')
      return
    }
    const isValid = await validateImageSignature(file)
    if (!isValid) {
      error.value = t('classify.err_spoof')
      return
    }
    if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
    selectedFile.value = file
    previewUrl.value = URL.createObjectURL(file)
  } finally {
    validating.value = false
  }
}

function onFileChange(event) {
  handleFile(event.target.files?.[0])
}

function onDrop(event) {
  handleFile(event.dataTransfer?.files?.[0])
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

    const res = await fetch(`${API_BASE}/api/classify`, {
      method: 'POST',
      body: formData
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err?.detail || `Server error: ${res.status}`)
    }

    const data = await res.json()
    results.value = [
      {
        modelName: 'Classical CNN',
        prediction: data.Classical_CNN.predicted_class,
        confidence: data.Classical_CNN.confidence * 100,
        latency: data.Classical_CNN.inference_latency_ms,
        color: '#06b6d4'
      },
      {
        modelName: 'Hybrid QNN',
        prediction: data.Hybrid_QNN.predicted_class,
        confidence: data.Hybrid_QNN.confidence * 100,
        latency: data.Hybrid_QNN.inference_latency_ms,
        color: '#8b5cf6'
      },
      {
        modelName: 'GPU-Hybrid',
        prediction: data.GPU_Hybrid.predicted_class,
        confidence: data.GPU_Hybrid.confidence * 100,
        latency: data.GPU_Hybrid.inference_latency_ms,
        color: '#10b981'
      }
    ]
  } catch (err) {
    error.value = err.message || t('classify.err_upload')
  } finally {
    loading.value = false
  }
}

// --- Computed ---
const topResult = computed(() => {
  if (!results.value) return null
  return results.value.reduce((prev, cur) => prev.confidence > cur.confidence ? prev : cur)
})

const confidenceChartData = computed(() => {
  if (!results.value) return null
  return {
    labels: results.value.map(r => r.modelName),
    datasets: [{
      label: t('classify.confidence') + ' (%)',
      data: results.value.map(r => parseFloat(r.confidence.toFixed(1))),
      backgroundColor: results.value.map(r => r.color),
      borderRadius: 5,
      borderSkipped: false
    }]
  }
})

const latencyChartData = computed(() => {
  if (!results.value) return null
  return {
    labels: results.value.map(r => r.modelName),
    datasets: [{
      label: t('classify.latency') + ' (ms)',
      data: results.value.map(r => parseFloat(r.latency.toFixed(1))),
      backgroundColor: results.value.map(r => r.color),
      borderRadius: 5,
      borderSkipped: false
    }]
  }
})

// --- Reset ---
function reset() {
  selectedFile.value = null
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = null
  results.value = null
  error.value = null
  exportSuccess.value = null
  validating.value = false
  if (fileInput.value) fileInput.value.value = ''
}

// --- Export ---
function exportToCSV() {
  if (!results.value) return
  const headers = ['Model Name', 'Prediction Class', 'Confidence Score (%)', 'Inference Latency (ms)']
  const csvContent = [
    headers.join(','),
    ...results.value.map(r =>
      `"${r.modelName}","${r.prediction}",${r.confidence.toFixed(1)},${r.latency.toFixed(1)}`
    )
  ].join('\n')
  const blob = new Blob([csvContent], { type: 'text/csv' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `defect-detection-${Date.now()}.csv`
  a.click()
  window.URL.revokeObjectURL(url)
  exportSuccess.value = 'Results exported to CSV successfully!'
}

function sanitizeFilename(name) {
  if (!name) return 'unknown'
  return name.replace(/[^a-zA-Z0-9.-]/g, '_')
}

function exportToJSON() {
  if (!results.value) return
  const jsonData = {
    timestamp: new Date().toISOString(),
    fileName: sanitizeFilename(selectedFile.value?.name),
    topPrediction: topResult.value,
    allResults: results.value
  }
  const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `defect-detection-${Date.now()}.json`
  a.click()
  window.URL.revokeObjectURL(url)
  exportSuccess.value = 'Results exported to JSON successfully!'
}

onUnmounted(() => {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
})
</script>

<template>
  <div :dir="dir" class="min-h-screen bg-slate-50 dark:bg-slate-950 transition-colors duration-300">
    <div class="max-w-4xl mx-auto px-4 py-10 sm:px-6">

      <!-- Page Header -->
      <div class="text-center mb-10">
        <span class="inline-block text-xs font-mono tracking-widest text-cyan-600 dark:text-cyan-400 border border-cyan-200 dark:border-cyan-800 bg-cyan-50 dark:bg-cyan-950 px-3 py-1 rounded-full mb-4">
          QUANTUM · CLASSICAL · HYBRID
        </span>
        <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-50 tracking-tight mb-2">
          {{ t('classify.title') }}
        </h1>
        <p class="text-slate-500 dark:text-slate-400 text-sm">
          {{ t('classify.subtitle') }}
        </p>
      </div>

      <!-- Upload Card -->
      <Card class="mb-4 shadow-sm">
        <template #content>

          <!-- Native file input — hidden, triggered by drop zone or button -->
          <input
            ref="fileInput"
            type="file"
            accept=".jpg,.jpeg,.png,.webp"
            class="hidden"
            @change="onFileChange"
          />

          <!-- Drop zone (no file selected yet) -->
          <div
            v-if="!selectedFile"
            class="border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-xl p-12 text-center cursor-pointer hover:border-cyan-400 dark:hover:border-cyan-500 hover:bg-cyan-50 dark:hover:bg-cyan-950/30 transition-all duration-200"
            @click="fileInput.click()"
            @dragover.prevent
            @drop.prevent="onDrop"
          >
            <i class="pi pi-cloud-upload text-5xl text-slate-300 dark:text-slate-600 mb-4 block" />
            <p class="text-slate-600 dark:text-slate-300 font-medium mb-1">
              {{ t('classify.dropzone') }}
              <span class="text-cyan-600 dark:text-cyan-400 underline underline-offset-2">{{ t('classify.choose') }}</span>
            </p>
            <p class="text-xs font-mono text-slate-400 dark:text-slate-500 mt-2">
              PNG · JPG · WEBP &nbsp;·&nbsp; {{ t('classify.max_size') }} &nbsp;·&nbsp;
            </p>
          </div>

          <!-- Preview (file selected) -->
          <div v-else class="space-y-4">

            <!-- PrimeVue Image with built-in zoom/preview -->
            <div class="relative overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-900 flex justify-center">
              <Image
                :src="previewUrl"
                :alt="selectedFile.name"
                imageClass="max-h-72 object-contain"
                preview
              />
              <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent px-4 py-3 pointer-events-none">
                <p class="text-white text-xs font-mono truncate">{{ selectedFile.name }}</p>
              </div>
            </div>

            <!-- Action buttons -->
            <div class="flex gap-3 justify-end pt-1">
              <Button
                :label= "t('classify.reset')"
                icon="pi pi-refresh"
                severity="secondary"
                outlined
                :disabled="loading || validating"
                @click="reset"
              />
              <Button
                :label="validating ? t('classify.validating') : loading ? t('classify.running') : t('classify.run')"
                :icon="loading || validating ? 'pi pi-spin pi-spinner' : 'pi pi-play'"
                :disabled="loading || validating"
                @click="uploadImage"
              />
            </div>
          </div>

        </template>
      </Card>

      <!-- Choose File button shown before file is picked -->
      <div v-if="!selectedFile" class="flex justify-center mb-6">
        <Button
          :label="t('classify.choose')"
          icon="pi pi-folder-open"
          severity="contrast"
          :loading="validating"
          @click="fileInput.click()"
        />
      </div>

      <!-- Error -->
      <Message v-if="error" severity="error" :closable="true" class="mb-4" @close="error = null">
        {{ error }}
      </Message>

      <!-- Results -->
      <div v-if="results && topResult" class="space-y-4 animate-fadein">

        <!-- Top Prediction Card -->
        <Card
          class="shadow-sm border-l-4 transition-colors duration-300"
          :class="topResult.prediction === 'No Defect'
            ? 'border-l-emerald-500 dark:border-l-emerald-400'
            : 'border-l-red-500 dark:border-l-red-400'"
        >
          <template #content>
            <div class="flex items-center gap-4">
              <i
                class="text-4xl flex-shrink-0"
                :class="topResult.prediction === 'No Defect'
                  ? 'pi pi-check-circle text-emerald-500 dark:text-emerald-400'
                  : 'pi pi-exclamation-triangle text-red-500 dark:text-red-400'"
              />
              <div class="flex-1 min-w-0">
                <p class="text-xs font-mono tracking-widest text-slate-400 dark:text-slate-500 mb-1 uppercase">
                  {{ t('classify.top_prediction') }}
                </p>
                <p class="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">
                  {{ t('classify.' + topResult.prediction) }}
                </p>
                <p class="text-xs font-mono text-slate-400 dark:text-slate-500">
                  {{ topResult.modelName }} &nbsp;·&nbsp;
                  {{ topResult.confidence.toFixed(1) }}% {{ t('classify.confidence').toLowerCase() }} &nbsp;·&nbsp;
                  {{ topResult.latency.toFixed(1) }}ms
                </p>
              </div>
              <Tag
                :value="topResult.prediction === 'No Defect' ? t('classify.safe') : t('classify.defect')"
                :severity="topResult.prediction === 'No Defect' ? 'success' : 'danger'"
              />
            </div>
          </template>
        </Card>

        <!-- Model Comparison Table -->
        <Card class="shadow-sm">
          <template #title>
            <span class="text-xs font-mono tracking-widest text-slate-400 dark:text-slate-500 uppercase">
              {{ t('classify.model_comparison') }}
            </span>
          </template>
          <template #content>
            <DataTable :value="results" stripedRows responsiveLayout="scroll">

              <Column field="modelName" :header="t('classify.model')">
                <template #body="{ data }">
                  <div class="flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" :style="{ background: data.color }" />
                    <span class="font-medium text-slate-800 dark:text-slate-200">{{ data.modelName }}</span>
                  </div>
                </template>
              </Column>

              <Column field="prediction" :header="t('classify.prediction')">
                <template #body="{ data }">
                  <Tag
                    :value="t('classify.' + data.prediction)"
                    :severity="data.prediction === 'No Defect' ? 'success' : 'danger'"
                  />
                </template>
              </Column>

              <Column field="confidence" :header="t('classify.confidence')">
                <template #body="{ data }">
                  <div class="flex items-center gap-3 min-w-40">
                    <!-- PrimeVue ProgressBar with per-model color via passthrough -->
                    <ProgressBar
                      :value="parseFloat(data.confidence.toFixed(1))"
                      :showValue="false"
                      class="flex-1"
                      :pt="{
                        root: { style: 'height: 6px;' },
                        value: { style: `background: ${data.color};` }
                      }"
                    />
                    <span class="text-xs font-mono text-slate-500 dark:text-slate-400 w-12 text-right shrink-0">
                      {{ data.confidence.toFixed(1) }}%
                    </span>
                  </div>
                </template>
              </Column>

              <Column field="latency" :header="t('classify.latency')">
                <template #body="{ data }">
                  <span class="text-xs font-mono text-slate-500 dark:text-slate-400">
                    {{ data.latency.toFixed(1) }} ms
                  </span>
                </template>
              </Column>

            </DataTable>
          </template>
        </Card>

        <!-- Charts -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card class="shadow-sm">
            <template #title>
              <span class="text-xs font-mono tracking-widest text-slate-400 dark:text-slate-500 uppercase">
                {{ t('classify.confidence_scores') }}
              </span>
            </template>
            <template #content>
              <Chart type="bar" :data="confidenceChartData" :options="chartOptions" class="h-52" />
            </template>
          </Card>

          <Card class="shadow-sm">
            <template #title>
              <span class="text-xs font-mono tracking-widest text-slate-400 dark:text-slate-500 uppercase">
                {{ t('classify.inference_latency') }}
              </span>
            </template>
            <template #content>
              <Chart type="bar" :data="latencyChartData" :options="chartOptions" class="h-52" />
            </template>
          </Card>

          <Card class="shadow-sm lg:col-span-2">
            <template #title>
              <span class="text-xs font-mono tracking-widest text-slate-400 dark:text-slate-500 uppercase">
                {{ t('classify.distribution') }}
              </span>
            </template>
            <template #content>
              <div class="flex justify-center">
                <Chart type="pie" :data="confidenceChartData" :options="pieOptions" class="h-60 max-w-sm w-full" />
              </div>
            </template>
          </Card>
        </div>

        <!-- Export -->
        <Card class="shadow-sm">
          <template #content>
            <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <p class="font-semibold text-slate-800 dark:text-slate-100 mb-0.5">{{ t('classify.export_title') }}</p>
                <p class="text-sm text-slate-500 dark:text-slate-400">
                  {{ t('classify.export_subtitle') }}
                </p>
              </div>
              <div class="flex gap-2 flex-shrink-0">
                <Button
                  label="CSV"
                  icon="pi pi-file-excel"
                  severity="success"
                  outlined
                  size="small"
                  @click="exportToCSV"
                />
                <Button
                  label="JSON"
                  icon="pi pi-file"
                  severity="info"
                  outlined
                  size="small"
                  @click="exportToJSON"
                />
              </div>
            </div>
          </template>
        </Card>

        <!-- Export success -->
        <Message v-if="exportSuccess" severity="success" :closable="true" @close="exportSuccess = null">
          {{ exportSuccess }}
        </Message>

      </div>
    </div>
  </div>
</template>

<style scoped>
.animate-fadein {
  animation: fadein 0.35s ease both;
}

@keyframes fadein {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>
