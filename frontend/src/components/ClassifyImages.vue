<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'

const { t, locale } = useI18n()

const STORAGE_KEY = 'classifyState'
const isRestored = ref(false)
const storedDataUrl = ref(null)
const appliedNoiseLevel = ref(null)

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref(null)
const loading = ref(false)
const validating = ref(false)
const error = ref(null)
const results = ref(null)
const exportSuccess = ref(null)
const compareWithNoise = ref(false)
const noiseLevel = ref(0.3)
const noisyResults = ref(null)

const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
const MAX_SIZE_MB = 5

// --- Noise Severity ---
const noiseSeverityLabel = computed(() => {
  // Always use the raw noiseLevel here so the logic remains consistent
  if (noiseLevel.value <= 0.3) return t('classify.noise_low')
  if (noiseLevel.value <= 0.6) return t('classify.noise_medium')
  return t('classify.noise_high')
})

const sliderNoiseLevel = computed({
  get() {
    return noiseLevel.value
  },
  set(value) {
    noiseLevel.value = value
  }
})

const noisyTopResult = computed(() => {
  if (!noisyResults.value) return null
  return noisyResults.value.reduce((prev, cur) => prev.confidence > cur.confidence ? prev : cur)
})

// --- Chart Configuration ---
const chartOptions = computed(() => ({
  plugins: { legend: { display: false } },
  scales: {
    y: {
      position: 'left',
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

    // Keeps uploaded image so results don't go away with refresh
    selectedFile.value = file
    previewUrl.value = URL.createObjectURL(file)

    // Read as data URL so we can persist across refreshes
    const reader = new FileReader()
    reader.onload = (e) => {
      storedDataUrl.value = e.target.result
      persistState()
    }
    reader.readAsDataURL(file)
  } finally {
    validating.value = false
  }
}

function persistState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      fileName:       selectedFile.value?.name,
      previewDataUrl: storedDataUrl.value,
      results:        results.value,
      noisyResults:   noisyResults.value,
      compareWithNoise: compareWithNoise.value,
      noiseLevel: noiseLevel.value,
      appliedNoiseLevel: appliedNoiseLevel.value,
    }))
  } catch {
    // Quota exceeded (large image) — fail silently
  }
}

function onFileChange(event) {
  handleFile(event.target.files?.[0])
}

function onDrop(event) {
  handleFile(event.dataTransfer?.files?.[0])
}

async function uploadImage() {
  if (!selectedFile.value || selectedFile.value.restored) return

  if (!selectedFile.value) return
  loading.value = true
  error.value = null
  results.value = null
  noisyResults.value = null
  exportSuccess.value = null

  try {
    const snapshotNoiseLevel = compareWithNoise.value ? noiseLevel.value : null
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    formData.append('compare_with_noise', compareWithNoise.value)
    if (compareWithNoise.value) {
      formData.append('noise_level', noiseLevel.value.toFixed(2))
    }

    const res = await fetch("/api/v1/classify", {
      method: 'POST',
      body: formData
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err?.detail || `Server error: ${res.status}`)
    }

    const data = await res.json()

const modelMeta = computed(() => ({
  CNN: {
    key: 'CNN',
    label: t('classify.models.cnn'),
    color: '#2563eb' // blue (clear, strong anchor)
  },
  QNN_CPU: {
    key: 'QNN_CPU',
    label: t('classify.models.qnn_cpu'),
    color: '#0d9488' // teal (distinct from blue)
  },
  QNN_GPU: {
    key: 'QNN_GPU',
    label: t('classify.models.qnn_gpu'),
    color: '#7c3aed' // green (separate from teal)
  }
}))

    const parseSet = (set) => [
    {
      modelKey: modelMeta.value.CNN.key,
      modelName: modelMeta.value.CNN.label,
      prediction: set.CNN.predicted_class,
      confidence: set.CNN.confidence * 100,
      latency: set.CNN.inference_latency_ms,
      color: modelMeta.value.CNN.color
    },
    {
      modelKey: modelMeta.value.QNN_CPU.key,
      modelName: modelMeta.value.QNN_CPU.label,
      prediction: set.QNN_CPU.predicted_class,
      confidence: set.QNN_CPU.confidence * 100,
      latency: set.QNN_CPU.inference_latency_ms,
      color: modelMeta.value.QNN_CPU.color
    },
    ...(set.QNN_GPU ? [{
      modelKey: modelMeta.value.QNN_GPU.key,
      modelName: modelMeta.value.QNN_GPU.label,
      prediction: set.QNN_GPU.predicted_class,
      confidence: set.QNN_GPU.confidence * 100,
      latency: set.QNN_GPU.inference_latency_ms,
      color: modelMeta.value.QNN_GPU.color
    }] : [])
  ]


    results.value = parseSet(data.clean)
    appliedNoiseLevel.value = snapshotNoiseLevel
    if (data.noisy) noisyResults.value = parseSet(data.noisy)

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
      label: 'Confidence (%)',
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
      label: 'Latency (ms)',
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
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)  // was revokeObjectObject — typo
  previewUrl.value = null
  results.value = null
  noisyResults.value = null        // ← charts/table won't clear without this
  error.value = null
  exportSuccess.value = null
  validating.value = false
  compareWithNoise.value = false   // ← reset toggle back to off
  noiseLevel.value = 0.3           // ← reset slider to default
  if (fileInput.value) fileInput.value.value = ''
  localStorage.removeItem(STORAGE_KEY)
  isRestored.value    = false
  storedDataUrl.value = null
  appliedNoiseLevel.value = null
}

// --- Export ---
function exportToCSV() {
  if (!results.value) return

  const headers = ['Condition', 'Model Name', 'Prediction Class', 'Confidence Score (%)', 'Inference Latency (ms)']
  const rows = [
    headers.join(','),
    ...results.value.map(r =>
      `"Clean","${r.modelName}","${r.prediction}",${r.confidence.toFixed(1)},${r.latency.toFixed(1)}`
    ),
    ...(noisyResults.value
      ? noisyResults.value.map(r =>
          `"Noisy (level ${appliedNoiseLevel.value?.toFixed(2)})","${r.modelName}","${r.prediction}",${r.confidence.toFixed(1)},${r.latency.toFixed(1)}`
        )
      : [])
  ]

  const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `defect-detection-${Date.now()}.csv`
  a.click()
  window.URL.revokeObjectURL(url)
  exportSuccess.value = t('classify.export_csv_success')
}

function sanitizeFilename(name) {
  if (!name) return 'unknown'
  return name.replace(/[^a-zA-Z0-9.-]/g, '_')
}

function stripColor(results) {
  return results.map(({ color, ...rest }) => rest)
}

function exportToJSON() {
  if (!results.value) return

  const jsonData = {
    timestamp: new Date().toISOString(),
    fileName: sanitizeFilename(selectedFile.value?.name),
    clean: {
    topPrediction: stripColor([topResult.value])[0],
    allResults: stripColor(results.value)
  },
  ...(noisyResults.value && {
    noisy: {
      noiseLevel: appliedNoiseLevel.value,
      topPrediction: stripColor([noisyTopResult.value])[0],
      allResults: stripColor(noisyResults.value)
    }
  })
  }

  const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `defect-detection-${Date.now()}.json`
  a.click()
  window.URL.revokeObjectURL(url)
  exportSuccess.value = t('classify.export_json_success')
}

onMounted(() => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return
    const saved = JSON.parse(raw)

    // Results first — before any watched refs are assigned
    if (saved.results)      results.value      = saved.results
    if (saved.noisyResults) noisyResults.value = saved.noisyResults

    // Locked-in noise level for the restored results
    if (saved.appliedNoiseLevel != null) appliedNoiseLevel.value = saved.appliedNoiseLevel

    // Watched slider refs last — watchers may fire here, results are already in place
    compareWithNoise.value = saved.compareWithNoise ?? false
    noiseLevel.value       = saved.noiseLevel       ?? 0.3

    // Preview / file sentinel
    if (saved.previewDataUrl) {
      previewUrl.value    = saved.previewDataUrl
      storedDataUrl.value = saved.previewDataUrl
      selectedFile.value  = { name: saved.fileName ?? 'image', restored: true }
      isRestored.value    = true
    }
  } catch {
    // Ignore corrupted storage
  }
})

watch(results,      persistState, { deep: true })
watch(noisyResults, persistState, { deep: true })
watch([compareWithNoise, noiseLevel], persistState)

onUnmounted(() => {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
})
</script>

<template>
  <div class="classify-shell" :class="{ 'classify-shell--ar': locale === 'AR' }">
    <div class="classify-header">
      <span class="classify-eyebrow">
        {{ t('classify.eyebrow') }}
      </span>
      <h1 class="classify-title">{{ t('classify.title') }}</h1>
      <p class="classify-subtitle">
        {{ t('classify.subtitle') }}
      </p>
    </div>

    <Card class="classify-card q-glass">
      <template #content>
        <input
          ref="fileInput"
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          class="hidden"
          @change="onFileChange"
        />

        <div
          v-if="!selectedFile"
          class="upload-dropzone"
          @click="fileInput.click()"
          @dragover.prevent
          @drop.prevent="onDrop"
        >
          <div class="upload-dropzone__icon-wrap">
            <i class="pi pi-cloud-upload upload-dropzone__icon" />
          </div>
          <p class="upload-dropzone__title">
            {{ t('classify.dropzone') }}
          </p>
          <p class="upload-dropzone__meta">
            <span class="upload-dropzone__link">{{ t('classify.choose') }}</span>
            <span> · </span>
            <span>{{ t('classify.max_size') }}</span>
          </p>
        </div>

        <div v-else dir="ltr" class="classify-preview-stack">
          <Transition name="slide-down">
              <div v-if="isRestored" class="restored-banner">
                <i class="pi pi-history" />
                {{ t('classify.restored_hint') }}
              </div>
            </Transition>
          <div class="preview-frame">
            <Image
              :src="previewUrl"
              :alt="selectedFile.name"
              imageClass="max-h-72 object-contain"
              preview
            />
            <div class="preview-overlay">
              <p class="preview-filename">{{ selectedFile.name }}</p>
            </div>
            
          </div>

          <div class="noise-panel" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
  <div class="noise-panel__header">
    <div>
      <p class="noise-panel__title">{{ t('classify.noise_toggle') }}</p>
      <p class="noise-panel__subtitle">{{ t('classify.noise_toggle_sub') }}</p>
    </div>
    <ToggleSwitch v-model="compareWithNoise" />
  </div>

  <Transition name="slide-down">
    <div v-if="compareWithNoise" class="noise-panel__body">
      <div class="noise-panel__meta">
        <span class="noise-panel__label">{{ t('classify.noise_severity') }}</span>
        <span
          class="noise-panel__badge"
          :class="{
            'noise-panel__badge--mild': noiseLevel <= 0.3,
            'noise-panel__badge--degraded': noiseLevel > 0.3 && noiseLevel <= 0.6,
            'noise-panel__badge--severe': noiseLevel > 0.6
          }"
        >
          {{ noiseSeverityLabel }} · {{ noiseLevel.toFixed(2) }}
        </span>
      </div>

      <div class="noise-slider-wrap" dir="ltr">
        <Slider
          v-model="sliderNoiseLevel"
          :min="0.05"
          :max="1.0"
          :step="0.05"
          class="w-full noise-slider"
        />
      </div>

      <div class="noise-panel__scale" dir="ltr">
        <span>{{ t('classify.noise_low') }}</span>
        <span>{{ t('classify.noise_medium') }}</span>
        <span>{{ t('classify.noise_high') }}</span>
      </div>
        </div>
      </Transition>
    </div>

          <div class="classify-actions">
            <Button
              :label="t('classify.reset')"
              icon="pi pi-refresh"
              severity="secondary"
              outlined
              :disabled="loading || validating"
              @click="reset"
            />
            <Button
              :label="validating ? t('classify.validating') : loading ? t('classify.running') : t('classify.run')"
              :icon="loading || validating ? 'pi pi-spin pi-spinner' : 'pi pi-play'"
              :disabled="loading || validating || isRestored"
              class="classify-run-btn"
              @click="uploadImage"
            />
          </div>
        </div>
      </template>
    </Card>

    <div v-if="!selectedFile" class="classify-choose-row">
      <Button
        :label="t('classify.choose')"
        icon="pi pi-folder-open"
        severity="contrast"
        :loading="validating"
        @click="fileInput.click()"
      />
    </div>

    <Message v-if="error" severity="error" :closable="true" class="mb-4" @close="error = null">
      {{ error }}
    </Message>

    <div v-if="results && topResult" class="results-stack animate-fadein">

        <!-- Top Prediction Card -->
        <Card class="q-glass result-card transition-colors duration-300 border-l-4 border-l-[var(--q-teal)]">

          <template #content>
            <div class="flex items-center gap-4">
              <i class="pi pi-chart-bar flex-shrink-0 text-4xl text-[var(--q-teal)]" />
              <div class="flex-1 min-w-0">
                <p class="mb-1 font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500">
                  {{ t('classify.top_prediction') }}
                </p>
                <p class="mb-1 text-xl font-bold text-[var(--q-text)]">
                  {{ t('classify.' + topResult.prediction) }}
                </p>
                <p class="font-mono text-xs text-[var(--q-muted)]">
                  {{ topResult.modelName }} &nbsp;·&nbsp;
                  {{ topResult.confidence.toFixed(1) }}% {{ t('classify.confidence').toLowerCase() }} &nbsp;·&nbsp;
                  {{ topResult.latency.toFixed(1) }}ms
                </p>
              </div>
            </div>
          </template>
        </Card>

        <!-- Model Comparison Table -->
        <Card class="q-glass result-card">
          <template #title>
            <span class="font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500">
              {{ t('classify.model_comparison') }}
            </span>
          </template>
          <template #content>
            <DataTable :value="results"  responsiveLayout="scroll">

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
                    severity="danger"
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
                    <span class="w-12 font-mono text-xs text-right text-slate-500 dark:text-slate-400 shrink-0">
                      {{ data.confidence.toFixed(1) }}%
                    </span>
                  </div>
                </template>
              </Column>

              <Column field="latency" :header="t('classify.latency')">
                <template #body="{ data }">
                  <span class="font-mono text-xs text-slate-500 dark:text-slate-400">
                    {{ data.latency.toFixed(1) }} ms
                  </span>
                </template>
              </Column>

            </DataTable>
          </template>
        </Card>

        <!-- Charts -->
        <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card class="q-glass result-card">
            <template #title>
              <span
                class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                :dir="locale === 'AR' ? 'rtl' : 'ltr'"
                :class="{ 'chart-card-title--ar': locale === 'AR' }"
              >
                {{ t('classify.confidence_scores') }}
              </span>
            </template>
            <template #content>
              <div class="chart-wrap" dir="ltr">
                <Chart type="bar" :data="confidenceChartData" :options="chartOptions" class="h-52" />
              </div>
            </template>
          </Card>

          <Card class="q-glass result-card">
            <template #title>
              <span
                class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                :dir="locale === 'AR' ? 'rtl' : 'ltr'"
                :class="{ 'chart-card-title--ar': locale === 'AR' }"
              >
                {{ t('classify.inference_latency') }}
              </span>
            </template>
            <template #content>
              <div class="chart-wrap" dir="ltr">
                <Chart type="bar" :data="latencyChartData" :options="chartOptions" class="h-52" />
              </div>
            </template>
          </Card>

          <Card class="q-glass result-card lg:col-span-2">
            <template #title>
              <span
                class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                :dir="locale === 'AR' ? 'rtl' : 'ltr'"
                :class="{ 'chart-card-title--ar': locale === 'AR' }"
              >
                {{ t('classify.distribution') }}
              </span>
            </template>
            <template #content>
              <div class="flex justify-center">
                <div class="chart-wrap chart-wrap--center" dir="ltr">
                  <Chart type="pie" :data="confidenceChartData" :options="pieOptions" class="w-full max-w-sm h-60" />
                </div>
              </div>
            </template>
          </Card>
        </div>

        <!-- Export -->
        <Card class="q-glass result-card">
          <template #content>
            <div class="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
              <div>
                <p class="font-semibold text-slate-800 dark:text-slate-100 mb-0.5">{{ t('classify.export_title') }}</p>
                <p class="text-sm text-slate-500 dark:text-slate-400">
                  {{ t('classify.export_subtitle') }}
                </p>
              </div>
              <div class="flex flex-shrink-0 gap-2">
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
      <!-- Noisy Results -->
<div v-if="noisyResults && noisyTopResult" class="space-y-4 animate-fadein">
  
  <!-- Section divider -->
  <div class="flex items-center gap-3 pt-2">
    <div class="flex-1 border-t border-dashed border-slate-200 dark:border-slate-700" />
    <span class="px-3 py-1 font-mono text-xs tracking-widest border rounded-full text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950">
      {{ t('classify.noisy_results') }} · {{ t('classify.noise_severity') }} {{ appliedNoiseLevel?.toFixed(2) }}
    </span>
    <div class="flex-1 border-t border-dashed border-slate-200 dark:border-slate-700" />
  </div>

  <!-- Noisy Top Prediction -->
  <Card class="q-glass result-card transition-colors duration-300 border-l-4 border-l-amber-500 dark:border-l-amber-400">
    <template #content>
      <div class="flex items-center gap-4">
        <i class="flex-shrink-0 text-4xl pi pi-sliders-h text-amber-500 dark:text-amber-400" />
        <div class="flex-1 min-w-0">
          <p class="mb-1 font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500">
            {{ t('classify.noisy_top_prediction') }}
          </p>
          <p class="mb-1 text-xl font-bold text-[var(--q-text)]">
            {{ t('classify.' + noisyTopResult.prediction) }}
          </p>
          <p class="font-mono text-xs text-[var(--q-muted)]">
            {{ noisyTopResult.modelName }} &nbsp;·&nbsp;
            {{ noisyTopResult.confidence.toFixed(1) }}% {{ t('classify.confidence').toLowerCase() }}
          </p>
        </div>
      </div>
    </template>
  </Card>

  <!-- Noisy Model Comparison Table -->
  <Card class="q-glass result-card">
    <template #title>
      <span class="font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500">
        {{ t('classify.model_comparison') }} · {{ t('classify.noisy_label') }}
      </span>
    </template>
    <template #content>
      <DataTable :value="noisyResults"  responsiveLayout="scroll">
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
            severity="danger"
          />
          </template>
        </Column>
        <Column field="confidence" :header="t('classify.confidence')">
          <template #body="{ data }">
            <div class="flex items-center gap-3 min-w-40">
              <ProgressBar
                :value="parseFloat(data.confidence.toFixed(1))"
                :showValue="false"
                class="flex-1"
                :pt="{
                  root: { style: 'height: 6px;' },
                  value: { style: `background: ${data.color};` }
                }"
              />
              <span class="w-12 font-mono text-xs text-right text-slate-500 dark:text-slate-400 shrink-0">
                {{ data.confidence.toFixed(1) }}%
              </span>
            </div>
          </template>
        </Column>
        <Column field="latency" :header="t('classify.latency')">
          <template #body="{ data }">
            <span class="font-mono text-xs text-slate-500 dark:text-slate-400">
              {{ data.latency.toFixed(1) }} ms
            </span>
          </template>
        </Column>
      </DataTable>
    </template>
  </Card>
</div>
    </div>
</template>

<style scoped>
.classify-shell {
  max-width: 1180px;
  margin: 0 auto;
  padding: 1rem 1rem 3rem;
}

.classify-header {
  max-width: 760px;
  margin: 0 auto 2rem;
  text-align: center;
}

.classify-eyebrow {
  display: inline-block;
  margin-bottom: 0.9rem;
  padding: 0.45rem 0.85rem;
  border-radius: 999px;
  border: 1px solid var(--q-bar-border);
  background: var(--q-teal-soft);
  color: var(--q-teal);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.classify-title {
  margin: 0 0 0.7rem;
  color: var(--q-text);
  font-family: var(--q-font-display);
  font-size: clamp(2.3rem, 4vw, 3.6rem);
  line-height: 1.05;
}

.classify-subtitle {
  margin: 0;
  color: var(--q-muted);
  font-size: 1rem;
  line-height: 1.9;
}

.classify-card,
.result-card {
  border-radius: var(--q-radius-lg);
}

.classify-card :deep(.p-card-body),
.result-card :deep(.p-card-body) {
  padding: 1.25rem;
}

.upload-dropzone {
  padding: 4rem 1.5rem;
  text-align: center;
  border: 1.5px dashed var(--q-bar-border);
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.35);
  cursor: pointer;
  transition: border-color 0.2s ease, transform 0.2s ease, background 0.2s ease;
}

.p-dark .upload-dropzone {
  background: rgba(255, 255, 255, 0.05);
}

.upload-dropzone:hover {
  border: 2px dashed rgba(42, 184, 184, 0.411);
  background: var(--q-teal-soft);
  transform: translateY(-2px);
}

.upload-dropzone__icon-wrap {
  display: inline-grid;
  place-items: center;
  width: 76px;
  height: 76px;
  margin-bottom: 1rem;
  border-radius: 22px;
  background: var(--q-surface-soft);
}

.upload-dropzone__icon {
  font-size: 2rem;
  color: var(--q-teal);
}

.upload-dropzone__title {
  margin: 0 0 0.45rem;
  color: var(--q-text);
  font-size: 1rem;
  font-weight: 700;
}

.upload-dropzone__meta {
  margin: 0;
  color: var(--q-muted);
  font-size: 0.9rem;
}

.upload-dropzone__link {
  color: var(--q-teal);
  font-weight: 700;
}

.classify-preview-stack {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.preview-frame {
  position: relative;
  display: flex;
  justify-content: center;
  overflow: hidden;
  border: 1px solid var(--q-bar-border);
  border-radius: 24px;
  background: var(--q-bar-bg);
}

.preview-overlay {
  position: absolute;
  inset: auto 0 0 0;
  padding: 1rem;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.62), transparent);
}

.preview-filename {
  margin: 0;
  color: white;
  font-size: 0.78rem;
  font-family: monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.noise-panel {
  padding: 1rem;
  border: 1px solid var(--q-bar-border);
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.32);
}

.p-dark .noise-panel {
  background: rgba(255, 255, 255, 0.06);
}

.noise-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.noise-panel__title {
  margin: 0 0 0.2rem;
  color: var(--q-text);
  font-size: 0.95rem;
  font-weight: 700;
}

.noise-panel__subtitle {
  margin: 0;
  color: var(--q-muted);
  font-size: 0.85rem;
}

.noise-panel__body {
  padding-top: 0.9rem;
}

.noise-panel__meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.85rem;
  gap: 0.75rem;
}

.noise-panel__label {
  color: var(--q-muted);
  font-size: 0.8rem;
  font-weight: 600;
}

.noise-panel__badge {
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 700;
  border: 1px solid transparent;
}

.noise-panel__badge--mild {
  background: rgba(250, 204, 21, 0.14);
  color: #ca8a04;
  border-color: rgba(250, 204, 21, 0.28);
}

.noise-panel__badge--degraded {
  background: rgba(249, 115, 22, 0.14);
  color: #ea580c;
  border-color: rgba(249, 115, 22, 0.28);
}

.noise-panel__badge--severe {
  background: rgba(239, 68, 68, 0.14);
  color: #dc2626;
  border-color: rgba(239, 68, 68, 0.28);
}
.p-dark .noise-panel__badge--mild {
  color: #facc15;
}

.p-dark .noise-panel__badge--degraded {
  color: #fb923c;
}

.p-dark .noise-panel__badge--severe {
  color: #f87171;
}

.noise-panel__scale {
  display: flex;
  justify-content: space-between;
  margin-top: 0.65rem;
  color: var(--q-muted);
  font-size: 0.76rem;
}

.classify-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  padding-top: 0.25rem;
}


:deep(.classify-run-btn.p-button),
:deep(.classify-run-btn.p-button:hover),
:deep(.classify-run-btn.p-button:active),
:deep(.classify-run-btn.p-button:focus),
:deep(.classify-run-btn.p-button:focus-visible) {
  background: var(--q-teal) !important;
  border-color: var(--q-teal) !important;
  color: white !important;
  background-image: none !important;
  filter: none !important;
  opacity: 1 !important;
  box-shadow: 0 14px 34px rgba(42, 184, 184, 0.22)
}

:deep(.noise-panel .p-toggleswitch) {
  --p-toggleswitch-width: 3rem;
  --p-toggleswitch-height: 1.7rem;
}

:deep(.noise-panel .p-toggleswitch .p-toggleswitch-slider) {
  background: rgba(13, 31, 45, 0.16) !important;
  border: 1px solid var(--q-bar-border) !important;
}

:deep(.noise-panel .p-toggleswitch.p-toggleswitch-checked .p-toggleswitch-slider) {
  background: var(--q-teal) !important;
}

:deep(.noise-panel .p-slider .p-slider-range) {
  background: var(--q-teal) !important;
}

:deep(.noise-panel .p-slider .p-slider-handle) {
  border-color: var(--q-teal) !important;
  background: white !important;
}

.p-dark :deep(.noise-panel .p-slider .p-slider-handle) {
  background: var(--q-navy) !important;
}
.noise-slider-wrap,
.noise-slider-wrap :deep(.p-slider) {
  direction: ltr !important;
}

:deep(.classify-run-btn.p-button:hover:not(:disabled)) {
  background: var(--q-teal) !important;
  border-color: var(--q-teal) !important;
}


.classify-choose-row {
  display: flex;
  justify-content: center;
  margin: 1.25rem 0 1.5rem;
}

.results-stack {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1.5rem;
}

.animate-fadein {
  animation: fadein 0.35s ease both;
}

@keyframes fadein {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.25s ease;
  overflow: hidden;
}

.slide-down-enter-from,
.slide-down-leave-to {
  opacity: 0;
  max-height: 0;
}

.slide-down-enter-to,
.slide-down-leave-from {
  opacity: 1;
  max-height: 200px;
}

@media (max-width: 640px) {
  .classify-shell {
    padding-inline: 0.75rem;
  }

  .classify-actions {
    flex-direction: column;
  }

  .noise-panel__header,
  .noise-panel__meta {
    align-items: flex-start;
    flex-direction: column;
  }
}
.lang-ar .noise-panel__scale,
.lang-ar .noise-panel__scale span {
    direction: ltr !important;
    unicode-bidi: normal !important;
}
.classify-shell--ar .upload-dropzone__title,
.classify-shell--ar .upload-dropzone__meta,
.classify-shell--ar .noise-panel__title,
.classify-shell--ar .noise-panel__subtitle,
.classify-shell--ar .noise-panel__label,
.classify-shell--ar .form-text,
.classify-shell--ar .result-text {
  direction: rtl;
  unicode-bidi: plaintext;
}
.classify-shell--ar .classify-header,
.classify-shell--ar .upload-dropzone {
  text-align: center;
}
.lang-ar .p-slider,
.lang-ar .p-slider * {
    direction: ltr !important;
}
.chart-wrap {
  direction: ltr;
}

.chart-wrap--center {
  display: flex;
  justify-content: center;
}
.chart-card-title--ar {
  direction: rtl;
  text-align: right;
  unicode-bidi: plaintext;
  letter-spacing: 0;
  text-transform: none;
}
.classify-shell--ar :deep(.p-card-title) {
  text-align: right;
}
.restored-banner {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.65rem 1rem;
  border-radius: 14px;
  border: 1px solid rgba(42, 184, 184, 0.25);
  background: var(--q-teal-soft);
  color: var(--q-teal);
  font-size: 0.82rem;
  font-weight: 600;
}
.lang-ar .restored-banner{
  text-align: right;
  direction: rtl;
}
html[lang="ar"] .restored-banner{
  text-align: right;
  direction: rtl;
}
</style>
