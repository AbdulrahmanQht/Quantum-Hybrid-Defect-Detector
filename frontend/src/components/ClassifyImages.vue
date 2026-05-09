<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { Info } from 'lucide-vue-next'
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
const activeResultsView = ref('clean')
const noisyPreviewUrl = ref(null)

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

const resultsViewOptions = computed(() => [
  { label: t('classify.clean'), value: 'clean' },
  ...(noisyResults.value ? [{ label: t('classify.noisy'), value: 'noisy' }] : []),
  ...(noisyResults.value ? [{ label: t('classify.compare_label'), value: 'compare' }] : [])
])

// --- Chart Configuration ---
const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
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
  responsive: true,
  maintainAspectRatio: false,
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
      fileName: selectedFile.value?.name,
      previewDataUrl: storedDataUrl.value,
      results: results.value,
      noisyResults: noisyResults.value,
      compareWithNoise: compareWithNoise.value,
      noiseLevel: noiseLevel.value,
      appliedNoiseLevel: appliedNoiseLevel.value,
      noisyPreviewUrl: noisyPreviewUrl.value
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
    previewUrl.value = data.clean_image_base64
    storedDataUrl.value = data.clean_image_base64
    if (data.noisy) {
      noisyResults.value = parseSet(data.noisy)
      noisyPreviewUrl.value = data.noisy.noisy_image_base64 || null
    }
    appliedNoiseLevel.value = snapshotNoiseLevel
    if (data.noisy) noisyResults.value = parseSet(data.noisy)
    activeResultsView.value = 'clean'

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
      label: t('classify.confidence_percent'),
      data: results.value.map(r => parseFloat(r.confidence.toFixed(2))),
      backgroundColor: results.value.map(r => r.color),
      borderRadius: 5,
      borderSkipped: false
    }]
  }
})

const noisyConfidenceChartData = computed(() => {
  if (!noisyResults.value) return null
  return {
    labels: noisyResults.value.map(r => r.modelName),
    datasets: [{
      label: t('classify.confidence_percent'),
      data: noisyResults.value.map(r => parseFloat(r.confidence.toFixed(2))),
      backgroundColor: noisyResults.value.map(r => r.color),
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
      label: t('classify.latency_ms'),
      data: results.value.map(r => parseFloat(r.latency.toFixed(2))),
      backgroundColor: results.value.map(r => r.color),
      borderRadius: 5,
      borderSkipped: false
    }]
  }
})

const noisyLatencyChartData = computed(() => {
  if (!noisyResults.value) return null
  return {
    labels: noisyResults.value.map(r => r.modelName),
    datasets: [{
      label: t('classify.latency_ms'),
      data: noisyResults.value.map(r => parseFloat(r.latency.toFixed(2))),
      backgroundColor: noisyResults.value.map(r => r.color),
      borderRadius: 5,
      borderSkipped: false
    }]
  }
})

const comparisonChartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'bottom',
      labels: { color: '#94a3b8', padding: 16, font: { size: 12 } }
    }
  },
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

const cleanNoisyComparisonRows = computed(() => {
  if (!results.value || !noisyResults.value) return []

  return results.value
    .map((clean) => {
      const noisy = noisyResults.value.find(item => item.modelKey === clean.modelKey)
      if (!noisy) return null

      return {
        modelKey: clean.modelKey,
        modelName: clean.modelName,
        color: clean.color,
        cleanPrediction: clean.prediction,
        noisyPrediction: noisy.prediction,
        cleanConfidence: clean.confidence,
        noisyConfidence: noisy.confidence,
        confidenceDelta: noisy.confidence - clean.confidence,
        cleanLatency: clean.latency,
        noisyLatency: noisy.latency,
        latencyDelta: noisy.latency - clean.latency
      }
    })
    .filter(Boolean)
})

const confidenceComparisonChartData = computed(() => {
  if (!cleanNoisyComparisonRows.value.length) return null

  return {
    labels: cleanNoisyComparisonRows.value.map(r => r.modelName),
    datasets: [
      {
        label: t('classify.clean_label'),
        data: cleanNoisyComparisonRows.value.map(r => parseFloat(r.cleanConfidence.toFixed(2))),
        backgroundColor: 'rgba(42, 184, 184, 0.78)',
        borderRadius: 5,
        borderSkipped: false
      },
      {
        label: t('classify.noisy_label'),
        data: cleanNoisyComparisonRows.value.map(r => parseFloat(r.noisyConfidence.toFixed(2))),
        backgroundColor: 'rgba(245, 158, 11, 0.78)',
        borderRadius: 5,
        borderSkipped: false
      }
    ]
  }
})

const cnnIsHighestConfidenceUnderNoise = computed(() => {
  if (!noisyResults.value || noisyResults.value.length < 2) return false
  const cnn = noisyResults.value.find(r => r.modelKey === 'CNN')
  if (!cnn) return false
  return noisyResults.value.every(r => r.modelKey === 'CNN' || cnn.confidence >= r.confidence)
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
  activeResultsView.value = 'clean'
  if (fileInput.value) fileInput.value.value = ''
  localStorage.removeItem(STORAGE_KEY)
  isRestored.value = false
  storedDataUrl.value = null
  appliedNoiseLevel.value = null
  noisyPreviewUrl.value = null
}

// --- Export ---
function exportToCSV() {
  if (!results.value) return

  const headers = [
    'Condition',
    'Model Name',
    'Prediction Class',
    'Confidence (%)',
    'Latency (ms)',
    'Match Status',
    'Confidence Delta (%)',
    'Latency Delta (ms)'
  ]
  const rows = [headers.join(',')]

  // 2. Add Clean Rows (Baseline)
  results.value.forEach(r => {
    rows.push(`"Clean","${r.modelName}","${r.prediction}",${r.confidence.toFixed(2)},${r.latency.toFixed(2)},"-","-","-"`)
  })

  // 3. Add Noisy Rows with Comparison Data
  if (noisyResults.value && cleanNoisyComparisonRows.value.length > 0) {
    cleanNoisyComparisonRows.value.forEach(c => {
      const condition = `"Noisy (level ${appliedNoiseLevel.value?.toFixed(2)})"`
      const matchStatus = c.cleanPrediction === c.noisyPrediction ? 'Match' : 'Mismatch'
      const confDelta = `${c.confidenceDelta >= 0 ? '+' : ''}${c.confidenceDelta.toFixed(2)}`
      const latDelta = `${c.latencyDelta >= 0 ? '+' : ''}${c.latencyDelta.toFixed(2)}`

      rows.push(`${condition},"${c.modelName}","${c.noisyPrediction}",${c.noisyConfidence.toFixed(2)},${c.noisyLatency.toFixed(2)},"${matchStatus}","${confDelta}","${latDelta}"`)
    })
  }

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
  // eslint-disable-next-line no-unused-vars
  return results.map(({ color, ...rest }) => rest)
}

function exportToJSON() {
  if (!results.value) return

  const formatResults = (dataArr) => dataArr.map(r => ({
    ...stripColor([r])[0],
    confidence: `${r.confidence.toFixed(2)}%`,
    latency: `${r.latency.toFixed(2)}ms`
  }))

  const jsonData = {
    timestamp: new Date().toISOString(),
    fileName: sanitizeFilename(selectedFile.value?.name),

    // Clean Section with Units
    clean: {
      topPrediction: formatResults([topResult.value])[0],
      allResults: formatResults(results.value)
    },

    // Noisy Section with Units
    ...(noisyResults.value && {
      noisy: {
        noiseLevel: appliedNoiseLevel.value,
        topPrediction: formatResults([noisyTopResult.value])[0],
        allResults: formatResults(noisyResults.value)
      },

      // Comparison Section
      comparison: cleanNoisyComparisonRows.value.map(c => ({
        modelKey: c.modelKey,
        modelName: c.modelName,
        isMatch: c.cleanPrediction === c.noisyPrediction,
        cleanLabel: c.cleanPrediction,
        noisyLabel: c.noisyPrediction,
        confidence: `${c.noisyConfidence.toFixed(2)}%`,
        latency: `${c.noisyLatency.toFixed(2)}ms`,
        confidenceDelta: `${c.confidenceDelta >= 0 ? '+' : ''}${c.confidenceDelta.toFixed(2)}%`,
        latencyDelta: `${c.latencyDelta >= 0 ? '+' : ''}${c.latencyDelta.toFixed(2)}ms`
      }))
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
    if (saved.results) results.value = saved.results
    if (saved.noisyResults) noisyResults.value = saved.noisyResults
    activeResultsView.value = 'clean'

    // Locked-in noise level for the restored results
    if (saved.appliedNoiseLevel != null) appliedNoiseLevel.value = saved.appliedNoiseLevel

    // Watched slider refs last — watchers may fire here, results are already in place
    compareWithNoise.value = saved.compareWithNoise ?? false
    noiseLevel.value = saved.noiseLevel ?? 0.3

    // Preview / file sentinel
    if (saved.previewDataUrl) {
      previewUrl.value = saved.previewDataUrl
      storedDataUrl.value = saved.previewDataUrl
      selectedFile.value = { name: saved.fileName ?? 'image', restored: true }
      isRestored.value = true
    }
    if (saved.noisyPreviewUrl) noisyPreviewUrl.value = saved.noisyPreviewUrl
  } catch {
    // Ignore corrupted storage
  }
})

watch(results, persistState, { deep: true })
watch(noisyResults, persistState, { deep: true })
watch([compareWithNoise, noiseLevel], persistState)
watch(noisyResults, (value) => {
  if (!value && activeResultsView.value !== 'clean') activeResultsView.value = 'clean'
})

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
      <h1 class="classify-title">
        {{ t('classify.title') }}
      </h1>
      <p class="classify-subtitle">
        {{ t('classify.subtitle') }}
      </p>
    </div>

    <Card class="classify-card q-glass">
      <template #content>
        <input ref="fileInput" type="file" accept=".jpg,.jpeg,.png,.webp" class="hidden" @change="onFileChange">

        <div v-if="!selectedFile" class="upload-dropzone" @click="fileInput.click()"
          @keydown.enter.prevent="fileInput.click()" @keydown.space.prevent="fileInput.click()" @dragover.prevent
          @drop.prevent="onDrop">
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
          <div class="preview-frame" :class="{ 'preview-frame--split': noisyPreviewUrl }">
            <div class="preview-pane">
              <Image :src="previewUrl" :alt="selectedFile.name" image-class="preview-img" preview />
              <span v-if="noisyPreviewUrl" class="preview-pane__label flex items-center gap-1.5">
                {{ t('classify.clean') }}
                <Info v-tooltip.top="t('classify.tooltips.clean_image')"
                  class="classify-info-icon-img w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </span>
            </div>

            <Transition name="slide-in">
              <div v-if="noisyPreviewUrl" class="preview-pane preview-pane--noisy">
                <Image :src="noisyPreviewUrl" :alt="t('classify.noisy')" image-class="preview-img" preview />
                <span class="preview-pane__label flex items-center gap-1.5" :class="{
                  'preview-pane__label--mild': appliedNoiseLevel <= 0.3,
                  'preview-pane__label--degraded': appliedNoiseLevel > 0.3 && appliedNoiseLevel <= 0.6,
                  'preview-pane__label--severe': appliedNoiseLevel > 0.6
                }">
                  {{ t('classify.noisy') }} · {{ appliedNoiseLevel?.toFixed(2) }}
                  <Info v-tooltip.top="t('classify.tooltips.noisy_image')"
                    class="classify-info-icon-img w-3.5 h-3.5 cursor-help flex-shrink-0" />
                </span>
              </div>
            </Transition>
          </div>

          <div class="noise-panel" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
            <div class="noise-panel__header">
              <div>
                <p class="noise-panel__title">
                  {{ t('classify.noise_toggle') }}
                </p>
                <p class="noise-panel__subtitle">
                  {{ t('classify.noise_toggle_sub') }}
                </p>
              </div>
              <ToggleSwitch v-model="compareWithNoise" />
            </div>

            <Transition name="slide-down">
              <div v-if="compareWithNoise" class="noise-panel__body">
                <div class="noise-panel__meta">
                  <span class="noise-panel__label">{{ t('classify.noise_severity') }}</span>
                  <span class="noise-panel__badge" :class="{
                    'noise-panel__badge--mild': noiseLevel <= 0.3,
                    'noise-panel__badge--degraded': noiseLevel > 0.3 && noiseLevel <= 0.6,
                    'noise-panel__badge--severe': noiseLevel > 0.6
                  }">
                    {{ noiseSeverityLabel }} · {{ noiseLevel.toFixed(2) }}
                  </span>
                </div>

                <div class="noise-slider-wrap" dir="ltr">
                  <Slider v-model="sliderNoiseLevel" :min="0.05" :max="1.0" :step="0.05" class="w-full noise-slider" />
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
            <Button :label="t('classify.reset')" icon="pi pi-refresh" severity="secondary" outlined
              :disabled="loading || validating" @click="reset" />
            <Button :label="validating ? t('classify.validating') : loading ? t('classify.running') : t('classify.run')"
              :icon="loading || validating ? 'pi pi-spin pi-spinner' : 'pi pi-play'"
              :disabled="loading || validating || isRestored" class="classify-run-btn" @click="uploadImage" />
          </div>
        </div>
      </template>
    </Card>

    <div v-if="!selectedFile" class="classify-choose-row">
      <Button :label="t('classify.choose')" icon="pi pi-folder-open" severity="contrast" :loading="validating"
        @click="fileInput.click()" />
    </div>

    <Message v-if="error" severity="error" :closable="true" role="alert" class="mb-4" style="margin-top: 1rem;"
      @close="error = null">
      {{ error }}
    </Message>

    <div v-if="results && topResult" class="results-stack animate-fadein">
      <div class="result-summary-grid" :class="{ 'result-summary-grid--single': !noisyTopResult }">
        <!-- Top Prediction Card -->
        <Card class="q-glass result-card result-summary-card result-summary-card--clean transition-colors duration-300">
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
                  {{ topResult.confidence.toFixed(2) }}% {{ t('classify.confidence').toLowerCase() }} &nbsp;·&nbsp;
                  {{ topResult.latency.toFixed(2) }}ms
                </p>
              </div>
            </div>
          </template>
        </Card>

        <Card v-if="noisyTopResult"
          class="q-glass result-card result-summary-card result-summary-card--noisy transition-colors duration-300">
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
                  {{ noisyTopResult.confidence.toFixed(2) }}% {{ t('classify.confidence').toLowerCase() }} &nbsp;
                  ·&nbsp;
                  {{ noisyTopResult.latency.toFixed(2) }}ms
                </p>
              </div>
            </div>
          </template>
        </Card>
      </div>

      <div v-if="compareWithNoise" class="results-view-switch" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
        <SelectButton v-model="activeResultsView" :options="resultsViewOptions" option-label="label"
          option-value="value" :allow-empty="false" />
      </div>

      <div v-if="activeResultsView === 'clean'" class="results-view-panel animate-fadein">
        <!-- Model Comparison Table -->
        <Card class="q-glass result-card">
          <template #title>
            <div class="flex items-center gap-2">
              <span class="font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500">
                {{ t('classify.model_comparison') }}
              </span>
              <Info v-tooltip.top="t('classify.tooltips.model_comparison')"
                class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
            </div>
          </template>
          <template #content>
            <DataTable :value="results" responsive-layout="scroll" class="classify-results-table">
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
                  <Tag :value="t('classify.' + data.prediction)" severity="danger" />
                </template>
              </Column>

              <Column field="confidence" :header="t('classify.confidence')">
                <template #body="{ data }">
                  <div class="flex items-center gap-3 min-w-40">
                    <ProgressBar :value="parseFloat(data.confidence.toFixed(2))" :show-value="false" class="flex-1" :pt="{
                      root: { style: 'height: 6px;' },
                      value: { style: `background: ${data.color};` }
                    }" />
                    <span class="w-12 font-mono text-xs text-right text-slate-500 dark:text-slate-400 shrink-0">
                      {{ data.confidence.toFixed(2) }}%
                    </span>
                  </div>
                </template>
              </Column>

              <Column field="latency" :header="t('classify.latency')">
                <template #body="{ data }">
                  <span class="font-mono text-xs text-slate-500 dark:text-slate-400">
                    {{ data.latency.toFixed(2) }} ms
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
              <div class="flex items-center gap-2" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
                <span
                  class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                  :class="{ 'chart-card-title--ar': locale === 'AR' }">
                  {{ t('classify.confidence_scores') }}
                </span>
                <Info v-tooltip.top="t('classify.tooltips.confidence_scores')"
                  class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
            </template>
            <template #content>
              <div class="chart-wrap" role="img" :aria-label="t('classify.confidence_scores')" dir="ltr">
                <Chart type="bar" :data="confidenceChartData" :options="chartOptions" class="h-52" />
              </div>
            </template>
          </Card>

          <Card class="q-glass result-card">
            <template #title>
              <div class="flex items-center gap-2" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
                <span
                  class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                  :class="{ 'chart-card-title--ar': locale === 'AR' }">
                  {{ t('classify.inference_latency') }}
                </span>
                <Info v-tooltip.top="t('classify.tooltips.inference_latency')"
                  class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
            </template>
            <template #content>
              <div class="chart-wrap" role="img" :aria-label="t('classify.inference_latency')" dir="ltr">
                <Chart type="bar" :data="latencyChartData" :options="chartOptions" class="h-52" />
              </div>
            </template>
          </Card>

          <Card class="q-glass result-card lg:col-span-2">
            <template #title>
              <div class="flex items-center gap-2" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
                <span
                  class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                  :class="{ 'chart-card-title--ar': locale === 'AR' }">
                  {{ t('classify.distribution') }}
                </span>
                <Info v-tooltip.top="t('classify.tooltips.distribution')"
                  class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
            </template>
            <template #content>
              <div class="flex justify-center">
                <div class="chart-wrap chart-wrap--center" role="img" :aria-label="t('classify.distribution')"
                  dir="ltr">
                  <Chart type="pie" :data="confidenceChartData" :options="pieOptions" class="w-full max-w-sm h-60" />
                </div>
              </div>
            </template>
          </Card>
        </div>
      </div>

      <!-- Noisy Results -->
      <div v-if="noisyResults && noisyTopResult && activeResultsView === 'noisy'"
        class="results-view-panel animate-fadein">
        <!-- Noisy Model Comparison Table -->
        <Card class="q-glass result-card">
          <template #title>
            <div class="noisy-card-header">
              <div class="flex items-center gap-2">
                <span class="font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500">
                  {{ t('classify.model_comparison') }} · {{ t('classify.noisy_label') }}
                </span>
                <Info v-tooltip.top="t('classify.tooltips.noisy_confidence')"
                  class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
              <span class="noise-level-chip">
                {{ t('classify.noise_severity') }} {{ appliedNoiseLevel?.toFixed(2) }}
              </span>
            </div>
          </template>
          <template #content>
            <DataTable :value="noisyResults" responsive-layout="scroll" class="classify-results-table">
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
                  <Tag :value="t('classify.' + data.prediction)" severity="danger" />
                </template>
              </Column>
              <Column field="confidence" :header="t('classify.confidence')">
                <template #body="{ data }">
                  <div class="flex items-center gap-3 min-w-40">
                    <ProgressBar :value="parseFloat(data.confidence.toFixed(2))" :show-value="false" class="flex-1" :pt="{
                      root: { style: 'height: 6px;' },
                      value: { style: `background: ${data.color};` }
                    }" />
                    <span class="w-12 font-mono text-xs text-right text-slate-500 dark:text-slate-400 shrink-0">
                      {{ data.confidence.toFixed(2) }}%
                    </span>
                  </div>
                </template>
              </Column>
              <Column field="latency" :header="t('classify.latency')">
                <template #body="{ data }">
                  <span class="font-mono text-xs text-slate-500 dark:text-slate-400">
                    {{ data.latency.toFixed(2) }} ms
                  </span>
                </template>
              </Column>
            </DataTable>
          </template>
        </Card>

        <!-- CNN overconfidence warning -->
        <Transition name="slide-down">
          <div v-if="cnnIsHighestConfidenceUnderNoise" class="classify-overconfidence-banner">
            <Info class="w-4 h-4 flex-shrink-0" />
            <span>{{ t('classify.cnn_overconfidence_warning') }}</span>
          </div>
        </Transition>

        <!-- Noisy Charts -->
        <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card class="q-glass result-card">
            <template #title>
              <div class="flex items-center gap-2" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
                <span
                  class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                  :class="{ 'chart-card-title--ar': locale === 'AR' }">
                  {{ t('classify.noisy_confidence_scores') }}
                </span>
                <Info v-tooltip.top="t('classify.tooltips.noisy_confidence')"
                  class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
            </template>
            <template #content>
              <div class="chart-wrap" role="img" :aria-label="t('classify.noisy_confidence_scores')" dir="ltr">
                <Chart type="bar" :data="noisyConfidenceChartData" :options="chartOptions" class="h-52" />
              </div>
            </template>
          </Card>

          <Card class="q-glass result-card">
            <template #title>
              <div class="flex items-center gap-2" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
                <span
                  class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                  :class="{ 'chart-card-title--ar': locale === 'AR' }">
                  {{ t('classify.noisy_inference_latency') }}
                </span>
                <Info v-tooltip.top="t('classify.tooltips.inference_latency')"
                  class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
            </template>
            <template #content>
              <div class="chart-wrap" role="img" :aria-label="t('classify.noisy_inference_latency')" dir="ltr">
                <Chart type="bar" :data="noisyLatencyChartData" :options="chartOptions" class="h-52" />
              </div>
            </template>
          </Card>

          <Card class="q-glass result-card lg:col-span-2">
            <template #title>
              <div class="flex items-center gap-2" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
                <span
                  class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                  :class="{ 'chart-card-title--ar': locale === 'AR' }">
                  {{ t('classify.noisy_distribution') }}
                </span>
                <Info v-tooltip.top="t('classify.tooltips.distribution')"
                  class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
            </template>
            <template #content>
              <div class="flex justify-center">
                <div class="chart-wrap chart-wrap--center" role="img" :aria-label="t('classify.noisy_distribution')"
                  dir="ltr">
                  <Chart type="pie" :data="noisyConfidenceChartData" :options="pieOptions"
                    class="w-full max-w-sm h-60" />
                </div>
              </div>
            </template>
          </Card>
        </div>
      </div>

      <!-- Clean vs Noisy Comparison -->
      <div v-if="noisyResults && noisyTopResult && activeResultsView === 'compare'"
        class="results-view-panel animate-fadein">
        <Card class="q-glass result-card">
          <template #title>
            <div class="noisy-card-header">
              <div class="flex items-center gap-2">
                <span class="font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500">
                  {{ t('classify.clean_noisy_delta') }}
                </span>
                <Info v-tooltip.top="t('classify.tooltips.clean_noisy_delta')"
                  class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
              <span class="noise-level-chip">
                {{ t('classify.noise_severity') }} {{ appliedNoiseLevel?.toFixed(2) }}
              </span>
            </div>
          </template>
          <template #content>
            <DataTable :value="cleanNoisyComparisonRows" responsive-layout="scroll" class="classify-results-table">
              <Column field="modelName" :header="t('classify.model')">
                <template #body="{ data }">
                  <div class="flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" :style="{ background: data.color }" />
                    <span class="font-medium text-slate-800 dark:text-slate-200">{{ data.modelName }}</span>
                  </div>
                </template>
              </Column>
              <Column :header="t('classify.clean_label')">
                <template #body="{ data }">
                  <div class="flex flex-col leading-tight">
                    <span class="font-semibold text-slate-700 dark:text-slate-200">
                      {{ t('classify.' + data.cleanPrediction) }}
                    </span>
                    <span class="font-mono text-[11px] text-slate-500 dark:text-slate-400">
                      {{ data.cleanConfidence.toFixed(2) }}% {{ t('classify.confidence').toLowerCase() }}
                    </span>
                  </div>
                </template>
              </Column>

              <Column :header="t('classify.noisy_label')">
                <template #body="{ data }">
                  <div class="flex flex-col leading-tight">
                    <span class="font-semibold text-slate-700 dark:text-slate-200">
                      {{ t('classify.' + data.noisyPrediction) }}
                    </span>
                    <span class="font-mono text-[11px] text-slate-500 dark:text-slate-400">
                      {{ data.noisyConfidence.toFixed(2) }}% {{ t('classify.confidence').toLowerCase() }}
                    </span>
                  </div>
                </template>
              </Column>
              <Column :header="t('classify.prediction_match')">
                <template #body="{ data }">
                  <Tag :value="data.cleanPrediction === data.noisyPrediction
                    ? t('classify.match')
                    : t('classify.mismatch')"
                    :severity="data.cleanPrediction === data.noisyPrediction ? 'success' : 'danger'" />
                </template>
              </Column>
              <Column :header="t('classify.delta_confidence')">
                <template #body="{ data }">
                  <Tag :value="`${data.confidenceDelta >= 0 ? '+' : ''}${data.confidenceDelta.toFixed(2)}%`"
                    :severity="data.confidenceDelta < 0 ? 'warning' : 'success'" />
                </template>
              </Column>
              <Column :header="t('classify.delta_latency')">
                <template #body="{ data }">
                  <div class="flex flex-col font-mono text-m leading-tight">
                    <span class="text-[12px] text-slate-400 dark:text-slate-500">
                      {{ data.cleanLatency.toFixed(2) }}ms ➔ {{ data.noisyLatency.toFixed(2) }}ms
                    </span>
                    <span :class="data.latencyDelta > 0 ? 'text-amber-500' : 'text-emerald-500'" class="font-bold">
                      {{ data.latencyDelta >= 0 ? '+' : '' }}{{ data.latencyDelta.toFixed(2) }} ms
                    </span>
                  </div>
                </template>
              </Column>
            </DataTable>
          </template>
        </Card>
        <!-- CNN overconfidence warning -->
        <Transition name="slide-down">
          <div v-if="cnnIsHighestConfidenceUnderNoise" class="classify-overconfidence-banner">
            <Info class="w-4 h-4 flex-shrink-0" />
            <span>{{ t('classify.cnn_overconfidence_warning') }}</span>
          </div>
        </Transition>
        <Card class="q-glass result-card">
          <template #title>
            <div class="flex items-center gap-2" :dir="locale === 'AR' ? 'rtl' : 'ltr'">
              <span
                class="chart-card-title font-mono text-xs tracking-widest uppercase text-slate-400 dark:text-slate-500"
                :class="{ 'chart-card-title--ar': locale === 'AR' }">
                {{ t('classify.clean_vs_noisy') }}
              </span>
              <Info v-tooltip.top="t('classify.tooltips.clean_vs_noisy')"
                class="classify-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
            </div>
          </template>
          <template #content>
            <div class="chart-wrap" role="img" :aria-label="t('classify.clean_vs_noisy')" dir="ltr">
              <Chart type="bar" :data="confidenceComparisonChartData" :options="comparisonChartOptions"
                class="w-full h-72" />
            </div>
          </template>
        </Card>
      </div>

      <!-- Export -->
      <Card v-if="results && topResult" class="q-glass result-card results-export-card">
        <template #content>
          <div class="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
            <div>
              <p class="font-semibold text-slate-800 dark:text-slate-100 mb-0.5">
                {{ t('classify.export_title') }}
              </p>
              <p class="text-sm text-slate-500 dark:text-slate-400">
                {{ t('classify.export_subtitle') }}
              </p>
            </div>
            <div class="flex flex-shrink-0 gap-2">
              <Button label="CSV" icon="pi pi-file-excel" severity="success" outlined size="small"
                data-testid="export-csv-btn" @click="exportToCSV" />
              <Button label="JSON" icon="pi pi-file" severity="info" outlined size="small" data-testid="export-json-btn"
                @click="exportToJSON" />
            </div>
          </div>
        </template>
      </Card>
    </div>

    <!-- Export success -->
    <Message v-if="exportSuccess" severity="success" :closable="true" @close="exportSuccess = null">
      {{ exportSuccess }}
    </Message>
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

/* ── Single-pane (default) — original style, untouched ── */
.preview-frame {
  position: relative;
  display: flex;
  justify-content: center;
  overflow: hidden;
  border: 1px solid var(--q-bar-border);
  border-radius: 24px;
  background: var(--q-bar-bg);
}

/* Single pane fills the frame transparently */
.preview-frame:not(.preview-frame--split) .preview-pane {
  width: 100%;
  min-height: 26rem;
  border: none;
  border-radius: 0;
  background: transparent;
  padding: 0;
}

/* ── Split mode ── */
.preview-frame--split {
  border: none;
  border-radius: 0;
  background: transparent;
  overflow: visible;
  gap: 0.75rem;
  align-items: stretch;
}

.preview-frame--split .preview-pane {
  flex: 1 1 0%;
  /* zero basis = equal 50/50, ignores image dimensions */
  min-width: 0;
  height: 26rem;
  /* fixed — both panes are identical */
  border: 1px solid var(--q-bar-border);
  border-radius: 20px;
  background: var(--q-bar-bg);
  overflow: hidden;
}

/* ── PrimeVue Image wrapper — must fill pane in both modes ── */
.preview-pane :deep(.p-image) {
  display: flex;
  width: 100%;
  height: 100%;
  align-items: center;
  justify-content: center;
}

/* Single-pane */
.preview-frame:not(.preview-frame--split) .preview-pane :deep(.preview-img) {
  width: 100%;
  height: auto;
  max-height: 26rem;
  object-fit: contain;
  display: block;
}

/* Split — both images are 384×384, fill the pane equally */
.preview-frame--split .preview-pane :deep(.preview-img) {
  width: 100%;
  height: 100%;
  max-height: none;
  object-fit: contain;
  display: block;
}

/* ── Shared pane base ── */
.preview-pane {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
}

.preview-pane__label {
  position: absolute;
  bottom: 0.6rem;
  left: 50%;
  transform: translateX(-50%);
  padding: 0.22rem 0.65rem;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.55);
  color: white;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  white-space: nowrap;
  backdrop-filter: blur(4px);
}

.preview-pane__label--mild {
  background: rgba(234, 179, 8, 0.85);
}

.preview-pane__label--degraded {
  background: rgba(249, 115, 22, 0.85);
}

.preview-pane__label--severe {
  background: rgba(239, 68, 68, 0.85);
}

/* ── Slide-in transition ── */
.slide-in-enter-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.slide-in-enter-from {
  opacity: 0;
  transform: translateX(12px);
}

.slide-in-enter-to {
  opacity: 1;
  transform: translateX(0);
}

/* ── Mobile ── */
@media (max-width: 640px) {
  .preview-frame--split {
    flex-direction: column;
  }

  .preview-frame--split .preview-pane {
    height: auto;
    min-height: 200px;
  }
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

.result-summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.result-summary-grid--single {
  grid-template-columns: 1fr;
}

.result-summary-card {
  position: relative;
  overflow: hidden;
}

.result-summary-card::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  /* Top, Right, Bottom, Left */
  width: 3px;
  border-radius: 999px;
}

.classify-shell--ar .result-summary-card::before {
  /* This moves the bar to the right (Right: 0, Left: auto) */
  inset: 0 0 0 auto;
}

.result-summary-card--clean::before {
  background: var(--q-teal);
}

.result-summary-card--noisy::before {
  background: #f59e0b;
}


.results-view-switch {
  display: flex;
  justify-content: center;
  margin: 1.5rem 0 1rem;
}

/* Container matched to .qnn-nav */
.results-view-switch :deep(.p-selectbutton) {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.3rem;
  border: 1px solid var(--q-bar-border);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.45);
}

.p-dark .results-view-switch :deep(.p-selectbutton) {
  background: rgba(255, 255, 255, 0.03);
}

/* Base Button State matched to .qnn-link */
.results-view-switch :deep(.p-togglebutton) {
  border: none !important;
  background: transparent !important;
  color: var(--q-muted) !important;
  border-radius: 999px !important;
  padding: 0.58rem 1.25rem;
  font-size: 0.88rem;
  font-weight: 500;
  box-shadow: none !important;
  transition: color 0.2s, background 0.2s, transform 0.2s, box-shadow 0.2s;
}

/* Hover state for INACTIVE buttons */
.results-view-switch :deep(.p-togglebutton:not(.p-highlight):not(.p-togglebutton-checked):hover) {
  color: var(--q-text) !important;
  background: var(--q-teal-soft) !important;
}

/* ACTIVE STATE: Bulletproof targeting for PV3 and PV4 */
.results-view-switch :deep(.p-togglebutton.p-highlight),
.results-view-switch :deep(.p-togglebutton.p-togglebutton-checked),
.results-view-switch :deep(.p-togglebutton[aria-pressed="true"]) {
  background: var(--q-teal-soft) !important;
  background-color: var(--q-teal-soft) !important;
  color: var(--q-text) !important;
  box-shadow: inset 0 0 0 1px rgba(42, 184, 184, 0.15) !important;
}

/* Kill the default PrimeVue interaction layers (the gray box culprits) */
.results-view-switch :deep(.p-togglebutton::before),
.results-view-switch :deep(.p-togglebutton::after) {
  display: none !important;
}

/* Make sure the inner label spans don't block the background */
.results-view-switch :deep(.p-togglebutton-content),
.results-view-switch :deep(.p-togglebutton-label) {
  background: transparent !important;
}

/* Focus ring */
.results-view-switch :deep(.p-togglebutton:focus-visible) {
  outline: none;
  box-shadow: 0 0 0 2px var(--q-bar-bg), 0 0 0 4px var(--q-teal) !important;
}

/* Mobile responsive */
@media (max-width: 640px) {
  .results-view-switch :deep(.p-selectbutton) {
    width: 100%;
    grid-auto-flow: row;
    border-radius: 22px;
  }

  .results-view-switch :deep(.p-togglebutton) {
    border-radius: 14px !important;
  }
}

.results-view-panel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.results-export-card {
  margin-top: 1rem;
}

.animate-fadein {
  animation: fadein 0.35s ease both;
}

@keyframes fadein {
  from {
    opacity: 0;
    transform: translateY(10px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
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

  .result-summary-grid {
    grid-template-columns: 1fr;
  }

  .results-view-switch :deep(.p-selectbutton) {
    width: 100%;
    grid-auto-flow: row;
  }

  .results-view-switch :deep(.p-togglebutton) {
    width: 100%;
    min-width: 0;
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
  width: 100%;
}

.chart-wrap--center {
  display: flex;
  justify-content: center;
}

.chart-wrap :deep(canvas) {
  width: 100% !important;
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

.classify-shell--ar :deep(.classify-results-table),
.classify-shell--ar :deep(.classify-results-table table) {
  direction: rtl;
}

.classify-shell--ar :deep(.classify-results-table th),
.classify-shell--ar :deep(.classify-results-table td) {
  text-align: right;
}

.classify-shell--ar :deep(.classify-results-table .p-column-header-content) {
  justify-content: flex-start;
}

.classify-shell--ar :deep(.classify-results-table td > .flex) {
  direction: rtl;
}

.classify-shell--ar :deep(.classify-results-table .text-right) {
  text-align: left;
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

.lang-ar .restored-banner {
  text-align: right;
  direction: rtl;
}

html[lang="ar"] .restored-banner {
  text-align: right;
  direction: rtl;
}

.results-view-panel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.noisy-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.noise-level-chip {
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  border: 1px solid rgba(245, 158, 11, 0.3);
  background: rgba(245, 158, 11, 0.1);
  color: #d97706;
  font-size: 0.72rem;
  font-weight: 700;
  font-family: monospace;
  letter-spacing: 0.04em;
  white-space: nowrap;
}

.p-dark .noise-level-chip {
  color: #fbbf24;
  border-color: rgba(245, 158, 11, 0.25);
  background: rgba(245, 158, 11, 0.08);
}

.classify-info-icon-img {
  color: white;
  opacity: 0.65;
  transition: opacity 0.15s ease;
}

.classify-info-icon-img:hover {
  opacity: 1;
}

.classify-info-icon {
  color: var(--q-teal);
  opacity: 0.65;
  transition: opacity 0.15s ease;
}

.classify-info-icon:hover {
  opacity: 1;
}

.classify-overconfidence-banner {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(245, 158, 11, 0.3);
  background: rgba(245, 158, 11, 0.07);
  color: #b45309;
  font-size: 0.8rem;
  line-height: 1.5;
}

.p-dark .classify-overconfidence-banner {
  color: #fbbf24;
  border-color: rgba(245, 158, 11, 0.25);
  background: rgba(245, 158, 11, 0.06);
}

.classify-shell--ar .classify-overconfidence-banner {
  direction: rtl;
  text-align: right;
}
</style>
