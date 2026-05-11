<script setup>
import { computed, onMounted, ref, nextTick, onErrorCaptured } from 'vue'
import { useI18n } from 'vue-i18n'
import deformationImage from '../assets/benchmark/deformation.jpg'
import depositionImage from '../assets/benchmark/deposition.jpg'
import disconnectImage from '../assets/benchmark/disconnect.jpg'
import misalignmentImage from '../assets/benchmark/misalignment.jpg'
import obstacleImage from '../assets/benchmark/obstacle.jpg'
import ruptureImage from '../assets/benchmark/rupture.jpg'
import { AlertCircle, Info } from 'lucide-vue-next'
import {
  Chart as ChartJS, Title, Tooltip, Legend, BarElement, CategoryScale,
  LinearScale, PointElement, LineElement, RadialLinearScale
} from 'chart.js'
import { Line, Radar, Scatter } from 'vue-chartjs'

ChartJS.register(
  CategoryScale, LinearScale, BarElement, PointElement,
  LineElement, RadialLinearScale, Title, Tooltip, Legend
)
import { useRouter } from 'vue-router'

const router = useRouter()
const { t, tm, locale } = useI18n({ useScope: 'global' })

const benchmarkData = ref(null)
const isLoading = ref(false)
const error = ref(null)
const selectedModelKey = ref('CNN')
const selectedNoiseType = ref('gaussian')

const datasetImageMap = {
  Deformation: deformationImage,
  Deposition: depositionImage,
  Disconnect: disconnectImage,
  Misalignment: misalignmentImage,
  Obstacle: obstacleImage,
  Rupture: ruptureImage,
}

const modelMeta = {
  CNN: { accent: 'benchmark-accent--blue' },
  QNN_CPU: { accent: 'benchmark-accent--teal' },
  QNN_GPU: { accent: 'benchmark-accent--violet' },
}

const fetchBenchmarkData = async () => {
  isLoading.value = true
  error.value = null
  try {
    const response = await fetch('/api/v1/benchmark')
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    benchmarkData.value = await response.json()
  } catch (err) {
    console.error('Failed to fetch benchmark results:', err)
    error.value = 'benchmark.states.error'
  } finally {
    isLoading.value = false
  }
}

onMounted(async () => {
  await fetchBenchmarkData()          // fetch data
  await nextTick()                    // wait for DOM update

  // Restore scroll position saved before refresh
  const saved = localStorage.getItem('scrollRestore')
  if (saved) {
    const { path, top } = JSON.parse(saved)
    if (path === router.currentRoute.value.fullPath) {
      window.scrollTo({ top, behavior: 'instant' })  // no type assertion
    }
  }
})

const evaluation = computed(() => benchmarkData.value?.clean_evaluation ?? {})
const latencyMap = computed(() => benchmarkData.value?.inference_latency_ms ?? {})
const config = computed(() => benchmarkData.value?.config ?? {})
const noiseRobustness = computed(() => benchmarkData.value?.noise_robustness ?? {})

const models = computed(() => {
  const order = ['CNN', 'QNN_CPU', 'QNN_GPU']
  return order.map((key) => {
    const model = evaluation.value?.[key] ?? {}
    return {
      key,
      name: t(`benchmark.models.${key}`),
      accent: modelMeta[key].accent,
      accuracy: Number(model?.accuracy ?? 0),
      f1: Number(model?.averages?.weighted?.f1 ?? 0),
      latency: Number(latencyMap.value?.[key] ?? 0),
      samples: Number(model?.n_samples ?? 0),
    }
  })
})

const datasetSummary = computed(() => ({
  testSetSize: config.value?.test_set_size ?? 0,
  imageResolution: config.value?.image_resolution ?? '-',
  classNames: config.value?.class_names ?? [],
  faultClasses: (config.value?.class_names ?? []).length,
  modelsEvaluated: models.value.length,
  batchSize: config.value?.batch_size ?? '-',
  trainingEpochs: config.value?.training_epochs ?? '-',
  nQubits: config.value?.n_qubits ?? '-',
  qDepth: config.value?.q_depth ?? '-',
  device: benchmarkData.value?.device ?? '-',
}))

const bestAccuracyKey = computed(() => models.value.length ? [...models.value].sort((a, b) => b.accuracy - a.accuracy)[0]?.key : null)
const fastestLatencyKey = computed(() => models.value.length ? [...models.value].sort((a, b) => a.latency - b.latency)[0]?.key : null)

const selectedModel = computed(() => evaluation.value?.[selectedModelKey.value] ?? {})
const selectedNoiseData = computed(() => noiseRobustness.value?.[selectedNoiseType.value] ?? [])
const classNames = computed(() => config.value?.class_names ?? [])
const localizedClassNames = computed(() => classNames.value.map(n => t(`classify.${n}`) || n))
const confusionMatrix = computed(() => selectedModel.value?.confusion_matrix ?? [])
const perClassMetrics = computed(() => selectedModel.value?.per_class ?? {})

const modelOptions = computed(() => models.value.map(m => ({ label: m.name, value: m.key })))
const noiseTypes = computed(() => Object.keys(noiseRobustness.value))
const noiseOptions = computed(() =>
  noiseTypes.value.map(key => ({ label: t(`benchmark.robustness.${key}`), value: key }))
)

const datasetSamples = computed(() => {
  const rows = tm('benchmark.dataset.samples')
  if (!Array.isArray(rows)) return []
  return rows.map(item => ({ ...item, image: datasetImageMap[item.key] }))
})

const datasetFacts = computed(() => {
  const rows = tm('benchmark.dataset.facts')
  return Array.isArray(rows) ? rows : []
})

const datasetSteps = computed(() => {
  const rows = tm('benchmark.dataset.pipeline_steps')
  return Array.isArray(rows) ? rows : []
})

const perClassRows = computed(() => {
  const rows = Object.entries(perClassMetrics.value).map(([className, metrics]) => ({
    className: t(`classify.${className}`) || className,
    precision: Number(metrics?.precision ?? 0),
    recall: Number(metrics?.recall ?? 0),
    f1: Number(metrics?.f1 ?? 0),
    support: Number(metrics?.support ?? 0),
  }));

  // Append macro‑average row if the data is available
  const macro = selectedModel.value?.averages?.macro;
  const totalSamples = selectedModel.value?.n_samples;
  if (macro && totalSamples !== undefined) {
    rows.push({
      className: t('benchmark.diagnostics.average') || 'Average',   // add translation key
      precision: Number(macro.precision ?? 0),
      recall: Number(macro.recall ?? 0),
      f1: Number(macro.f1 ?? 0),
      support: Number(totalSamples),
    });
  }
  return rows;
});

const robustnessRows = computed(() =>
  selectedNoiseData.value.map(row => ({
    level: row.level,
    CNN: Number(row.CNN ?? 0),
    QNN_CPU: Number(row.QNN_CPU ?? 0),
    QNN_GPU: Number(row.QNN_GPU ?? 0),
  }))
)

const configCards = computed(() => [
  { label: t('benchmark.summary.resolution'), value: datasetSummary.value.imageResolution },
  { label: t('benchmark.summary.batch_size'), value: datasetSummary.value.batchSize },
  { label: t('benchmark.summary.epochs'), value: datasetSummary.value.trainingEpochs },
  { label: t('benchmark.summary.qubits'), value: datasetSummary.value.nQubits },
  { label: t('benchmark.summary.q_depth'), value: datasetSummary.value.qDepth },
  { label: t('benchmark.summary.device'), value: datasetSummary.value.device },
])

const isArabic = computed(() => locale.value === 'AR')
const textDir = computed(() => isArabic.value ? 'rtl' : 'ltr')

const modelBadgeKey = (modelKey) => {
  if (modelKey === bestAccuracyKey.value) return 'benchmark.performance.best_accuracy'
  if (modelKey === fastestLatencyKey.value) return 'benchmark.performance.fastest'
  return 'benchmark.performance.benchmark'
}

const formatPercent = (value, digits = 1) => `${Number(value || 0).toFixed(digits)}%`
const formatMetric = (value, digits = 2) => Number(value || 0).toFixed(digits)

const getModelColor = (key, alpha = 1) => {
  const colors = {
    CNN: `rgba(37, 99, 235, ${alpha})`,
    QNN_CPU: `rgba(13, 148, 136, ${alpha})`,
    QNN_GPU: `rgba(124, 58, 237, ${alpha})`,
  }
  return colors[key] || '#94a3b8'
}

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'inherit' } } }
  },
  scales: {
    y: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94a3b8' } },
    x: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94a3b8' } },
  },
}

const radarOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'inherit' } } }
  },
  scales: {
    r: {
      min: 75,
      max: 100,
      grid: { color: 'rgba(148, 163, 184, 0.15)' },
      ticks: {
        color: '#94a3b8',
        backdropColor: 'transparent',
        stepSize: 5,
      },
      pointLabels: { color: '#94a3b8', font: { size: 12 } },
    },
  },
}

const _baseScatterOptions = {
  ...chartOptions,
  scales: {
    x: { ...chartOptions.scales.x, title: { display: true, text: 'Latency (ms)', color: '#94a3b8' } },
    y: { ...chartOptions.scales.y, title: { display: true, text: 'Accuracy (%)', color: '#94a3b8' } },
  },
  plugins: {
    ...chartOptions.plugins,
    tooltip: {
      callbacks: {
        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}% @ ${ctx.parsed.x.toFixed(2)} ms`,
      },
    },
  },
}

const robustnessChartOptions = computed(() => {
  if (!benchmarkData.value?.extended_robustness) return chartOptions
  const rows = benchmarkData.value.extended_robustness.filter(d => d.noise_type === selectedNoiseType.value)
  const vals = rows.flatMap(d => [d.CNN_accuracy, d.QNN_CPU_accuracy, d.QNN_GPU_accuracy]).filter(v => v != null)
  if (!vals.length) return chartOptions
  return {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: { ...chartOptions.scales.y, min: Math.max(0, Math.floor(Math.min(...vals)) - 3), max: Math.min(100, Math.ceil(Math.max(...vals)) + 1) },
    },
  }
})

// Tighten scatter axes dynamically around actual values
const scatterOptions = computed(() => {
  if (!benchmarkData.value?.latency_data) return _baseScatterOptions
  const lats = benchmarkData.value.latency_data.map(d => d.avg_latency_ms)
  const accs = benchmarkData.value.latency_data.map(d =>
    benchmarkData.value.clean_evaluation?.[d.model]?.accuracy ?? 0)
  return {
    ..._baseScatterOptions,
    scales: {
      x: { ..._baseScatterOptions.scales.x, min: Math.floor(Math.min(...lats) * 0.7), max: Math.ceil(Math.max(...lats) * 1.3) },
      y: { ..._baseScatterOptions.scales.y, min: Math.floor(Math.min(...accs)) - 2, max: Math.min(100, Math.ceil(Math.max(...accs)) + 1) },
    },
  }
})

const radarChartData = computed(() => {
  if (!benchmarkData.value?.maun_summary) return { labels: [], datasets: [] }

  const metrics = ['accuracy', 'f1', 'precision', 'recall', 'overall_maun']
  const modelKeys = ['CNN', 'QNN_CPU', 'QNN_GPU']

  return {
    labels: metrics.map(m => t(`benchmark.metrics.${m}`)),
    datasets: modelKeys.map(m => {
      const evalData = benchmarkData.value.clean_evaluation?.[m]
      if (!evalData) return null
      const maunData = benchmarkData.value.maun_summary.find(s => s.model === m)
      return {
        label: m,
        data: [
          evalData.accuracy ?? 0,
          (evalData.averages?.weighted?.f1 ?? 0),
          (evalData.averages?.weighted?.precision ?? 0),
          (evalData.averages?.weighted?.recall ?? 0),
          maunData?.overall_maun ?? 0,
        ],
        borderColor: getModelColor(m),
        backgroundColor: getModelColor(m, 0.2),
      }
    }).filter(Boolean),
  }
})
const scatterChartData = computed(() => {
  if (!benchmarkData.value?.latency_data) return { datasets: [] }

  return {
    datasets: benchmarkData.value.latency_data.map(item => ({
      label: item.model,
      data: [{
        x: item.avg_latency_ms,
        y: benchmarkData.value.clean_evaluation?.[item.model]?.accuracy ?? 0,
      }],
      backgroundColor: getModelColor(item.model),
      pointRadius: 10,
      hoverRadius: 12,
    })),
  }
})

const robustnessChartData = computed(() => {
  if (!benchmarkData.value?.extended_robustness) return { labels: [], datasets: [] }
  const data = benchmarkData.value.extended_robustness.filter(d => d.noise_type === selectedNoiseType.value)
  return {
    labels: data.map(d => d.level),
    datasets: [
      { label: 'CNN', data: data.map(d => d.CNN_accuracy), borderColor: getModelColor('CNN'), tension: 0.3, fill: false },
      { label: 'QNN CPU', data: data.map(d => d.QNN_CPU_accuracy), borderColor: getModelColor('QNN_CPU'), tension: 0.3, fill: false },
      { label: 'QNN GPU', data: data.map(d => d.QNN_GPU_accuracy), borderColor: getModelColor('QNN_GPU'), tension: 0.3, fill: false },
    ],
  }
})

const reliabilityChartData = computed(() => {
  if (!benchmarkData.value?.extended_robustness) return { labels: [], datasets: [] }
  const data = benchmarkData.value.extended_robustness.filter(d => d.noise_type === selectedNoiseType.value)
  const m = selectedModelKey.value
  return {
    labels: data.map(d => d.level),
    datasets: [
      {
        label: t('benchmark.charts.actual_acc_label'),
        data: data.map(d => d[`${m}_accuracy`]),
        borderColor: getModelColor(m),
        fill: false,
      },
      {
        label: t('benchmark.charts.confidence_label'),
        data: data.map(d => d[`${m}_mean_conf`]),
        borderColor: '#94a3b8',
        borderDash: [5, 5],
        fill: false,
      },
    ],
  }
})


onErrorCaptured((err) => {
  console.error('[BenchmarkView] caught error:', err)
  error.value = 'benchmark.states.error' // ← always an i18n key, never a raw message
  isLoading.value = false
  return false
})
</script>

<template>
  <div class="benchmark-page mx-auto px-4 pb-20 pt-6 md:px-8 xl:px-14" :class="{ 'benchmark-page--ar': isArabic }"
    :dir="textDir">
    <!-- ── Loading ── -->
    <div v-if="isLoading" class="benchmark-state">
      <Card class="glass-card">
        <template #content>
          <div class="state-stack">
            <ProgressSpinner stroke-width="4" class="benchmark-spinner" />
            <p class="section-text">
              {{ t('benchmark.states.loading') }}
            </p>
            <div class="skeleton-grid">
              <Skeleton v-for="item in 3" :key="item" height="11rem" border-radius="24px" />
            </div>
          </div>
        </template>
      </Card>
    </div>

    <!-- Error  -->
    <div v-else-if="error" class="px-4 mx-auto max-w-screen-2xl sm:px-6 lg:px-8">
      <div class="glass-card flex flex-col items-center p-6 text-center">
        <AlertCircle class="w-12 h-12 mb-3 text-red-500" />
        <!-- t() gracefully returns the key if it's not a translation key -->
        <h3 class="text-lg font-medium text-red-500">
          {{ t(error) }}
        </h3>
      </div>
    </div>

    <template v-else-if="benchmarkData">
      <!-- ══════════════════════════════════════════
           HERO
      ══════════════════════════════════════════ -->
      <section class="hero-shell">
        <!-- Left copy -->
        <div class="hero-copy">
          <h1 class="hero-title" :dir="textDir">
            {{ t('benchmark.hero.title') }}
          </h1>
          <h2 class="hero-subtitle" :dir="textDir">
            {{ t('benchmark.hero.subtitle') }}
          </h2>
          <p class="hero-description">
            {{ t('benchmark.hero.description') }}
          </p>

          <div class="hero-stats">
            <div class="hero-stat-card">
              <span class="hero-stat-value">{{ datasetSummary.testSetSize }}</span>
              <span class="hero-stat-label">{{ t('benchmark.hero.test_samples') }}</span>
            </div>
            <div class="hero-stat-card">
              <span class="hero-stat-value">{{ datasetSummary.faultClasses }}</span>
              <span class="hero-stat-label">{{ t('benchmark.hero.fault_classes') }}</span>
            </div>
            <div class="hero-stat-card">
              <span class="hero-stat-value">{{ datasetSummary.modelsEvaluated }}</span>
              <span class="hero-stat-label">{{ t('benchmark.hero.models_evaluated') }}</span>
            </div>
          </div>
        </div>

        <Card class="glass-card hero-panel">
          <template #content>
            <div class="panel-topline">
              <span class="eyebrow">{{ t('benchmark.hero.accuracy_overview') }}</span>
              <Tag :value="t('benchmark.hero.loaded')" severity="success" rounded />
            </div>

            <div class="hero-metric-list">
              <div v-for="model in models" :key="model.key" class="hero-metric-row">
                <div class="hero-metric-head">
                  <div class="hero-metric-title">
                    <span class="model-dot" :class="model.accent" />
                    <span>{{ model.name }}</span>
                  </div>
                  <strong>{{ formatPercent(model.accuracy, 2) }}</strong>
                </div>
                <ProgressBar :value="model.accuracy" :show-value="false" :class="['metric-progress', model.accent]" />
              </div>
            </div>
          </template>
        </Card>
      </section>

      <!-- ══════════════════════════════════════════
           DATASET
      ══════════════════════════════════════════ -->
      <section class="dataset-shell">
        <Card class="glass-card dataset-card">
          <template #content>
            <div class="section-heading dataset-heading" :class="{ 'section-heading--rtl': isArabic }">
              <div class="section-heading__text dataset-heading-text" :dir="textDir">
                <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">
                  {{ t('benchmark.dataset.eyebrow') }}
                </span>
                <h3 class="section-title dataset-title" :dir="textDir">
                  {{ t('benchmark.dataset.title') }}
                </h3>
              </div>
              <Tag :value="t('benchmark.dataset.carousel_badge')" rounded class="hero-badge" />
            </div>

            <!-- carousel + right metadata side by side -->
            <div class="dataset-layout">
              <!-- Left: carousel only -->
              <div class="dataset-left">
                <div class="dataset-carousel-wrap">
                  <Carousel :value="datasetSamples" :num-visible="1" :num-scroll="1" circular :autoplay-interval="4500"
                    class="dataset-carousel" dir="ltr">
                    <template #item="{ data }">
                      <div class="dataset-slide">
                        <img :src="data.image" :alt="data.title" class="dataset-image" loading="lazy" decoding="async">
                        <div class="dataset-overlay" :dir="textDir">
                          <Tag :value="data.label" rounded class="slide-tag" />
                          <h4 class="dataset-slide-title">
                            {{ data.title }}
                          </h4>
                          <p class="dataset-slide-text">
                            {{ data.text }}
                          </p>
                        </div>
                      </div>
                    </template>
                  </Carousel>
                </div>
              </div>

              <!-- Right: facts + notes + classes -->
              <div class="dataset-copy">
                <div class="fact-grid">
                  <div v-for="fact in datasetFacts" :key="fact.label" class="fact-card">
                    <span class="fact-label">{{ fact.label }}</span>
                    <span class="fact-value">{{ fact.value }}</span>
                  </div>
                </div>

                <div class="dataset-notes">
                  <p class="section-text dataset-note">
                    <strong>{{ t('benchmark.dataset.transform_title') }}</strong>
                    {{ t('benchmark.dataset.transform_text') }}
                  </p>
                  <p class="section-text dataset-note">
                    <strong>{{ t('benchmark.dataset.source_label') }}</strong>
                    <a :href="t('benchmark.dataset.source_url')" target="_blank" rel="noreferrer" class="dataset-link">
                      {{ t('benchmark.dataset.source_name') }}
                    </a>
                  </p>
                </div>

                <div>
                  <span class="classes-heading">{{ t('benchmark.dataset.fault_classes', 'Fault Classes') }}</span>
                  <div class="classes-grid">
                    <span v-for="item in localizedClassNames" :key="item" class="class-chip">{{ item }}</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="dataset-steps-card">
              <span class="steps-title">{{ t('benchmark.dataset.pipeline_title') }}</span>
              <ol class="dataset-steps">
                <li v-for="step in datasetSteps" :key="step">
                  {{ step }}
                </li>
              </ol>
            </div>
          </template>
        </Card>
      </section>

      <!-- ══════════════════════════════════════════
           PERFORMANCE
      ══════════════════════════════════════════ -->
      <section class="content-section">
        <Card class="glass-card">
          <template #content>
            <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
              <div class="section-heading__text" :dir="textDir">
                <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.performance.eyebrow')
                  }}</span>
                <h3 class="section-title" :dir="textDir">
                  {{ t('benchmark.performance.title') }}
                </h3>
                <p class="section-text-nowrap">
                  {{ t('benchmark.performance.description') }}
                </p>
              </div>
            </div>

            <div class="performance-grid">
              <div v-for="model in models" :key="model.key" class="performance-inner-card">
                <div class="performance-card-head">
                  <div class="hero-metric-title">
                    <span class="model-dot" :class="model.accent" />
                    <span>{{ model.name }}</span>
                  </div>
                  <Tag :value="t(modelBadgeKey(model.key))" rounded />
                </div>

                <div class="kpi-value">
                  {{ formatPercent(model.accuracy, 2) }}
                </div>
                <p class="metric-caption">
                  {{ t('benchmark.performance.clean_accuracy') }}
                </p>

                <div class="metric-block">
                  <div class="metric-row">
                    <span>{{ t('benchmark.performance.latency') }}</span>
                    <strong>{{ formatMetric(model.latency, 3) }} ms</strong>
                  </div>
                  <div class="metric-line">
                    <span>{{ t('benchmark.performance.weighted_f1') }}</span>
                    <strong>{{ formatMetric(model.f1, 2) }}</strong>
                  </div>
                  <ProgressBar :value="model.f1" :show-value="false" :class="['metric-progress', model.accent]" />
                </div>
              </div>
            </div>
            <div class="chart-pair-grid">
              <div class="chart-card">
                <div class="flex items-center gap-2">
                  <span class="chart-card-title">{{ t('benchmark.charts.radar_title') }}</span>
                  <Info v-tooltip.top="t('benchmark.charts.radar_tooltip')" tabindex="0"
                    class="bm-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
                </div>
                <p class="chart-card-sub">
                  {{ t('benchmark.charts.radar_sub') }}
                </p>
                <div class="chart-shell chart-shell--lg" role="img" :aria-label="t('benchmark.charts.radar_title')">
                  <Radar :data="radarChartData" :options="radarOptions" />
                </div>
              </div>

              <div class="chart-card">
                <div class="flex items-center gap-2">
                  <span class="chart-card-title">{{ t('benchmark.charts.scatter_title') }}</span>
                  <Info v-tooltip.top="t('benchmark.charts.scatter_tooltip')" tabindex="0"
                    class="bm-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
                </div>
                <p class="chart-card-sub">
                  {{ t('benchmark.charts.scatter_sub') }}
                </p>
                <div class="chart-shell chart-shell--lg" role="img" :aria-label="t('benchmark.charts.scatter_title')">
                  <Scatter :data="scatterChartData" :options="scatterOptions" />
                </div>
              </div>
            </div>
          </template>
        </Card>
      </section>

      <!-- ══════════════════════════════════════════
           NOISE ROBUSTNESS
      ══════════════════════════════════════════ -->
      <section class="content-section">
        <Card class="glass-card">
          <template #content>
            <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
              <div class="section-heading__text" :dir="textDir">
                <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.robustness.eyebrow')
                  }}</span>
                <h3 class="section-title" :dir="textDir">
                  {{ t('benchmark.robustness.title') }}
                </h3>
                <p class="section-text-nowrap">
                  {{ t('benchmark.robustness.description') }}
                </p>
              </div>
            </div>

            <div class="flex flex-wrap gap-2 mb-4" role="tablist">
              <button v-for="opt in noiseOptions" :key="opt.value" role="tab"
                :aria-selected="selectedNoiseType === opt.value"
                class="px-3 py-1.5 rounded-lg text-sm transition-colors"
                :class="selectedNoiseType === opt.value ? 'bm-tab-active' : 'bm-tab-inactive'"
                @click="selectedNoiseType = opt.value">
                {{ opt.label }}
              </button>
            </div>
            <Card class="q-glass bm-inner-card">
              <template #content>
                <DataTable :value="robustnessRows" responsive-layout="scroll" class="bm-datatable-robustness"
                  :class="isArabic ? 'bm-table-rtl' : ''">
                  <Column field="level" :header="t('benchmark.robustness.level')" />
                  <Column field="CNN" :header="t('benchmark.models.CNN')">
                    <template #body="{ data }">
                      <span class="font-mono" style="color: #2563eb">{{ formatPercent(data.CNN, 2) }}</span>
                    </template>
                  </Column>
                  <Column field="QNN_CPU" :header="t('benchmark.models.QNN_CPU')">
                    <template #body="{ data }">
                      <span class="font-mono" style="color: #0d9488">{{ formatPercent(data.QNN_CPU, 2) }}</span>
                    </template>
                  </Column>
                  <Column field="QNN_GPU" :header="t('benchmark.models.QNN_GPU')">
                    <template #body="{ data }">
                      <span class="font-mono" style="color: #7c3aed">{{ formatPercent(data.QNN_GPU, 2) }}</span>
                    </template>
                  </Column>
                </DataTable>
              </template>
            </Card>
            <div class="chart-card" style="margin-top: 1rem;">
              <div class="flex items-center gap-2">
                <span class="chart-card-title">{{ t('benchmark.charts.robustness_title') }}</span>
                <Info v-tooltip.top="t('benchmark.charts.robustness_tooltip')" tabindex="0"
                  class="bm-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
              <p class="chart-card-sub">
                {{ t('benchmark.charts.robustness_sub') }}
              </p>
              <div class="chart-shell chart-shell--md" :aria-label="t('benchmark.charts.robustness_title')">
                <Line :data="robustnessChartData" :options="robustnessChartOptions" />
              </div>
            </div>
          </template>
        </Card>
      </section>

      <!-- ══════════════════════════════════════════
           DIAGNOSTICS
      ══════════════════════════════════════════ -->
      <section class="content-section">
        <Card class="glass-card">
          <template #content>
            <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
              <div class="section-heading__text" :dir="textDir">
                <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.diagnostics.eyebrow')
                  }}</span>
                <h3 class="section-title" :dir="textDir">
                  {{ t('benchmark.diagnostics.title') }}
                </h3>
                <p class="section-text">
                  {{ t('benchmark.diagnostics.description') }}
                </p>
              </div>
            </div>

            <div class="flex flex-wrap gap-2 mb-4">
              <button v-for="opt in modelOptions" :key="opt.value" role="tab"
                :aria-selected="selectedModelKey === opt.value" class="px-3 py-1.5 rounded-lg text-sm transition-colors"
                :class="selectedModelKey === opt.value ? 'bm-tab-active' : 'bm-tab-inactive'"
                @click="selectedModelKey = opt.value">
                {{ opt.label }}
              </button>
            </div>

            <div class="diagnostics-grid">
              <!-- Confusion matrix -->
              <div class="diag-inner-card">
                <span class="diag-title">{{ t('benchmark.diagnostics.confusion_matrix') }}</span>
                <div class="matrix-wrap">
                  <table class="matrix-table">
                    <thead>
                      <tr>
                        <th scope="col">{{ t('benchmark.diagnostics.actual') }}</th>
                        <th v-for="name in localizedClassNames" :key="`head-${name}`" scope="col">
                          {{ name }}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(row, rowIndex) in confusionMatrix" :key="`row-${rowIndex}`">
                        <td class="matrix-axis">
                          {{ localizedClassNames[rowIndex] || rowIndex }}
                        </td>
                        <td v-for="(value, colIndex) in row" :key="`cell-${rowIndex}-${colIndex}`" scope="row"
                          :class="{ 'matrix-cell--diag': rowIndex === colIndex }">
                          {{ value }}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div class="diag-inner-card">
                <span class="diag-title">{{ t('benchmark.diagnostics.per_class_metrics') }}</span>
                <Card class="q-glass bm-inner-card bm-fill-card">
                  <template #content>
                    <DataTable :value="perClassRows" responsive-layout="scroll" class="bm-datatable-diag"
                      :class="isArabic ? 'bm-table-rtl' : ''">
                      <Column field="className" :header="t('benchmark.diagnostics.class')" />
                      <Column field="precision" :header="t('benchmark.diagnostics.precision')">
                        <template #body="{ data }">
                          <span class="font-mono">{{ formatMetric(data.precision, 2)
                            }}</span>
                        </template>
                      </Column>
                      <Column field="recall" :header="t('benchmark.diagnostics.recall')">
                        <template #body="{ data }">
                          <span class="font-mono">{{ formatMetric(data.recall, 2)
                            }}</span>
                        </template>
                      </Column>
                      <Column field="f1" :header="t('benchmark.diagnostics.f1')">
                        <template #body="{ data }">
                          <span class="font-mono" style="color: var(--q-teal)">{{
                            formatMetric(data.f1, 2) }}</span>
                        </template>
                      </Column>
                      <Column field="support" :header="t('benchmark.diagnostics.support')" />
                    </DataTable>
                  </template>
                </Card>
              </div>
            </div>
            <div class="diag-inner-card diag-inner-card--full" style="margin-top: 1rem;">
              <div class="flex items-center gap-2">
                <span class="diag-title">{{ t('benchmark.charts.reliability_title') }}</span>
                <Info v-tooltip.top="t('benchmark.charts.reliability_tooltip')" tabindex="0"
                  class="bm-info-icon w-3.5 h-3.5 cursor-help flex-shrink-0" />
              </div>
              <p class="chart-card-sub" style="margin: -0.5rem 0 1rem;">
                {{ t('benchmark.charts.reliability_sub') }}
              </p>
              <div class="chart-shell chart-shell--md" role="img" :aria-label="t('benchmark.charts.reliability_title')">
                <Line :data="reliabilityChartData" :options="chartOptions" />
              </div>
            </div>
          </template>
        </Card>
      </section>

      <!-- ══════════════════════════════════════════
           CONFIGURATION
      ══════════════════════════════════════════ -->
      <section class="content-section">
        <Card class="glass-card">
          <template #content>
            <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
              <div class="section-heading__text" :dir="textDir">
                <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.config.eyebrow') }}</span>
                <h3 class="section-title" :dir="textDir">
                  {{ t('benchmark.config.title') }}
                </h3>
              </div>
            </div>

            <div class="config-grid">
              <div v-for="item in configCards" :key="item.label" class="config-card-item">
                <span class="config-label">{{ item.label }}</span>
                <strong class="config-value">{{ item.value }}</strong>
              </div>
            </div>
          </template>
        </Card>
      </section>
    </template>

    <div v-else class="benchmark-state">
      <Message severity="warn" :closable="false">
        {{ t('benchmark.states.empty') }}
      </Message>
    </div>
  </div>
</template>

<style scoped>
/* ══════════════════════════════════════════════
   PAGE SHELL
══════════════════════════════════════════════ */
.benchmark-page {
  color: var(--q-text);
  overflow-x: clip;
}

/* ══════════════════════════════════════════════
   GLASS CARD
══════════════════════════════════════════════ */
.glass-card {
  border: 2px solid var(--q-bar-border);
  background: rgba(255, 255, 255, 0.64);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: var(--q-shadow);
  border-radius: 28px;
}

.p-dark .glass-card {
  background: rgba(14, 28, 41, 0.74);
}

.hero-shell {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(300px, 0.75fr);
  gap: 1.5rem;
  align-items: start;
  /* was stretch */
}

.hero-copy {
  padding: 1.25rem 0;
}

.hero-badge {
  margin-bottom: 1rem;
  background: var(--q-teal-soft);
  color: var(--q-teal);
  border: 1px solid rgba(42, 184, 184, 0.18);
}

.hero-title {

  margin: 0 0 0.7rem;
  color: var(--q-text);
  font-family: var(--q-font-display);
  font-size: clamp(2.3rem, 4vw, 3.6rem);
  line-height: 1.05;
}

.hero-subtitle {
  font-size: clamp(1.1rem, 2vw, 1.5rem);
  font-weight: 700;
  color: var(--q-teal);
  margin: 0 0 1rem;
}

.hero-description,
.section-text,
.dataset-slide-text,
.metric-caption,
.control-label {
  color: var(--q-muted);
  line-height: 1.8;

}

.section-text-nowrap {
  color: var(--q-muted);
  line-height: 1.8;
  white-space: nowrap;

}

@media (max-width: 1080px) {
  .section-text-nowrap {
    white-space: normal;
  }
}

.hero-stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 2rem;
}

.hero-stat-card {
  padding: 1rem;
  border-radius: 18px;
  background: var(--q-teal-soft);
}

.hero-stat-value {
  display: block;
  font-size: 1.4rem;
  font-weight: 800;
  color: var(--q-text);
  font-family: var(--q-font-display);
}

.hero-stat-label {
  display: block;
  margin-top: 0.35rem;
  color: var(--q-muted);
}

.hero-panel {
  margin-bottom: 15px;
  align-self: end;
}

.panel-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.hero-metric-list {
  display: grid;
  gap: 1rem;
}

.hero-metric-row {
  display: grid;
  gap: 0.55rem;
}

.hero-metric-head,
.hero-metric-title,
.metric-line,
.metric-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}

.hero-metric-title {
  justify-content: flex-start;
  font-weight: 700;
  color: var(--q-text);
}

/* ══════════════════════════════════════════════
   MODEL DOT / PROGRESS COLORS
══════════════════════════════════════════════ */
.model-dot {
  width: 0.8rem;
  height: 0.8rem;
  border-radius: 999px;
  display: inline-block;
  flex-shrink: 0;
}

.benchmark-accent--blue {
  background: #2563eb;
}

.benchmark-accent--teal {
  background: #0d9488;
}

.benchmark-accent--violet {
  background: #7c3aed;
}

:deep(.metric-progress .p-progressbar) {
  height: 0.6rem;
  background: rgba(13, 31, 45, 0.08);
  border-radius: 999px;
}

.p-dark :deep(.metric-progress .p-progressbar) {
  background: rgba(255, 255, 255, 0.08);
}

:deep(.metric-progress.benchmark-accent--blue .p-progressbar-value) {
  background: #2563eb;
}

:deep(.metric-progress.benchmark-accent--teal .p-progressbar-value) {
  background: #0d9488;
}

:deep(.metric-progress.benchmark-accent--violet .p-progressbar-value) {
  background: #7c3aed;
}

/* ══════════════════════════════════════════════
   DATASET
══════════════════════════════════════════════ */
.dataset-shell,
.content-section {
  margin-top: 2rem;
}

.dataset-heading-text {
  max-width: none;
}

.dataset-title {
  font-size: clamp(1.4rem, 2.2vw, 2rem) !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* allow wrap at small breakpoints where nowrap would overflow */
@media (max-width: 1080px) {
  .dataset-title {
    white-space: normal;
  }
}

.dataset-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.9fr);
  gap: 1.25rem;
  align-items: start;
  margin-top: 1rem;
}

.dataset-left {
  display: flex;
  flex-direction: column;
}

.dataset-carousel-wrap {
  min-width: 0;
  direction: ltr;
}

:deep(.dataset-carousel .p-carousel),
:deep(.dataset-carousel .p-carousel-content),
:deep(.dataset-carousel .p-carousel-container),
:deep(.dataset-carousel .p-carousel-items-content) {
  direction: ltr;
  min-width: 0;
}

:deep(.dataset-carousel .p-carousel-indicator button) {
  background: rgba(42, 184, 184, 0.35) !important;
  border-radius: 999px;
  transition: background 0.2s;
}

:deep(.dataset-carousel .p-carousel-indicator.p-highlight button),
:deep(.dataset-carousel .p-carousel-indicator button:hover) {
  background: var(--q-teal) !important;
}

:deep(.dataset-carousel .p-carousel-item) {
  display: block;
}

.dataset-slide {
  position: relative;
  overflow: hidden;
  border-radius: 24px;
  min-height: 420px;
  border: 2px solid var(--q-bar-border);
}

.dataset-image {
  width: 100%;
  height: 420px;
  object-fit: cover;
  display: block;
}

.dataset-overlay {
  position: absolute;
  inset-inline: 0;
  bottom: 0;
  padding: 1.25rem;
  background: linear-gradient(180deg, rgba(8, 19, 29, 0.05), rgba(8, 19, 29, 0.85));
  color: white;
}

.slide-tag {
  background: var(--q-teal) !important;
  color: white !important;
  border-color: var(--q-teal) !important;
}

.dataset-slide-title {
  margin: 0.85rem 0 0.4rem;
  font-size: 1.35rem;
  font-weight: 800;
  font-family: var(--q-font-display);
}

.dataset-slide-text {
  color: rgba(255, 255, 255, 0.86);
  margin: 0;
}

/* Right column */
.dataset-copy {
  display: grid;
  gap: 1rem;
}

.fact-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
}

.fact-card {
  padding: 0.9rem 1rem;
  border-radius: 16px;
  background: var(--q-teal-soft);
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.fact-value {
  display: block;
  font-size: 1.3rem;
  font-weight: 800;
  color: var(--q-text);
  font-family: var(--q-font-display);
}

.fact-label {
  display: block;
  color: var(--q-muted);
  font-size: 0.82rem;
}

.dataset-notes {
  display: grid;
  gap: 0.75rem;
}

.dataset-note {
  margin: 0;
}

.dataset-note strong {
  color: var(--q-text);
  margin-inline-end: 0.4rem;
}

.dataset-link {
  color: var(--q-teal);
  text-decoration: none;
  font-weight: 700;
}

.dataset-link:hover {
  text-decoration: underline;
}

.classes-heading {
  display: block;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--q-muted);
  margin-bottom: 0.6rem;
}

.classes-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.class-chip {
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  border: 1px solid rgba(42, 184, 184, 0.3);
  background: var(--q-teal-soft);
  color: var(--q-teal);
  font-size: 0.82rem;
  font-weight: 600;
}

.dataset-steps-card {
  margin-top: 1.25rem;
  padding: 1rem 1.5rem;
  border: 2px solid var(--q-bar-border);
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.42);
}

.p-dark .dataset-steps-card {
  background: rgba(12, 24, 36, 0.82);
}

.steps-title {
  display: block;
  font-weight: 700;
  font-size: 0.9rem;
  color: var(--q-text);
  margin-bottom: 0.75rem;
}

/* Horizontal flow — looks natural across the full width */
.dataset-steps {
  margin: 0;
  padding-inline-start: 1.25rem;
  /* indent the numbers */
  color: var(--q-muted);
  line-height: 1.75;
  font-size: 0.9rem;
  list-style-type: square;
  /* ensures numbers (1,2,3…) */
  /* list-style-position: outside;  (default) */
}

/* ══════════════════════════════════════════════
   SECTION HEADINGS
══════════════════════════════════════════════ */
.section-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}



.section-heading__text {
  max-width: 820px;
}

.eyebrow {
  display: inline-block;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--q-teal);
  margin-bottom: 0.5rem;
}

.eyebrow--ar {
  letter-spacing: 0;
  text-transform: none;
}

.section-title {
  font-size: clamp(1.8rem, 3vw, 2.8rem);
  font-weight: 800;
  color: var(--q-text);
  margin: 0;
  font-family: var(--q-font-display);
}

/* ══════════════════════════════════════════════
   PERFORMANCE — inner cards inside glass wrapper
══════════════════════════════════════════════ */
.performance-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
}

.performance-inner-card {
  padding: 1.25rem;
  border-radius: 20px;
  border: 2px solid var(--q-bar-border);
  background: var(--q-surface-strong);
  display: flex;
  flex-direction: column;
  gap: 0;
}

.kpi-value {
  display: block;
  font-size: clamp(2.3rem, 4vw, 3.4rem);
  font-weight: 800;
  line-height: 1;
  margin: 1rem 0 0.2rem;
  color: var(--q-text);
  font-family: var(--q-font-display);
}

.metric-caption {
  margin: 0 0 1.1rem;
}

.metric-block {
  display: grid;
  gap: 0.55rem;
}

.performance-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

/* ══════════════════════════════════════════════
   TAB BUTTONS
══════════════════════════════════════════════ */
.bm-tab-active {
  background: var(--q-teal);
  color: #fff;
}

.bm-tab-inactive {
  background: var(--q-surface-strong);
  color: var(--q-muted);
  border: 2px solid var(--q-bar-border);
}

.bm-tab-inactive:hover {
  background: var(--q-teal-soft);
  color: var(--q-teal);
}

/* ══════════════════════════════════════════════
   INNER CARD + DATATABLE BASE
══════════════════════════════════════════════ */
.bm-inner-card {
  border-radius: 16px !important;
}

.bm-inner-card :deep(.p-card-body) {
  padding: 0 !important;
}

.bm-fill-card {
  flex: 1;
}

:deep(.p-datatable) {
  background: transparent;
  border-radius: 16px;
  overflow: hidden;
}

/* Base — compact, used by default unless overridden below */
:deep(.p-datatable-thead > tr > th) {
  background: transparent !important;
  color: #94a3b8;
  border-bottom: 1px solid var(--q-bar-border) !important;
  padding: 0.875rem 1.25rem;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  text-align: left;
}

:deep(.p-datatable-tbody > tr > td) {
  background: transparent !important;
  color: var(--q-text);
  border-bottom: 1px solid var(--q-bar-border) !important;
  padding: 0.875rem 1.25rem;
  font-size: 0.9rem;
  text-align: left;
}

:deep(.p-datatable-tbody > tr:last-child > td) {
  border-bottom: none !important;
}

:deep(.p-datatable-tbody > tr:hover > td) {
  background: var(--q-surface-soft) !important;
  transition: background 0.15s ease;
}

.robustness-inner-card :deep(.p-card-body) {
  padding: 0.5rem 0.75rem !important;
}

/* Reduce DataTable row padding while keeping font sizes */
.bm-datatable-robustness :deep(.p-datatable-tbody > tr > td) {
  padding: 0.5rem 1rem;
  /* smaller vertical padding */
}

.bm-datatable-robustness :deep(.p-datatable-thead > tr > th) {
  padding: 0.6rem 1rem;
}

/* Diagnostics per-class: slightly more breathing room (table is wide enough) */
.bm-datatable-diag :deep(.p-datatable-thead > tr > th) {
  font-size: 0.85rem;
  padding: 1rem 1.25rem;
}

.bm-datatable-diag :deep(.p-datatable-tbody > tr > td) {
  font-size: 1rem;
  padding: 1rem 1.25rem;
}

/* RTL table alignment */
:deep(.bm-table-rtl .p-datatable-thead > tr > th) {
  text-align: right !important;
}

:deep(.bm-table-rtl .p-datatable-tbody > tr > td) {
  text-align: right !important;
}

/* ══════════════════════════════════════════════
   DIAGNOSTICS — inside glass wrapper
══════════════════════════════════════════════ */
.diagnostics-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1rem;
  align-items: stretch;
}

.diag-inner-card {
  border: 2px solid var(--q-bar-border);
  border-radius: 20px;
  background: var(--q-surface-strong);
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.diag-title {
  display: block;
  font-weight: 700;
  font-size: 1rem;
  color: var(--q-text);
  margin-bottom: 1rem;
  flex-shrink: 0;
}

.matrix-wrap {
  position: relative;
  flex: 1;
  /* take remaining vertical space */
  overflow: auto;
}

.matrix-table {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border-collapse: separate;
  border-spacing: 0.35rem;
  /* keep the rest of your table styling */
}

.matrix-table th,
.matrix-table td {
  padding: 0.75rem 0.6rem;
  text-align: center;
  border-radius: 14px;
  background: rgba(13, 31, 45, 0.05);
  color: var(--q-text);
  font-size: 0.9rem;
}

.p-dark .matrix-table th,
.p-dark .matrix-table td {
  background: rgba(255, 255, 255, 0.06);
}

.matrix-table th {
  font-size: 0.8rem;
}

.matrix-axis {
  font-weight: 700;
}

.matrix-cell--diag {
  background: var(--q-teal) !important;
  color: white !important;
  font-weight: 800;
}

/* ══════════════════════════════════════════════
   CONFIG GRID
══════════════════════════════════════════════ */
.config-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.config-card-item {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  padding: 1rem 1.25rem;
  border-radius: 18px;
  border: 2px solid var(--q-bar-border);
  background: var(--q-surface-strong);
}

.config-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--q-muted);
}

.config-value {
  font-size: 1.4rem;
  font-weight: 800;
  color: var(--q-text);
  font-family: var(--q-font-display);
}

/* ══════════════════════════════════════════════
   STATES
══════════════════════════════════════════════ */
.benchmark-state {
  margin-top: 1rem;
}

.state-stack {
  display: grid;
  gap: 1rem;
  text-align: center;
}

.benchmark-spinner {
  margin: 0 auto;
}

.skeleton-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
}


.benchmark-page--ar .hero-copy,
.benchmark-page--ar .section-heading__text,
.benchmark-page--ar .dataset-notes,
.benchmark-page--ar .dataset-steps,
.benchmark-page--ar .classes-grid {
  direction: rtl;
  text-align: right;
}

.benchmark-page--ar .hero-title,
.benchmark-page--ar .hero-subtitle,
.benchmark-page--ar .section-title {
  direction: rtl;
  text-align: right;
}

.benchmark-page--ar .config-grid,
.benchmark-page--ar .config-card-item {
  direction: rtl;
  text-align: right;
}

/* ══════════════════════════════════════════════
   RESPONSIVE
══════════════════════════════════════════════ */
@media (max-width: 1280px) {
  .config-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 1080px) {

  .hero-shell,
  .dataset-layout,
  .performance-grid,
  .diagnostics-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 820px) {

  .hero-stats,
  .fact-grid,
  .config-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {

  .hero-stats,
  .fact-grid,
  .config-grid {
    grid-template-columns: 1fr;
  }

  .dataset-image,
  .dataset-slide {
    min-height: 320px;
    height: 320px;
  }
}

.chart-shell {
  position: relative;
  /* required by Chart.js when maintainAspectRatio: false */
  width: 100%;
}

.chart-shell--sm {
  height: 260px;
}

.chart-shell--md {
  height: 320px;
}

.chart-shell--lg {
  height: 380px;
}

.chart-pair-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1.5rem;
}

.chart-card {
  border: 2px solid var(--q-bar-border);
  border-radius: 20px;
  background: var(--q-surface-strong);
  padding: 1.25rem;
}

.chart-card-title {
  display: block;
  font-weight: 700;
  font-size: 1rem;
  color: var(--q-text);
  margin-bottom: 0.25rem;
}

.chart-card-sub {
  color: var(--q-muted);
  font-size: 0.83rem;
  margin: 0 0 1rem;
  line-height: 1.5;
}

/* Full-width diagnostics card (reliability chart) */
.diag-inner-card--full {
  grid-column: 1 / -1;
}

/* Collapse chart-pair-grid on narrow screens */
@media (max-width: 1080px) {
  .chart-pair-grid {
    grid-template-columns: 1fr;
  }
}

.bm-info-icon {
  color: var(--q-teal);
  opacity: 0.7;
  transition: opacity 0.15s ease;
}

.bm-info-icon:hover {
  opacity: 1;
}

@media (max-width: 960px) {
  .diagnostics-grid {
    grid-template-columns: 1fr;
  }

  .diag-inner-card {
    overflow: hidden;
  }

  .matrix-wrap {
    overflow-x: auto;
    min-height: 0;
    -webkit-overflow-scrolling: touch;
  }

  .matrix-table {
    position: relative;
    width: max-content;
    min-width: 100%;
    height: auto;
  }

  .bm-fill-card {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
}

@media (max-width: 640px) {

  .matrix-table th,
  .matrix-table td {
    padding: 0.5rem 0.4rem;
    font-size: 0.75rem;
    border-radius: 8px;
  }

  .bm-datatable-diag :deep(.p-datatable-thead > tr > th),
  .bm-datatable-diag :deep(.p-datatable-tbody > tr > td) {
    padding: 0.6rem 0.75rem;
    font-size: 0.85rem;
  }
}

</style>