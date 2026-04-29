<script setup>
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import Carousel from 'primevue/carousel'
import SelectButton from 'primevue/selectbutton'
import deformationImage from '../assets/benchmark/deformation.jpg'
import depositionImage from '../assets/benchmark/deposition.jpg'
import disconnectImage from '../assets/benchmark/disconnect.jpg'
import misalignmentImage from '../assets/benchmark/misalignment.jpg'
import obstacleImage from '../assets/benchmark/obstacle.jpg'
import ruptureImage from '../assets/benchmark/rupture.jpg'

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

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    benchmarkData.value = await response.json()
  } catch (err) {
    console.error('Failed to fetch benchmark results:', err)
    error.value = 'benchmark.states.error'
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  fetchBenchmarkData()
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

const bestAccuracyKey = computed(() => {
  if (!models.value.length) return null
  return [...models.value].sort((a, b) => b.accuracy - a.accuracy)[0]?.key ?? null
})

const fastestLatencyKey = computed(() => {
  if (!models.value.length) return null
  return [...models.value].sort((a, b) => a.latency - b.latency)[0]?.key ?? null
})

const selectedModel = computed(() => evaluation.value?.[selectedModelKey.value] ?? {})
const selectedNoiseData = computed(() => noiseRobustness.value?.[selectedNoiseType.value] ?? [])
const classNames = computed(() => config.value?.class_names ?? [])
const localizedClassNames = computed(() => classNames.value.map((name) => t(`classify.${name}`) || name))
const confusionMatrix = computed(() => selectedModel.value?.confusion_matrix ?? [])
const perClassMetrics = computed(() => selectedModel.value?.per_class ?? {})

const modelOptions = computed(() =>
  models.value.map((model) => ({
    label: model.name,
    value: model.key,
  })),
)

const noiseOptions = computed(() =>
  Object.keys(noiseRobustness.value).map((key) => ({
    label: t(`benchmark.robustness.${key}`),
    value: key,
  })),
)

const selectedNoiseLabel = computed(() => {
  const option = noiseOptions.value.find((item) => item.value === selectedNoiseType.value)
  return option?.label ?? selectedNoiseType.value
})

const datasetSamples = computed(() => {
  const rows = tm('benchmark.dataset.samples')

  if (!Array.isArray(rows)) return []

  return rows.map((item) => ({
    ...item,
    image: datasetImageMap[item.key],
  }))
})

const datasetFacts = computed(() => {
  const rows = tm('benchmark.dataset.facts')
  return Array.isArray(rows) ? rows : []
})

const datasetSteps = computed(() => {
  const rows = tm('benchmark.dataset.pipeline_steps')
  return Array.isArray(rows) ? rows : []
})

const perClassRows = computed(() =>
  Object.entries(perClassMetrics.value).map(([className, metrics]) => ({
    className,
    precision: Number(metrics?.precision ?? 0),
    recall: Number(metrics?.recall ?? 0),
    f1: Number(metrics?.f1 ?? 0),
    support: Number(metrics?.support ?? 0),
  })),
)

const robustnessRows = computed(() =>
  selectedNoiseData.value.map((row) => ({
    level: row.level,
    CNN: Number(row.CNN ?? 0),
    QNN_CPU: Number(row.QNN_CPU ?? 0),
    QNN_GPU: Number(row.QNN_GPU ?? 0),
  })),
)

const overviewCards = computed(() => [
  { label: t('benchmark.summary.batch_size'), value: datasetSummary.value.batchSize },
  { label: t('benchmark.summary.epochs'), value: datasetSummary.value.trainingEpochs },
  { label: t('benchmark.summary.qubits'), value: datasetSummary.value.nQubits },
  { label: t('benchmark.summary.q_depth'), value: datasetSummary.value.qDepth },
  { label: t('benchmark.summary.resolution'), value: datasetSummary.value.imageResolution },
  { label: t('benchmark.summary.device'), value: datasetSummary.value.device },
])

const generatedAt = computed(() => benchmarkData.value?.generated_at?.slice(0, 10) || t('benchmark.hero.live'))
const isArabic = computed(() => locale.value === 'AR')

const modelBadgeKey = (modelKey) => {
  if (modelKey === bestAccuracyKey.value) return 'benchmark.performance.best_accuracy'
  if (modelKey === fastestLatencyKey.value) return 'benchmark.performance.fastest'
  return 'benchmark.performance.benchmark'
}

const formatPercent = (value, digits = 1) => `${Number(value || 0).toFixed(digits)}%`
const formatMetric = (value, digits = 2) => Number(value || 0).toFixed(digits)
</script>

<template>
  <div class="benchmark-page px-4 pb-20 pt-6 md:px-8 xl:px-14" :class="{ 'benchmark-page--ar': isArabic }">
    <div v-if="isLoading" class="benchmark-state">
      <Card class="glass-card">
        <template #content>
          <div class="state-stack">
            <ProgressSpinner strokeWidth="4" class="benchmark-spinner" />
            <p class="section-text">{{ t('benchmark.states.loading') }}</p>
            <div class="skeleton-grid">
              <Skeleton v-for="item in 3" :key="item" height="11rem" borderRadius="24px" />
            </div>
          </div>
        </template>
      </Card>
    </div>

    <div v-else-if="error" class="benchmark-state">
      <Message severity="error" :closable="false">{{ t(error) }}</Message>
    </div>

    <template v-else-if="benchmarkData">
      <section class="hero-shell">
        <div class="hero-copy">
          <Tag :value="`${t('benchmark.hero.badge')} · ${generatedAt}`" rounded class="hero-badge" />
          <h1 class="hero-title">{{ t('benchmark.hero.title') }}</h1>
          <h2 class="hero-subtitle">{{ t('benchmark.hero.subtitle') }}</h2>
          <p class="hero-description">{{ t('benchmark.hero.description') }}</p>

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
                    <span class="model-dot" :class="model.accent"></span>
                    <span>{{ model.name }}</span>
                  </div>
                  <strong>{{ formatPercent(model.accuracy, 2) }}</strong>
                </div>
                <ProgressBar :value="model.accuracy" :showValue="false" :class="['metric-progress', model.accent]" />
              </div>
            </div>

            <Divider />

            <div class="hero-panel-footer">
              <Chip :label="`${t('benchmark.summary.resolution')}: ${datasetSummary.imageResolution}`" />
              <Chip :label="`${t('benchmark.summary.device')}: ${datasetSummary.device}`" />
            </div>
          </template>
        </Card>
      </section>

      <section class="dataset-shell">
        <Card class="glass-card dataset-card">
          <template #content>
            <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
              <div class="section-heading__text" :dir="isArabic ? 'rtl' : 'ltr'">
                <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">
                  {{ t('benchmark.dataset.eyebrow') }}
                </span>
                <h3 class="section-title">{{ t('benchmark.dataset.title') }}</h3>
                <p class="section-text dataset-description">{{ t('benchmark.dataset.description') }}</p>
              </div>
              <Tag :value="t('benchmark.dataset.carousel_badge')" rounded class="hero-badge" />
            </div>

            <div class="dataset-layout">
              <div class="dataset-carousel-wrap">
                <Carousel
                  :value="datasetSamples"
                  :numVisible="1"
                  :numScroll="1"
                  circular
                  :autoplayInterval="4500"
                  class="dataset-carousel"
                  dir="ltr"
                >
                  <template #item="{ data }">
                    <div class="dataset-slide" :dir="isArabic ? 'rtl' : 'ltr'">
                      <img :src="data.image" :alt="data.title" class="dataset-image" />
                      <div class="dataset-overlay">
                        <Tag :value="data.label" rounded />
                        <h4 class="dataset-slide-title">{{ data.title }}</h4>
                        <p class="dataset-slide-text">{{ data.text }}</p>
                      </div>
                    </div>
                  </template>
                </Carousel>
              </div>

              <div class="dataset-copy">
                <div class="fact-grid">
                  <div v-for="fact in datasetFacts" :key="fact.label" class="fact-card">
                    <span class="fact-value">{{ fact.value }}</span>
                    <span class="fact-label">{{ fact.label }}</span>
                  </div>
                </div>

                <div class="dataset-notes">
                  <p class="section-text dataset-note">
                    <strong>{{ t('benchmark.dataset.transform_title') }}</strong>
                    {{ t('benchmark.dataset.transform_text') }}
                  </p>

                  <p class="section-text dataset-note">
                    <strong>{{ t('benchmark.dataset.source_label') }}</strong>
                    <a
                      :href="t('benchmark.dataset.source_url')"
                      target="_blank"
                      rel="noreferrer"
                      class="dataset-link"
                    >
                      {{ t('benchmark.dataset.source_name') }}
                    </a>
                  </p>
                </div>

                <ul class="dataset-list">
                  <li v-for="item in localizedClassNames" :key="item">{{ item }}</li>
                </ul>

                <Card class="dataset-steps-card">
                  <template #content>
                    <div class="table-title-row">
                      <span class="table-title">{{ t('benchmark.dataset.pipeline_title') }}</span>
                    </div>

                    <ol class="dataset-steps">
                      <li v-for="step in datasetSteps" :key="step">{{ step }}</li>
                    </ol>
                  </template>
                </Card>
              </div>
            </div>
          </template>
        </Card>
      </section>

      <section class="summary-grid">
        <Card v-for="item in overviewCards" :key="item.label" class="glass-card summary-card">
          <template #content>
            <span class="summary-label">{{ item.label }}</span>
            <strong class="summary-value">{{ item.value }}</strong>
          </template>
        </Card>
      </section>

      <section class="content-section">
        <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
          <div class="section-heading__text" :dir="isArabic ? 'rtl' : 'ltr'">
            <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.performance.eyebrow') }}</span>
            <h3 class="section-title">{{ t('benchmark.performance.title') }}</h3>
            <p class="section-text">{{ t('benchmark.performance.description') }}</p>
          </div>
        </div>

        <div class="performance-grid">
          <Card v-for="model in models" :key="model.key" class="glass-card performance-card">
            <template #content>
              <div class="performance-card-head">
                <div class="hero-metric-title">
                  <span class="model-dot" :class="model.accent"></span>
                  <span>{{ model.name }}</span>
                </div>
                <Tag :value="t(modelBadgeKey(model.key))" rounded />
              </div>

              <div class="kpi-value">{{ formatPercent(model.accuracy, 2) }}</div>
              <p class="metric-caption">{{ t('benchmark.performance.clean_accuracy') }}</p>

              <div class="metric-block">
                <div class="metric-line">
                  <span>{{ t('benchmark.performance.weighted_f1') }}</span>
                  <strong>{{ formatMetric(model.f1, 2) }}</strong>
                </div>
                <ProgressBar :value="model.f1" :showValue="false" :class="['metric-progress', model.accent]" />
              </div>

              <div class="metric-row">
                <span>{{ t('benchmark.performance.latency') }}</span>
                <strong>{{ formatMetric(model.latency, 3) }} ms</strong>
              </div>

              <div class="metric-row">
                <span>{{ t('benchmark.performance.samples') }}</span>
                <strong>{{ model.samples }}</strong>
              </div>
            </template>
          </Card>
        </div>
      </section>

      <section class="content-section">
        <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
          <div class="section-heading__text" :dir="isArabic ? 'rtl' : 'ltr'">
            <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.robustness.eyebrow') }}</span>
            <h3 class="section-title">{{ t('benchmark.robustness.title') }}</h3>
            <p class="section-text">{{ t('benchmark.robustness.description') }}</p>
          </div>
        </div>

        <Card class="glass-card">
          <template #content>
            <div class="controls-row">
              <label class="control-label">{{ t('benchmark.robustness.filter_label') }}</label>
              <SelectButton
                v-model="selectedNoiseType"
                :options="noiseOptions"
                optionLabel="label"
                optionValue="value"
                class="benchmark-select"
              />
            </div>

            <div class="table-title-row">
              <span class="table-title">{{ selectedNoiseLabel }}</span>
              <Tag :value="`${robustnessRows.length} ${t('benchmark.robustness.points')}`" rounded />
            </div>

            <DataTable :value="robustnessRows" responsiveLayout="scroll" class="benchmark-table">
              <Column field="level" :header="t('benchmark.robustness.level')" />
              <Column field="CNN" :header="t('benchmark.models.CNN')">
                <template #body="{ data }">{{ formatPercent(data.CNN, 2) }}</template>
              </Column>
              <Column field="QNN_CPU" :header="t('benchmark.models.QNN_CPU')">
                <template #body="{ data }">{{ formatPercent(data.QNN_CPU, 2) }}</template>
              </Column>
              <Column field="QNN_GPU" :header="t('benchmark.models.QNN_GPU')">
                <template #body="{ data }">{{ formatPercent(data.QNN_GPU, 2) }}</template>
              </Column>
            </DataTable>
          </template>
        </Card>
      </section>

      <section class="content-section">
        <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
          <div class="section-heading__text" :dir="isArabic ? 'rtl' : 'ltr'">
            <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.latency.eyebrow') }}</span>
            <h3 class="section-title">{{ t('benchmark.latency.title') }}</h3>
            <p class="section-text">{{ t('benchmark.latency.description') }}</p>
          </div>
        </div>

        <div class="latency-grid">
          <Card v-for="model in models" :key="`${model.key}-latency`" class="glass-card latency-card">
            <template #content>
              <div class="latency-card-head">
                <span class="hero-metric-title">
                  <span class="model-dot" :class="model.accent"></span>
                  <span>{{ model.name }}</span>
                </span>
                <strong>{{ formatMetric(model.latency, 3) }} ms</strong>
              </div>
              <ProgressBar :value="Math.max(1, 100 - model.latency)" :showValue="false" :class="['metric-progress', model.accent]" />
            </template>
          </Card>
        </div>
      </section>

      <section class="content-section">
        <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
          <div class="section-heading__text" :dir="isArabic ? 'rtl' : 'ltr'">
            <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.diagnostics.eyebrow') }}</span>
            <h3 class="section-title">{{ t('benchmark.diagnostics.title') }}</h3>
            <p class="section-text">{{ t('benchmark.diagnostics.description') }}</p>
          </div>
        </div>

        <div class="controls-row">
          <label class="control-label">{{ t('benchmark.diagnostics.model') }}</label>
          <SelectButton
            v-model="selectedModelKey"
            :options="modelOptions"
            optionLabel="label"
            optionValue="value"
            class="benchmark-select"
          />
        </div>

        <div class="diagnostics-grid">
          <Card class="glass-card">
            <template #content>
              <div class="table-title-row">
                <span class="table-title">{{ t('benchmark.diagnostics.confusion_matrix') }}</span>
              </div>

              <div class="matrix-wrap">
                <table class="matrix-table">
                  <thead>
                    <tr>
                      <th>{{ t('benchmark.diagnostics.actual') }}</th>
                      <th v-for="name in localizedClassNames" :key="`head-${name}`">{{ name }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, rowIndex) in confusionMatrix" :key="`row-${rowIndex}`">
                      <td class="matrix-axis">{{ localizedClassNames[rowIndex] || rowIndex }}</td>
                      <td
                        v-for="(value, colIndex) in row"
                        :key="`cell-${rowIndex}-${colIndex}`"
                        :class="{ 'matrix-cell--diag': rowIndex === colIndex }"
                      >
                        {{ value }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </template>
          </Card>

          <Card class="glass-card">
            <template #content>
              <div class="table-title-row">
                <span class="table-title">{{ t('benchmark.diagnostics.per_class_metrics') }}</span>
              </div>

              <DataTable :value="perClassRows" responsiveLayout="scroll" class="benchmark-table">
                <Column field="className" :header="t('benchmark.diagnostics.class')" />
                <Column field="precision" :header="t('benchmark.diagnostics.precision')">
                  <template #body="{ data }">{{ formatMetric(data.precision, 2) }}</template>
                </Column>
                <Column field="recall" :header="t('benchmark.diagnostics.recall')">
                  <template #body="{ data }">{{ formatMetric(data.recall, 2) }}</template>
                </Column>
                <Column field="f1" :header="t('benchmark.diagnostics.f1')">
                  <template #body="{ data }">{{ formatMetric(data.f1, 2) }}</template>
                </Column>
                <Column field="support" :header="t('benchmark.diagnostics.support')" />
              </DataTable>
            </template>
          </Card>
        </div>
      </section>

      <section class="content-section">
        <Card class="glass-card config-card">
          <template #content>
            <div class="section-heading" :class="{ 'section-heading--rtl': isArabic }">
              <div class="section-heading__text" :dir="isArabic ? 'rtl' : 'ltr'">
                <span class="eyebrow" :class="{ 'eyebrow--ar': isArabic }">{{ t('benchmark.config.eyebrow') }}</span>
                <h3 class="section-title">{{ t('benchmark.config.title') }}</h3>
              </div>
            </div>

            <pre class="config-block">{{ JSON.stringify(config, null, 2) }}</pre>
          </template>
        </Card>
      </section>
    </template>

    <div v-else class="benchmark-state">
      <Message severity="warn" :closable="false">{{ t('benchmark.states.empty') }}</Message>
    </div>
  </div>
</template>

<style scoped>
.benchmark-page {
  color: var(--q-text);
  overflow-x: clip;
}

.hero-shell {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
  gap: 1.5rem;
  align-items: stretch;
}

.hero-copy,
.hero-panel,
.dataset-card,
.summary-card,
.performance-card,
.latency-card,
.config-card {
  height: 100%;
}

.glass-card {
  border: 1px solid var(--q-bar-border);
  background: rgba(255, 255, 255, 0.64);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: var(--q-shadow);
  border-radius: 28px;
}

.p-dark .glass-card {
  background: rgba(14, 28, 41, 0.74);
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
  font-size: clamp(2.7rem, 5.5vw, 4.8rem);
  line-height: 0.95;
  font-weight: 800;
  color: var(--q-text);
  letter-spacing: -0.04em;
  margin: 0 0 0.8rem;
  font-family: var(--q-font-display);
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

.hero-stats,
.summary-grid,
.performance-grid,
.latency-grid,
.diagnostics-grid,
.fact-grid {
  display: grid;
  gap: 1rem;
}

.hero-stats {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-top: 2rem;
}

.hero-stat-card,
.fact-card {
  padding: 1rem;
  border-radius: 18px;
  background: var(--q-teal-soft);
}

.hero-stat-value,
.summary-value,
.fact-value,
.kpi-value {
  display: block;
  color: var(--q-text);
  font-family: var(--q-font-display);
  font-weight: 800;
}

.hero-stat-value,
.summary-value,
.fact-value {
  font-size: 1.4rem;
}

.hero-stat-label,
.summary-label,
.fact-label {
  display: block;
  margin-top: 0.35rem;
  color: var(--q-muted);
}

.panel-topline,
.performance-card-head,
.table-title-row,
.latency-card-head,
.controls-row,
.hero-panel-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.hero-panel-footer,
.controls-row {
  flex-wrap: wrap;
}

.hero-metric-list,
.metric-block {
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

.model-dot {
  width: 0.8rem;
  height: 0.8rem;
  border-radius: 999px;
  display: inline-block;
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

.dataset-shell,
.content-section {
  margin-top: 2rem;
}

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

.section-title {
  font-size: clamp(1.8rem, 3vw, 2.8rem);
  font-weight: 800;
  color: var(--q-text);
  margin: 0;
  font-family: var(--q-font-display);
}

.dataset-description {
  margin-top: 0.75rem;
}

.dataset-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
  gap: 1.25rem;
  align-items: start;
}

.dataset-carousel-wrap {
  min-width: 0;
  direction: ltr;
}

:deep(.dataset-carousel .p-carousel) {
  direction: ltr;
}

:deep(.dataset-carousel .p-carousel-content),
:deep(.dataset-carousel .p-carousel-container),
:deep(.dataset-carousel .p-carousel-items-content) {
  direction: ltr;
  min-width: 0;
}

:deep(.dataset-carousel .p-carousel-item) {
  display: block;
}

.dataset-slide {
  position: relative;
  overflow: hidden;
  border-radius: 24px;
  min-height: 420px;
  border: 1px solid var(--q-bar-border);
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

.dataset-copy {
  display: grid;
  gap: 1rem;
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

.fact-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.dataset-list {
  margin: 0;
  padding-inline-start: 1.1rem;
  color: var(--q-text);
  display: grid;
  gap: 0.55rem;
}

.dataset-steps-card {
  border: 1px solid var(--q-bar-border);
  background: rgba(255, 255, 255, 0.42);
  border-radius: 22px;
}

.p-dark .dataset-steps-card {
  background: rgba(12, 24, 36, 0.82);
}

.dataset-steps {
  margin: 0;
  padding-inline-start: 1.25rem;
  display: grid;
  gap: 0.75rem;
  color: var(--q-text);
  line-height: 1.75;
}

.summary-grid {
  grid-template-columns: repeat(6, minmax(0, 1fr));
  margin-top: 1.5rem;
}

.summary-card :deep(.p-card-content),
.performance-card :deep(.p-card-content),
.latency-card :deep(.p-card-content) {
  padding: 0;
}

.kpi-value {
  font-size: clamp(2.3rem, 4vw, 3.4rem);
  line-height: 1;
  margin: 1rem 0 0.2rem;
}

.metric-caption {
  margin: 0 0 1.1rem;
}

.performance-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.latency-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.diagnostics-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 1rem;
}

.table-title {
  font-weight: 800;
  color: var(--q-text);
  font-size: 1.05rem;
}

.matrix-wrap {
  overflow-x: auto;
}

.matrix-table {
  width: 100%;
  min-width: 720px;
  border-collapse: separate;
  border-spacing: 0.35rem;
}

.matrix-table th,
.matrix-table td {
  padding: 0.8rem;
  text-align: center;
  border-radius: 14px;
  background: rgba(13, 31, 45, 0.05);
  color: var(--q-text);
}

.p-dark .matrix-table th,
.p-dark .matrix-table td {
  background: rgba(255, 255, 255, 0.06);
}

.matrix-table th {
  font-size: 0.85rem;
}

.matrix-axis {
  font-weight: 700;
}

.matrix-cell--diag {
  background: var(--q-teal) !important;
  color: white !important;
  font-weight: 800;
}

.config-block {
  margin: 0;
  padding: 1rem;
  border-radius: 20px;
  background: rgba(13, 31, 45, 0.92);
  color: #d8edf1;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
}

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

:deep(.benchmark-table .p-datatable-header-cell),
:deep(.benchmark-table .p-datatable-tbody > tr > td) {
  background: transparent;
  color: var(--q-text);
}

:deep(.benchmark-select .p-togglebutton) {
  border: 0;
  background: transparent;
  padding: 0;
  border-radius: 999px;
  overflow: visible;
  box-shadow: none;
}

.p-dark :deep(.benchmark-select .p-togglebutton) {
  background: transparent;
}

:deep(.benchmark-select .p-selectbutton) {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}

:deep(.benchmark-select .p-togglebutton .p-togglebutton-content) {
  border-radius: 999px;
  border: 1px solid var(--q-bar-border);
  background: rgba(255, 255, 255, 0.72);
  color: var(--q-text);
  padding: 0.62rem 1rem;
  transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
  box-shadow: 0 10px 24px rgba(13, 31, 45, 0.06);
}

.p-dark :deep(.benchmark-select .p-togglebutton .p-togglebutton-content) {
  background: rgba(12, 24, 36, 0.92);
  color: var(--q-text);
}

:deep(.benchmark-select .p-togglebutton:not(.p-disabled):hover .p-togglebutton-content) {
  background: var(--q-teal-soft);
  border-color: rgba(42, 184, 184, 0.28);
  color: var(--q-text);
}

:deep(.benchmark-select .p-togglebutton.p-togglebutton-checked) {
  box-shadow: none;
}

:deep(.benchmark-select .p-togglebutton.p-togglebutton-checked .p-togglebutton-content) {
  background: var(--q-teal);
  border-color: var(--q-teal);
  color: white;
  box-shadow: 0 12px 28px rgba(42, 184, 184, 0.18);
}

:deep(.benchmark-select .p-togglebutton.p-togglebutton-checked:hover .p-togglebutton-content) {
  background: var(--q-teal);
  border-color: var(--q-teal);
  color: white;
}

:deep(.benchmark-select .p-togglebutton .p-button-label) {
  font-weight: 700;
  white-space: nowrap;
}

:deep(.benchmark-select .p-togglebutton::before),
:deep(.benchmark-select .p-togglebutton::after),
:deep(.benchmark-select .p-togglebutton .p-togglebutton-content::before),
:deep(.benchmark-select .p-togglebutton .p-togglebutton-content::after) {
  display: none;
}

.benchmark-page--ar .hero-copy,
.benchmark-page--ar .section-heading__text {
  direction: rtl;
  text-align: right;
}

.section-heading--rtl {
  flex-direction: row-reverse;
}

.eyebrow--ar {
  letter-spacing: 0;
  text-transform: none;
}

@media (max-width: 1280px) {
  .summary-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 1080px) {
  .hero-shell,
  .dataset-layout,
  .performance-grid,
  .latency-grid,
  .diagnostics-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 820px) {
  .hero-stats,
  .fact-grid,
  .summary-grid,
  .skeleton-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .hero-stats,
  .fact-grid,
  .summary-grid,
  .skeleton-grid {
    grid-template-columns: 1fr;
  }

  .dataset-image,
  .dataset-slide {
    min-height: 320px;
    height: 320px;
  }

  :deep(.benchmark-select .p-selectbutton) {
    gap: 0.5rem;
  }

  :deep(.benchmark-select .p-togglebutton) {
    width: 100%;
    justify-content: center;
  }
}
</style>
