<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  TrendingUp, Cpu, Atom, Info, ArrowUp,
  Activity, Zap, Shield, FlaskConical, AlertCircle,
  Layers, Target, Gauge, BarChart3, Sigma, GitBranch
} from 'lucide-vue-next'
import { useRouter } from 'vue-router'

const router = useRouter()
const { t, locale } = useI18n({ useScope: 'global' })
const isRtl = computed(() => locale.value === 'AR')

// ── Data Fetching ──
const qaData = ref(null)
const isLoading = ref(true)
const error = ref(null)

const headerPt = { root: '!bg-transparent qa-accordion-header border-none' }
const contentPt = {
  root: '!bg-transparent border-none pb-4',
  transition: '!transition-none'  // kills the slow animation
}

const activePanels = ref([])

const fetchQAData = async () => {
  isLoading.value = true
  error.value = null
  try {
    const res = await fetch('api/v1/quantum-advantage')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    qaData.value = await res.json()
  } catch (err) {
    console.error('Failed to fetch QA data:', err)
    error.value = t('qa.errorMsg')
  } finally {
    isLoading.value = false
  }
}

onMounted(async () => {
  await fetchQAData()
  await nextTick()

  const saved = localStorage.getItem('scrollRestore')
  if (saved) {
    const { path, top } = JSON.parse(saved)
    if (path === router.currentRoute.value.fullPath) {
      window.scrollTo({ top, behavior: 'instant' })  // no type assertion
    }
  }
})

const formatModelName = (name) => name.replace(/_/g, ' ')

// ── Experiment 1: Feature Orthogonality ──
const featureOrthogonality = computed(() => {
  if (!qaData.value?.experiment_1_feature_orthogonality) return {}
  return qaData.value.experiment_1_feature_orthogonality
})

const primaryOrthogonality = computed(() => {
  return Object.values(featureOrthogonality.value)[0] || 0
})

// ── Experiment 2: Branch Ablation ──
const branchAblationModels = computed(() => {
  if (!qaData.value?.experiment_2_branch_ablation) return []
  return Object.entries(qaData.value.experiment_2_branch_ablation).map(([name, m]) => ({
    name: formatModelName(name),
    full_accuracy: m.full_accuracy || 0,
    classical_only_accuracy: m.classical_only_accuracy || 0,
    quantum_only_accuracy: m.quantum_only_accuracy || 0,
    quantum_gain_pct: m['quantum_gain_%'] || m.quantum_gain_pct || 0
  }))
})

const primaryQuantumGain = computed(() => {
  const models = branchAblationModels.value
  return models.length ? models[0].quantum_gain_pct.toFixed(2) : '0'
})

// ── Experiment 3: Re-upload Ablation ──
const reuploadData = computed(() => {
  if (!qaData.value?.experiment_3_reupload_ablation) return { with: 0, without: 0, contribution: 0 }
  const r = Object.values(qaData.value.experiment_3_reupload_ablation)[0]
  return {
    with: r?.with_reupload_accuracy?.toFixed(2) || 0,
    without: r?.without_reupload_accuracy?.toFixed(2) || 0,
    contribution: (r?.['reupload_contribution_%'] || r?.reupload_contribution_pct || 0).toFixed(2)
  }
})

// ── Experiment 4: Entanglement Entropy ──
const entanglementData = computed(() => {
  if (!qaData.value?.experiment_4_entanglement_entropy) return { perQubit: [], labels: [], overall: 0, interpretation: '' }
  const ent = Object.values(qaData.value.experiment_4_entanglement_entropy)[0]
  if (!ent) return { perQubit: [], labels: [], overall: 0, interpretation: '' }
  return {
    perQubit: Object.values(ent.mean_entropy_per_qubit),
    labels: Object.keys(ent.mean_entropy_per_qubit).map(k => `Q${k}`),
    overall: ent.overall_mean_entropy || 0,
    interpretation: ent.interpretation || ''
  }
})

// ── Experiment 5: Gradient Variance ──
const gradientVarianceRows = computed(() => {
  if (!qaData.value?.experiment_5_gradient_variance) return []
  return Object.entries(qaData.value.experiment_5_gradient_variance).map(([model, m]) => ({
    model: formatModelName(model),
    layer: m.target,
    mean_var: m.mean_grad_variance,
    abs_mean: m.mean_grad_abs_mean,
    batches: m.n_batches,
    highlight: model.toLowerCase().includes('baseline') || model.toLowerCase().includes('classical')
  }))
})

const gradientInterpretation = computed(() => {
  if (!qaData.value?.experiment_5_gradient_variance) return ''
  return Object.values(qaData.value.experiment_5_gradient_variance)[0]?.interpretation || ''
})

// ── Experiment 6: Noise Ablation ──
const selectedNoiseType = ref('gaussian')
const noiseTypes = computed(() => {
  if (!qaData.value?.experiment_6_noise_ablation) return []
  const first = Object.values(qaData.value.experiment_6_noise_ablation)[0]
  return first ? Object.keys(first) : []
})

const noiseChartData = computed(() => {
  if (!qaData.value?.experiment_6_noise_ablation) return { labels: [], datasets: [] }
  const exp = qaData.value.experiment_6_noise_ablation
  const type = selectedNoiseType.value
  const colors = ['#2ab8b8', '#6366F1']
  const datasets = Object.entries(exp).map(([model, data], i) => {
    const rows = data[type] || []
    return {
      label: formatModelName(model),
      data: rows.map(r => r['quantum_noise_gain_%'] ?? r.quantum_noise_gain_pct ?? 0),
      borderColor: colors[i % colors.length],
      backgroundColor: colors[i % colors.length] + '33',
      tension: 0.3,
      pointRadius: 4,
      fill: false
    }
  })
  const labels = Object.values(exp)[0]?.[type]?.map(r => r.level.toString()) || []
  return { labels, datasets }
})

const noiseChartOptions = ref({
  responsive: true, maintainAspectRatio: false,
  scales: {
    x: { title: { display: true, color: '#8db4bf' }, ticks: { color: '#8db4bf' }, grid: { color: 'rgba(42,184,184,0.1)' } },
    y: { title: { display: true, color: '#8db4bf' }, ticks: { color: '#8db4bf' }, grid: { color: 'rgba(42,184,184,0.1)' } }
  },
  plugins: { legend: { labels: { color: '#8db4bf' } } }
})

// ── Experiment 7: VQC Expressibility ──
const expressibilityData = computed(() => {
  if (!qaData.value?.experiment_7_vqc_expressibility) return []
  return Object.entries(qaData.value.experiment_7_vqc_expressibility).map(([model, m]) => ({
    model: formatModelName(model),
    kl: m.kl_divergence_from_haar,
    ref: m.haar_reference,
    interpretation: m.interpretation
  }))
})

// ── Experiment 8: Kernel Target Alignment ──
const ktaData = computed(() => {
  if (!qaData.value?.experiment_8_kernel_target_alignment) return []
  return Object.entries(qaData.value.experiment_8_kernel_target_alignment).map(([model, m]) => ({
    model: formatModelName(model),
    quantum: m.kta_quantum,
    classical: m.kta_classical,
    diff: m.kta_difference,
    wins: m.quantum_wins,
    interpretation: m.interpretation
  }))
})

// ── Experiment 9: Geometric Difference ──
const geoData = computed(() => {
  if (!qaData.value?.experiment_9_geometric_difference) return []
  return Object.entries(qaData.value.experiment_9_geometric_difference).map(([model, m]) => ({
    model: formatModelName(model),
    value: m.geometric_difference,
    advantage: m.advantage,
    interpretation: m.interpretation
  }))
})

// ── Experiment 10: Fisher Effective Dimension ──
const fisherData = computed(() => {
  if (!qaData.value?.experiment_10_fisher_effective_dim) return []
  return Object.entries(qaData.value.experiment_10_fisher_effective_dim).map(([model, m]) => ({
    model: formatModelName(model),
    nParams: m.n_quantum_params || m.n_params,
    dEff: m.effective_dimension,
    dEffPerParam: m.d_eff_per_param,
    interpretation: m.interpretation
  }))
})

// ── Experiment 11: Feature Effective Rank ──
const effectiveRankData = computed(() => {
  if (!qaData.value?.experiment_11_feature_effective_rank) return []
  return Object.entries(qaData.value.experiment_11_feature_effective_rank).map(([model, m]) => ({
    model: formatModelName(model),
    zUtil: m.z_utilisation,
    qUtil: m.q_emb_utilisation,
    interpretation: m.interpretation
  }))
})

// ── Experiment 12: Intrinsic Dimension ──
const intrinsicDimData = computed(() => {
  if (!qaData.value?.experiment_12_intrinsic_dimension) return []
  return Object.entries(qaData.value.experiment_12_intrinsic_dimension).map(([model, m]) => ({
    model: formatModelName(model),
    dimZ: m.intrinsic_dim_z,
    dimQ: m.intrinsic_dim_q_emb,
    interpretation: m.interpretation
  }))
})

// ── Experiment 13: Linear CKA ──
const ckaData = computed(() => {
  if (!qaData.value?.experiment_13_linear_cka) return []
  return Object.entries(qaData.value.experiment_13_linear_cka).map(([model, m]) => ({
    model: formatModelName(model),
    cka: m.cka_classical_vs_quantum,
    interpretation: m.interpretation
  }))
})

// ── Experiment 14: Class Separability ──
const separabilityData = computed(() => {
  if (!qaData.value?.experiment_14_class_separability) return []
  return Object.entries(qaData.value.experiment_14_class_separability).map(([model, m]) => ({
    model: formatModelName(model),
    fisherZ: m.fisher_criterion_z,
    fisherZProj: m.fisher_criterion_z_proj,
    fisherQ: m.fisher_criterion_q_emb,
    advantage: m.q_advantage,
    interpretation: m.interpretation
  }))
})

// ── Methodology Notes ──
const isArabic = computed(() => locale.value.toLowerCase() === 'ar')
const notesAr = {
  entanglement_entropy:
    'QNN GPU يستخدم lightning.gpu؛ الأوزان تُنسخ إلى default.qubit لاستخراج حالة التشابك.',
  expressibility:
    'تم استخدام الأوزان المدربة — يقيس قابلية التعبير للدارة المتعلمة، وليس فقط سعة النموذج.',
  kernel_experiments:
    'تم تقييم كل من K_Q و K_C على المدخلات الكمومية ذات 6 أبعاد (بعد المُحدد الكمومي وبعد قياس الزوايا). المقارنة صالحة لأن فضاء الإدخال متطابق.',
  fim:
    'QNN: مصفوفة FIM تجريبية كاملة 36×36 عبر التمايز العكسي لكل عينة. CNN: تقريب قطري بسبب العدد الكبير من معلمات الطبقة الأخيرة.',
  parameter_matched_ablation:
    'غير متضمن — يتطلب تدريب شبكة MLP كلاسيكية جديدة بنفس عدد معلمات VQC. يُشغل في سكربت تدريب منفصل.',
}

const methodologyNotes = computed(() => {
  const data = qaData.value
  if (!data?.config?.notes) return []

  const n = data.config.notes
  return Object.entries(n)
    .filter(([, v]) => v)
    .map(([k, v]) => {
      const tKey = `qa.methodology.notes.${k}`
      const translated = t(tKey)
      const title =
        translated !== tKey
          ? translated
          : k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

      // Pick the correct language text
      const text = isArabic.value ? (notesAr[k] || v) : v

      return { title, text }
    })
})


// ── Chart Configs ──
const barChartData = computed(() => ({
  labels: branchAblationModels.value.map(m => m.name),
  datasets: [
    { label: t('qa.exp2.fullModel'), backgroundColor: '#2ab8b8', data: branchAblationModels.value.map(m => m.full_accuracy), borderRadius: 4 },
    { label: t('qa.exp2.classicalOnly'), backgroundColor: '#6366F1', data: branchAblationModels.value.map(m => m.classical_only_accuracy), borderRadius: 4 },
    { label: t('qa.exp2.quantumOnly'), backgroundColor: '#F59E0B', data: branchAblationModels.value.map(m => m.quantum_only_accuracy), borderRadius: 4 },
  ]
}))

const barChartOptions = ref({
  responsive: true, maintainAspectRatio: false,
  scales: {
    x: { ticks: { color: '#8db4bf' }, grid: { color: 'rgba(42,184,184,0.1)', drawBorder: false } },
    y: { min: 0, max: 100, ticks: { color: '#8db4bf' }, grid: { color: 'rgba(42,184,184,0.1)', drawBorder: false } }
  },
  plugins: { legend: { labels: { color: '#8db4bf' } } }
})

const radarChartData = computed(() => ({
  labels: entanglementData.value.labels,
  datasets: [{
    label: 'Entropy',
    backgroundColor: 'rgba(42, 184, 184, 0.2)',
    borderColor: '#2ab8b8',
    pointBackgroundColor: '#2ab8b8',
    data: entanglementData.value.perQubit
  }]
}))

const radarChartOptions = ref({
  responsive: true, maintainAspectRatio: false,
  scales: {
    r: { min: 0, max: 1, grid: { color: 'rgba(42,184,184,0.15)' }, pointLabels: { color: '#8db4bf' }, ticks: { color: '#8db4bf', backdropColor: 'transparent' } }
  },
  plugins: { legend: { display: false } }
})

// ── SVG Gauge Helpers ──
const circSize = 160
const circR = (circSize - 14) / 2
const circCircumference = 2 * Math.PI * circR
const circOffset = computed(() => circCircumference * (1 - Math.min(entanglementData.value.overall / 1, 1)))

const halfAngle = computed(() => Math.PI * primaryOrthogonality.value)
const halfGaugeX = computed(() => 110 - 90 * Math.cos(halfAngle.value))
const halfGaugeY = computed(() => 130 - 90 * Math.sin(halfAngle.value))
const halfNeedleX = computed(() => 110 - 70 * Math.cos(halfAngle.value))
const halfNeedleY = computed(() => 130 - 70 * Math.sin(halfAngle.value))

// ── Helpers ──
const rowClass = (data) => data.highlight ? 'qa-row-highlight' : ''
const formatExp = (val) => val ? val.toExponential(2) : 'N/A'
const pct = (val) => (val * 100).toFixed(1)
</script>

<template>
  <div :class="['transition-colors pb-16', isRtl ? 'text-right' : 'text-left']" :dir="isRtl ? 'rtl' : 'ltr'">

    <!-- Hero -->
    <div class="px-4 pt-10 mx-auto mb-10 max-w-screen-2xl sm:px-6 lg:px-8">
      <div class="flex items-center gap-3 mb-2">
        <h1 class="text-3xl qa-title">{{ t('qa.title') }}</h1>
      </div>
      <p style="color: var(--q-muted)">{{ t('qa.subtitle') }}</p>
    </div>

    <!-- Loading -->
    <div v-if="isLoading" class="flex flex-col items-center justify-center py-20">
      <ProgressSpinner style="width: 50px; height: 50px" strokeWidth="4" animationDuration=".5s" />
      <p class="mt-4 animate-pulse" style="color: var(--q-muted)">{{ t('qa.loading') }}</p>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="px-4 mx-auto max-w-screen-2xl sm:px-6 lg:px-8">
      <div class="qa-card flex flex-col items-center p-6 text-center">
        <AlertCircle class="w-12 h-12 mb-3 text-red-500" />
        <h3 class="text-lg font-medium text-red-500">{{ t('qa.errorTitle') }}</h3>
        <p class="mt-1 text-red-400">{{ error }}</p>
      </div>
    </div>

    <!-- Main Content -->
    <div v-else-if="qaData" class="px-4 mx-auto space-y-6 max-w-screen-2xl sm:px-6 lg:px-8">

      <!-- Config Badges -->
      <div class="flex flex-wrap gap-3 mb-4 text-sm">
        <span class="qa-badge inline-flex items-center">
          <Cpu class="w-3.5 h-3.5" :class="isRtl ? 'ml-1' : 'mr-1'" />{{ qaData.device }}
        </span>
        <span class="qa-badge inline-flex items-center">
          <Atom class="w-3.5 h-3.5" :class="isRtl ? 'ml-1' : 'mr-1'" />{{ qaData.config.n_qubits }} {{ t('qa.qubits') }}, {{ t('qa.depth') }} {{ qaData.config.q_depth }}
        </span>
        <span class="qa-badge inline-flex items-center">
          <Cpu class="w-3.5 h-3.5" :class="isRtl ? 'ml-1' : 'mr-1'" />
          QNN CPU: default.qubit
        </span>
        <span class="qa-badge inline-flex items-center">
          <Cpu class="w-3.5 h-3.5" :class="isRtl ? 'ml-1' : 'mr-1'" />
          QNN GPU: lightning.gpu
        </span>
      </div>

      <!-- ─── Exp 2: Branch Ablation ─── -->
      <section class="qa-card">
        <div class="qa-card-header">
          <div class="qa-icon-wrap"><TrendingUp class="w-5 h-5" /></div>
          <div>
            <h2 class="qa-card-title">{{ t('qa.exp2.title') }}</h2>
            <p class="qa-card-desc">{{ t('qa.exp2.desc') }}</p>
          </div>
        </div>
        <div class="flex flex-wrap items-center gap-4 mb-6">
          <div class="inline-flex items-center gap-2 px-5 py-3 rounded-xl" style="background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3)">
            <ArrowUp class="w-5 h-5 text-[#22C55E]" />
            <span class="text-2xl text-[#22C55E]">+{{ primaryQuantumGain }}%</span>
            <span class="text-[#22C55E]/80">{{ t('qa.exp2.quantumGain') }}</span>
          </div>
          <Info v-tooltip.top="t('qa.exp2.tooltip')" class="w-4 h-4 cursor-help" style="color: var(--q-muted)" />
        </div>
        <div class="h-72">
          <Chart type="bar" :data="barChartData" :options="barChartOptions" class="w-full h-full" />
        </div>
      </section>

      <!-- ─── Exp 6: Noise Ablation ─── -->
      <section v-if="noiseTypes.length" class="qa-card">
        <div class="qa-card-header">
          <div class="qa-icon-wrap"><Shield class="w-5 h-5" /></div>
          <div>
            <h2 class="qa-card-title">{{ t('qa.exp6.title') }}</h2>
            <p class="qa-card-desc">{{ t('qa.exp6.desc') }}</p>
          </div>
        </div>
        <div class="flex flex-wrap gap-2 mb-4">
          <button v-for="nt in noiseTypes" :key="nt"
            @click="selectedNoiseType = nt"
            class="px-3 py-1.5 rounded-lg text-sm transition-colors"
            :class="selectedNoiseType === nt ? 'qa-tab-active' : 'qa-tab-inactive'">
            {{ t('qa.exp6.' + nt) }}
          </button>
        </div>
        <div class="h-72">
          <Chart type="line" :data="noiseChartData" :options="noiseChartOptions" class="w-full h-full" />
        </div>
        <div class="qa-note-green mt-4">
          <Info class="w-4 h-4 inline-block text-[#22C55E]" :class="isRtl ? 'ml-1' : 'mr-1'" />
          {{ t('qa.exp6.insight') }}
        </div>
      </section>

      <!-- ─── Exp 1: Feature Orthogonality ─── -->
      <section class="qa-card">
        <div class="qa-card-header">
          <div class="qa-icon-wrap"><Activity class="w-5 h-5" /></div>
          <div>
            <h2 class="qa-card-title">{{ t('qa.exp1.title') }}</h2>
            <p class="qa-card-desc">{{ t('qa.exp1.desc') }}</p>
          </div>
        </div>
        <div class="flex flex-col items-center gap-8 md:flex-row">
          <div class="flex justify-center flex-1">
            <svg width="220" height="140" viewBox="0 0 220 140">
              <path d="M 20 130 A 90 90 0 0 1 200 130" fill="none" stroke="var(--q-bar-border)" stroke-width="14" stroke-linecap="round" />
              <path :d="`M 20 130 A 90 90 0 0 1 ${halfGaugeX} ${halfGaugeY}`" fill="none" stroke="#22C55E" stroke-width="14" stroke-linecap="round" />
              <line x1="110" y1="130" :x2="halfNeedleX" :y2="halfNeedleY" stroke="var(--q-teal)" stroke-width="3" stroke-linecap="round" />
              <circle cx="110" cy="130" r="5" fill="var(--q-teal)" />
              <text x="15" y="138" class="text-xs" fill="var(--q-muted)">0.0</text>
              <text x="195" y="138" class="text-xs" fill="var(--q-muted)">1.0</text>
            </svg>
          </div>
          <div class="flex-1 space-y-3">
            <div class="text-4xl" style="color: var(--q-text)">{{ primaryOrthogonality.toFixed(3) }}</div>
            <div class="text-sm" style="color: var(--q-muted)">{{ t('qa.exp1.score') }}</div>
            <div class="inline-block px-3 py-1 bg-[#22C55E]/10 text-[#22C55E] rounded-full text-sm">
              {{ t('qa.exp1.target') }}
            </div>
            <p class="text-sm" style="color: var(--q-muted)">{{ t('qa.exp1.explanation') }}</p>
            <div v-if="Object.keys(featureOrthogonality).length > 1" class="flex gap-4 pt-2">
              <div v-for="(val, model) in featureOrthogonality" :key="model" class="text-sm">
                <span style="color: var(--q-muted)">{{ formatModelName(model) }}: </span>
                <span class="font-mono" style="color: var(--q-text)"> {{ val.toFixed(4) }}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- ─── Linear CKA + Re-upload: 2 cols ─── -->
      <div class="grid grid-cols-1 gap-6 lg:grid-cols-2">

    <!-- ─── Exp 13: Linear CKA ─── -->
    <section v-if="ckaData.length" class="qa-card">
      <div class="qa-card-header">
        <div class="qa-icon-wrap"><Layers class="w-5 h-5" /></div>
        <div>
          <h2 class="qa-card-title">{{ t('qa.exp13.title') }}</h2>
          <p class="qa-card-desc">{{ t('qa.exp13.desc') }}</p>
        </div>
      </div>
      <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div v-for="d in ckaData" :key="d.model" class="qa-metric-card">
          <div class="text-sm" style="color: var(--q-muted)">{{ d.model }}</div>
          <div class="text-3xl font-mono" style="color: var(--q-text)">{{ d.cka.toFixed(3) }}</div>
          <div class="text-xs mt-1" style="color: var(--q-muted)">{{ t('qa.exp13.scale') }}</div>
        </div>
      </div>
      <div class="qa-note mt-4">
        <Info class="w-4 h-4 inline-block" style="color: var(--q-teal)" :class="isRtl ? 'ml-1' : 'mr-1'" />
        {{ t('qa.exp13.insight') }}
      </div>
    </section>

    <!-- ─── Exp 3: Re-upload Contribution ─── -->
    <section class="qa-card">
      <div class="qa-card-header">
        <div class="qa-icon-wrap"><Zap class="w-5 h-5" /></div>
        <div>
          <h2 class="qa-card-title">{{ t('qa.exp3.title') }}</h2>
          <p class="qa-card-desc">{{ t('qa.exp3.desc') }}</p>
        </div>
      </div>
      <div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div class="qa-metric-card">
          <div class="text-sm" style="color: var(--q-muted)">{{ t('qa.exp3.with') }}</div>
          <div class="text-2xl" style="color: var(--q-text)">{{ reuploadData.with }}%</div>
        </div>
        <div class="qa-metric-card">
          <div class="text-sm" style="color: var(--q-muted)">{{ t('qa.exp3.without') }}</div>
          <div class="text-2xl" style="color: var(--q-text)">{{ reuploadData.without }}%</div>
        </div>
        <div class="qa-metric-card" style="background: rgba(34,197,94,0.05); border-color: rgba(34,197,94,0.2)">
          <div class="text-sm" style="color: var(--q-muted)">{{ t('qa.exp3.contribution') }}</div>
          <div class="flex items-center justify-center gap-1">
            <ArrowUp class="w-4 h-4 text-[#22C55E]" />
            <span class="text-2xl text-[#22C55E]">+{{ reuploadData.contribution }}%</span>
          </div>
        </div>
      </div>
    </section>

  </div>

      <!-- ─── Exp 4: Entanglement Entropy ─── -->
      <section class="qa-card">
        <div class="qa-card-header">
          <div class="qa-icon-wrap"><Atom class="w-5 h-5" /></div>
          <div>
            <h2 class="qa-card-title">{{ t('qa.exp4.title') }}</h2>
            <p class="qa-card-desc">{{ t('qa.exp4.desc') }}</p>
          </div>
        </div>
        <div class="flex flex-col gap-8 lg:flex-row">
          <div class="flex flex-col items-center gap-2">
            <div class="relative flex items-center justify-center">
              <svg :width="circSize" :height="circSize" class="-rotate-90">
                <circle :cx="circSize/2" :cy="circSize/2" :r="circR" fill="none" stroke="var(--q-bar-border)" stroke-width="10" />
                <circle :cx="circSize/2" :cy="circSize/2" :r="circR" fill="none" stroke="var(--q-teal)" stroke-width="10" :stroke-dasharray="circCircumference" :stroke-dashoffset="circOffset" stroke-linecap="round" />
              </svg>
              <div class="absolute flex flex-col items-center justify-center" :style="{ width: circSize+'px', height: circSize+'px' }">
                <span class="text-2xl" style="color: var(--q-text)">{{ entanglementData.overall.toFixed(2) }}</span>
              </div>
            </div>
            <span class="text-sm" style="color: var(--q-muted)">{{ t('qa.exp4.overallMean') }}</span>
          </div>
          <div class="flex-1 h-64">
            <Chart type="radar" :data="radarChartData" :options="radarChartOptions" class="w-full h-full" />
          </div>
        </div>
        <div class="qa-note mt-4">
          <Info class="w-4 h-4 inline-block" style="color: var(--q-teal)" :class="isRtl ? 'ml-1' : 'mr-1'" />
          {{ entanglementData.interpretation }}
        </div>
      </section>

      <!-- ─── Exp 5: Gradient Variance ─── -->
      <section class="qa-card">
        <div class="qa-card-header">
          <div class="qa-icon-wrap"><BarChart3 class="w-5 h-5" /></div>
          <div>
            <h2 class="qa-card-title">{{ t('qa.exp5.title') }}</h2>
            <p class="qa-card-desc">{{ t('qa.exp5.desc') }}</p>
          </div>
        </div>
        <Card class="q-glass qa-inner-card">
          <template #content>
            <DataTable :value="gradientVarianceRows" :rowClass="rowClass" :class="['text-sm p-datatable-sm', isRtl ? 'qa-table-rtl' : '']">
              <Column field="model" :header="t('qa.model')">
                <template #body="{ data }">
                  <span style="color: var(--q-text)" :class="{ 'font-medium': data.highlight }">{{ data.model }}</span>
                  <Tag v-if="data.highlight" value="Baseline" class="ml-2 bg-[#6366F1]/10 text-[#6366F1] !text-xs !py-0.5" />
                </template>
              </Column>
              <Column field="layer" :header="t('qa.exp5.target')">
                <template #body="{ data }"><span class="font-mono text-xs" style="color: var(--q-muted)">{{ data.layer }}</span></template>
              </Column>
              <Column field="mean_var" :header="t('qa.exp5.meanVar')">
                <template #body="{ data }"><span class="font-mono" style="color: var(--q-text)">{{ formatExp(data.mean_var) }}</span></template>
              </Column>
              <Column field="abs_mean" :header="t('qa.exp5.absMean')">
                <template #body="{ data }"><span class="font-mono" style="color: var(--q-text)">{{ formatExp(data.abs_mean) }}</span></template>
              </Column>
              <Column field="batches" :header="t('qa.exp5.batches')"></Column>
            </DataTable>
            </template>
            </Card>
        <div class="qa-note-green mt-4">
          <Info class="w-4 h-4 inline-block text-[#22C55E]" :class="isRtl ? 'ml-1' : 'mr-1'" />
          {{ gradientInterpretation }}
        </div>
      </section>

      <!-- ─── VQC Expressibility + Geometric Difference: 2 cols ─── -->
      <div class="grid grid-cols-1 gap-6 lg:grid-cols-2">

    <section v-if="expressibilityData.length" class="qa-card">
      <div class="qa-card-header">
        <div class="qa-icon-wrap"><Sigma class="w-5 h-5" /></div>
        <div>
          <h2 class="qa-card-title">{{ t('qa.exp7.title') }}</h2>
          <p class="qa-card-desc">{{ t('qa.exp7.desc') }}</p>
        </div>
      </div>
      <div class="grid grid-cols-1 gap-4">
        <div v-for="d in expressibilityData" :key="d.model" class="qa-metric-card">
          <div class="text-sm font-medium" style="color: var(--q-text)">{{ d.model }}</div>
          <div class="text-3xl font-mono mt-1" style="color: var(--q-teal)">{{ d.kl.toFixed(4) }}</div>
          <div class="text-xs mt-1" style="color: var(--q-muted)">{{ t('qa.exp7.klDiv') }}</div>
          <div class="text-xs mt-2" style="color: var(--q-muted)">{{ t('qa.exp7.ref') }}: {{ d.ref }}</div>
        </div>
      </div>
      <div class="qa-note mt-4">
        <Info class="w-4 h-4 inline-block" style="color: var(--q-teal)" :class="isRtl ? 'ml-1' : 'mr-1'" />
        {{ t('qa.exp7.insight') }}
      </div>
    </section>

    <section v-if="geoData.length" class="qa-card">
      <div class="qa-card-header">
        <div class="qa-icon-wrap"><Target class="w-5 h-5" /></div>
        <div>
          <h2 class="qa-card-title">{{ t('qa.exp9.title') }}</h2>
          <p class="qa-card-desc">{{ t('qa.exp9.desc') }}</p>
        </div>
      </div>
      <div class="grid grid-cols-1 gap-4">
        <div v-for="d in geoData" :key="d.model" class="qa-metric-card">
          <div class="text-sm" style="color: var(--q-muted)">{{ d.model }}</div>
          <div class="text-3xl font-mono" style="color: var(--q-text)">{{ d.value.toFixed(2) }}</div>
          <div class="mt-2">
            <Tag :value="d.advantage ? t('qa.confirmed') : t('qa.notConfirmed')"
              :class="d.advantage ? 'bg-[#22C55E]/10 text-[#22C55E]' : 'bg-red-500/10 text-red-400'" class="!text-xs" />
          </div>
        </div>
      </div>
      <div class="qa-note-green mt-4">
        <Info class="w-4 h-4 inline-block text-[#22C55E]" :class="isRtl ? 'ml-1' : 'mr-1'" />
        {{ t('qa.exp9.insight') }}
      </div>
    </section>

  </div>

      <!-- ─── KTA + Fisher: 2 cols ─── -->
  <div class="grid grid-cols-1 gap-6 lg:grid-cols-2">

    <section v-if="ktaData.length" class="qa-card">
      <div class="qa-card-header">
        <div class="qa-icon-wrap"><GitBranch class="w-5 h-5" /></div>
        <div>
          <h2 class="qa-card-title">{{ t('qa.exp8.title') }}</h2>
          <p class="qa-card-desc">{{ t('qa.exp8.desc') }}</p>
        </div>
      </div>
      <Card class="q-glass qa-inner-card">
        <template #content>
          <DataTable :value="ktaData" :class="['text-sm p-datatable-sm', isRtl ? 'qa-table-rtl' : '']">
            <Column field="model" :header="t('qa.model')">
              <template #body="{ data }"><span style="color: var(--q-text)">{{ data.model }}</span></template>
            </Column>
            <Column :header="t('qa.exp8.quantum')">
              <template #body="{ data }"><span class="font-mono" style="color: var(--q-teal)">{{ data.quantum.toFixed(4) }}</span></template>
            </Column>
            <Column :header="t('qa.exp8.classical')">
              <template #body="{ data }"><span class="font-mono" style="color: var(--q-text)">{{ data.classical.toFixed(4) }}</span></template>
            </Column>
            <Column :header="t('qa.exp8.diff')">
              <template #body="{ data }">
                <span class="font-mono" :class="data.diff > 0 ? 'text-[#22C55E]' : 'text-[#F59E0B]'">{{ data.diff > 0 ? '+' : '' }}{{ data.diff.toFixed(4) }}</span>
              </template>
            </Column>
          </DataTable>
        </template>
      </Card>
      <div class="qa-note mt-4">
        <Info class="w-4 h-4 inline-block" style="color: var(--q-teal)" :class="isRtl ? 'ml-1' : 'mr-1'" />
        {{ t('qa.exp8.insight') }}
      </div>
    </section>

    <section v-if="fisherData.length" class="qa-card">
      <div class="qa-card-header">
        <div class="qa-icon-wrap"><Gauge class="w-5 h-5" /></div>
        <div>
          <h2 class="qa-card-title">{{ t('qa.exp10.title') }}</h2>
          <p class="qa-card-desc">{{ t('qa.exp10.desc') }}</p>
        </div>
      </div>
      <Card class="q-glass qa-inner-card">
        <template #content>
          <DataTable :value="fisherData" :class="['text-sm p-datatable-sm', isRtl ? 'qa-table-rtl' : '']">
            <Column field="model" :header="t('qa.model')">
              <template #body="{ data }">
                <span style="color: var(--q-text)">{{ data.model }}</span>
                <Tag v-if="data.model.includes('CNN')" value="Baseline" class="ml-2 bg-[#6366F1]/10 text-[#6366F1] !text-xs !py-0.5" />
              </template>
            </Column>
            <Column :header="t('qa.exp10.params')">
              <template #body="{ data }"><span class="font-mono" style="color: var(--q-text)">{{ data.nParams }}</span></template>
            </Column>
            <Column :header="t('qa.exp10.dEff1000')">
              <template #body="{ data }"><span class="font-mono" style="color: var(--q-teal)">{{ data.dEff?.['1000']?.toFixed(2) || 'N/A' }}</span></template>
            </Column>
            <Column :header="t('qa.exp10.dEffPerParam')">
              <template #body="{ data }"><span class="font-mono" style="color: var(--q-text)">{{ data.dEffPerParam?.['1000']?.toFixed(6) || 'N/A' }}</span></template>
            </Column>
          </DataTable>
        </template>
      </Card>
      <div class="qa-note mt-4">
        <Info class="w-4 h-4 inline-block" style="color: var(--q-teal)" :class="isRtl ? 'ml-1' : 'mr-1'" />
        {{ t('qa.exp10.insight') }}
      </div>
    </section>

  </div>

      <!-- ─── Exp 11: Feature Effective Rank ─── -->
      <section v-if="effectiveRankData.length" class="qa-card">
        <div class="qa-card-header">
          <div class="qa-icon-wrap"><BarChart3 class="w-5 h-5" /></div>
          <div>
            <h2 class="qa-card-title">{{ t('qa.exp11.title') }}</h2>
            <p class="qa-card-desc">{{ t('qa.exp11.desc') }}</p>
          </div>
        </div>
        <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div v-for="d in effectiveRankData" :key="d.model" class="qa-metric-card space-y-3 text-start">
            <div class="text-sm font-medium" style="color: var(--q-text)">{{ d.model }}</div>
            <div class="flex justify-between text-sm">
              <span style="color: var(--q-muted)">{{ t('qa.exp11.classical') }} (z)</span>
              <span class="font-mono" style="color: var(--q-text)">{{ pct(d.zUtil) }}%</span>
            </div>
            <div class="w-full rounded-full h-2 overflow-hidden" style="background: var(--q-bar-border)">
              <div class="h-full rounded-full bg-[#6366F1]" :style="{ width: pct(d.zUtil)+'%' }"></div>
            </div>
            <div class="flex justify-between text-sm">
              <span style="color: var(--q-muted)">{{ t('qa.exp11.quantum') }} (q_emb)</span>
              <span class="font-mono" style="color: var(--q-teal)">{{ pct(d.qUtil) }}%</span>
            </div>
            <div class="w-full rounded-full h-2 overflow-hidden" style="background: var(--q-bar-border)">
              <div class="h-full rounded-full" style="background: var(--q-teal)" :style="{ width: pct(d.qUtil)+'%' }"></div>
            </div>
          </div>
        </div>
        <div class="qa-note mt-4">
          <Info class="w-4 h-4 inline-block" style="color: var(--q-teal)" :class="isRtl ? 'ml-1' : 'mr-1'" />
          {{ t('qa.exp11.insight') }}
        </div>
      </section>

      <!-- ─── Intrinsic Dimension + Class Separability: 2 cols ─── -->
  <div class="grid grid-cols-1 gap-6 lg:grid-cols-2">

    <section v-if="intrinsicDimData.length" class="qa-card">
      <div class="qa-card-header">
        <div class="qa-icon-wrap"><Sigma class="w-5 h-5" /></div>
        <div>
          <h2 class="qa-card-title">{{ t('qa.exp12.title') }}</h2>
          <p class="qa-card-desc">{{ t('qa.exp12.desc') }}</p>
        </div>
      </div>
      <div class="grid grid-cols-1 gap-4">
        <div v-for="d in intrinsicDimData" :key="d.model" class="qa-metric-card">
          <div class="text-sm font-medium" style="color: var(--q-text)">{{ d.model }}</div>
          <div class="flex gap-6 mt-2 justify-center">
            <div>
              <div class="text-xs" style="color: var(--q-muted)">{{ t('qa.exp12.classical') }}</div>
              <div class="text-xl font-mono" style="color: var(--q-text)">{{ d.dimZ.toFixed(2) }}</div>
            </div>
            <div>
              <div class="text-xs" style="color: var(--q-muted)">{{ t('qa.exp12.quantum') }}</div>
              <div class="text-xl font-mono" style="color: var(--q-teal)">{{ d.dimQ.toFixed(2) }}</div>
            </div>
          </div>
        </div>
      </div>
      <div class="qa-note mt-4">
        <Info class="w-4 h-4 inline-block" style="color: var(--q-teal)" :class="isRtl ? 'ml-1' : 'mr-1'" />
        {{ t('qa.exp12.insight') }}
      </div>
    </section>

    <section v-if="separabilityData.length" class="qa-card">
      <div class="qa-card-header">
        <div class="qa-icon-wrap"><Target class="w-5 h-5" /></div>
        <div>
          <h2 class="qa-card-title">{{ t('qa.exp14.title') }}</h2>
          <p class="qa-card-desc">{{ t('qa.exp14.desc') }}</p>
        </div>
      </div>
      <Card class="q-glass qa-inner-card">
        <template #content>
          <DataTable :value="separabilityData" :class="['text-sm p-datatable-sm', isRtl ? 'qa-table-rtl' : '']">
            <Column field="model" :header="t('qa.model')">
              <template #body="{ data }"><span style="color: var(--q-text)">{{ data.model }}</span></template>
            </Column>
            <Column header="J(z)">
              <template #body="{ data }"><span class="font-mono" style="color: var(--q-text)">{{ data.fisherZ.toFixed(2) }}</span></template>
            </Column>
            <Column header="J(z_proj)">
              <template #body="{ data }"><span class="font-mono" style="color: var(--q-text)">{{ data.fisherZProj.toFixed(2) }}</span></template>
            </Column>
            <Column header="J(q_emb)">
              <template #body="{ data }"><span class="font-mono" style="color: var(--q-teal)">{{ data.fisherQ.toFixed(2) }}</span></template>
            </Column>
            <Column :header="t('qa.exp14.advantage')">
              <template #body="{ data }">
                <Tag :value="data.advantage ? '✓' : '✗'"
                  :class="data.advantage ? 'bg-[#22C55E]/10 text-[#22C55E]' : 'bg-[#F59E0B]/10 text-[#F59E0B]'" class="!text-xs" />
              </template>
            </Column>
          </DataTable>
        </template>
      </Card>
      <div class="qa-note mt-4">
        <Info class="w-4 h-4 inline-block" style="color: var(--q-teal)" :class="isRtl ? 'ml-1' : 'mr-1'" />
        {{ t('qa.exp14.insight') }}
      </div>
    </section>

  </div>

      <!-- ─── Methodology Notes ─── -->
      <section v-if="methodologyNotes.length" class="qa-card">
        <div class="qa-card-header">
          <div class="qa-icon-wrap"><FlaskConical class="w-5 h-5" /></div>
          <h2 class="qa-card-title">{{ t('qa.methodology.title') }}</h2>
        </div>
        <Accordion v-model:value="activePanels" multiple>
          <AccordionPanel
            v-for="(note, i) in methodologyNotes"
            :key="i"
            :value="i"
          >
            <AccordionHeader :pt="headerPt">
              {{ note.title }}
            </AccordionHeader>
            <AccordionContent :pt="contentPt">
              <p class="m-0 text-sm" style="color: var(--q-muted)">
                {{ note.text }}
              </p>
            </AccordionContent>
          </AccordionPanel>
        </Accordion>
      </section>

    </div>
  </div>
</template>

<style scoped>
.qa-title{
  margin: 0 0 0.7rem;
  color: var(--q-text);
  font-family: var(--q-font-display);
  font-size: clamp(2.3rem, 4vw, 3.6rem);
  line-height: 1.05;
}

.qa-card {
  border: 1px solid var(--q-bar-border);
  background: var(--q-surface);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: var(--q-shadow);
  border-radius: var(--q-radius-sm);
  padding: 1.5rem;
}

.qa-card-header {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
}

.qa-card-title {
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--q-text);
  margin: 0;
  font-family: var(--q-font-display);
}

.qa-card-desc {
  font-size: 0.875rem;
  color: var(--q-muted);
  margin-top: 0.125rem;
}

.qa-icon-wrap {
  padding: 0.5rem;
  border-radius: 0.5rem;
  background: var(--q-teal-soft);
  color: var(--q-teal);
  flex-shrink: 0;
}

.qa-badge {
  padding: 0.375rem 0.75rem;
  border: 1px solid var(--q-bar-border);
  background: var(--q-surface-strong);
  border-radius: 0.5rem;
  color: var(--q-muted);
}

.qa-metric-card {
  padding: 1rem;
  border: 1px solid var(--q-bar-border);
  background: var(--q-surface-strong);
  border-radius: var(--q-radius-sm);
  text-align: center;
}

.qa-note {
  padding: 0.75rem 1rem;
  border: 1px solid rgba(42, 184, 184, 0.2);
  background: var(--q-surface-soft);
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: var(--q-muted);
}

.qa-note-green {
  padding: 0.75rem 1rem;
  border: 1px solid rgba(34, 197, 94, 0.2);
  background: rgba(34, 197, 94, 0.05);
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: var(--q-muted);
}

.qa-tab-active {
  background: var(--q-teal);
  color: #fff;
}

.qa-tab-inactive {
  background: var(--q-surface-strong);
  color: var(--q-muted);
  border: 1px solid var(--q-bar-border);
}
.qa-tab-inactive:hover {
  background: var(--q-teal-soft);
  color: var(--q-teal);
}

.qa-table-border {
  border: 1px solid var(--q-bar-border);
}

.qa-accordion-header {
  color: var(--q-text) !important;
}

.qa-inner-card {
  border-radius: var(--q-radius-sm) !important;
}
.qa-inner-card :deep(.p-card-body) {
  padding: 0 !important;
}

/* ── DataTable — same as ClassifyImage ── */
:deep(.p-datatable) {
  background: transparent;
  border-radius: var(--q-radius-sm);
  overflow: hidden;
}
:deep(.p-datatable-thead > tr > th) {
  background: transparent !important;
  color: #94a3b8;
  border-bottom: 1px solid var(--q-bar-border) !important;
  padding: 0.875rem 1.25rem;
  font-size: 0.72rem;
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
  font-size: 0.875rem;
  text-align: left;
}
:deep(.p-datatable-tbody > tr:last-child > td) {
  border-bottom: none !important;
}
:deep(.p-datatable-tbody > tr:hover > td) {
  background: var(--q-surface-soft) !important;
  transition: background 0.15s ease;
}
:deep(.p-datatable-tbody > tr.qa-row-highlight > td) {
  background: rgba(99, 102, 241, 0.05) !important;
}

/* ── RTL table alignment ── */
:deep(.qa-table-rtl .p-datatable-thead > tr > th) {
  text-align: right !important;
}
:deep(.qa-table-rtl .p-datatable-tbody > tr > td) {
  text-align: right !important;
}
:deep(.p-accordionpanel) {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--q-bar-border) !important;
}
:deep(.p-accordionheader-toggle) {
  background: transparent !important;
  color: var(--q-text) !important;
  border: none !important;
  box-shadow: none !important;
  padding: 1rem 0;
}
:deep(.p-accordionheader-toggle:hover) {
  background: var(--q-surface-soft) !important;
  border-radius: 8px;
}
:deep(.p-accordioncontent),
:deep(.p-accordioncontent-content) {
  background: transparent !important;
  border: none !important;
}
</style>
