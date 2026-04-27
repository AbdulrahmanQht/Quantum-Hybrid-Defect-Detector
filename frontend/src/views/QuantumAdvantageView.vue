<script setup>
import { ref, computed, onMounted } from 'vue';

import {
  TrendingUp, Cpu, Atom, Info, ArrowUp,
  Activity, Zap, Shield, FlaskConical, AlertCircle
} from 'lucide-vue-next';

// ── Props ──
const props = defineProps({
  language: {
    type: String,
    default: 'en'
  }
});

const isAr = computed(() => props.language === 'ar');

// ── Data Fetching State ──
const qaData = ref(null);
const isLoading = ref(true);
const error = ref(null);

const fetchQAData = async () => {
  isLoading.value = true;
  error.value = null;
  try {
    const response = await fetch('api/v1/quantum-advantage');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    qaData.value = await response.json();
  } catch (err) {
    console.error("Failed to fetch Quantum Advantage results:", err);
    error.value = "Failed to load quantum advantage data. Please ensure the backend is running and the results file is generated.";
  } finally {
    isLoading.value = false;
  }
};

onMounted(() => {
  fetchQAData();
});

// ── Computed Properties mapping FastAPI data to UI ──

// 1. Feature Orthogonality
const featureOrthogonality = computed(() => {
  if (!qaData.value?.experiment_1_feature_orthogonality) return 0;
  // Grab the first value from the dictionary
  return Object.values(qaData.value.experiment_1_feature_orthogonality)[0] || 0;
});

// 2. Branch Ablation Models
const branchAblationModels = computed(() => {
  if (!qaData.value?.experiment_2_branch_ablation) return [];
  return Object.entries(qaData.value.experiment_2_branch_ablation).map(([name, metrics]) => ({
    name,
    full_accuracy: metrics.full_accuracy || 0,
    classical_only_accuracy: metrics.classical_only_accuracy || 0,
    quantum_only_accuracy: metrics.quantum_only_accuracy || 0,
    quantum_gain_pct: metrics['quantum_gain_%'] || metrics.quantum_gain_pct || 0 // Handle Pydantic alias
  }));
});

const primaryQuantumGain = computed(() => {
  const models = branchAblationModels.value;
  if (!models.length) return 0;
  // Default to the first model's gain
  return models[0].quantum_gain_pct.toFixed(2);
});

// 3. Re-upload Ablation
const reuploadData = computed(() => {
  if (!qaData.value?.experiment_3_reupload_ablation) return { with: 0, without: 0, contribution: 0 };
  const reupload = Object.values(qaData.value.experiment_3_reupload_ablation)[0];
  return {
    with: reupload?.with_reupload_accuracy?.toFixed(2) || 0,
    without: reupload?.without_reupload_accuracy?.toFixed(2) || 0,
    contribution: (reupload?.['reupload_contribution_%'] || reupload?.reupload_contribution_pct || 0).toFixed(2)
  };
});

// 4. Entanglement Entropy
const entanglementData = computed(() => {
  if (!qaData.value?.experiment_4_entanglement_entropy) return { perQubit: [], labels: [], overall: 0, interpretation: '' };
  const ent = Object.values(qaData.value.experiment_4_entanglement_entropy)[0];
  if (!ent) return { perQubit: [], labels: [], overall: 0, interpretation: '' };
  
  return {
    perQubit: Object.values(ent.mean_entropy_per_qubit),
    labels: Object.keys(ent.mean_entropy_per_qubit).map(k => `Qubit ${k}`),
    overall: ent.overall_mean_entropy || 0,
    interpretation: ent.interpretation || ''
  };
});

// 5. Gradient Variance
const gradientVarianceRows = computed(() => {
  if (!qaData.value?.experiment_5_gradient_variance) return [];
  return Object.entries(qaData.value.experiment_5_gradient_variance).map(([model, metrics]) => ({
    model,
    layer: metrics.target,
    mean_var: metrics.mean_grad_variance,
    abs_mean: metrics.mean_grad_abs_mean,
    batches: metrics.n_batches,
    highlight: model.toLowerCase().includes('baseline') || model.toLowerCase().includes('classical')
  }));
});

const gradientInterpretation = computed(() => {
  if (!qaData.value?.experiment_5_gradient_variance) return '';
  return Object.values(qaData.value.experiment_5_gradient_variance)[0]?.interpretation || '';
});

// 6. Methodology Notes (From Config)
const methodologyNotes = computed(() => {
  if (!qaData.value?.config?.notes) return [];
  const notes = qaData.value.config.notes;
  return [
    { title: 'Entanglement Entropy', text: notes.entanglement_entropy },
    { title: 'Branch Ablation', text: notes.branch_ablation },
    { title: 'Gradient Variance', text: notes.gradient_variance },
  ];
});

// ── Reactive Chart Configurations ──
const barChartData = computed(() => ({
  labels: branchAblationModels.value.map(m => m.name),
  datasets: [
    { label: 'Full Model', backgroundColor: '#14B8A6', data: branchAblationModels.value.map(m => m.full_accuracy), borderRadius: 4 },
    { label: 'Classical Only', backgroundColor: '#6366F1', data: branchAblationModels.value.map(m => m.classical_only_accuracy), borderRadius: 4 },
    { label: 'Quantum Only', backgroundColor: '#F59E0B', data: branchAblationModels.value.map(m => m.quantum_only_accuracy), borderRadius: 4 },
  ]
}));

const barChartOptions = ref({
  responsive: true, maintainAspectRatio: false,
  scales: {
    x: { ticks: { color: '#94A3B8' }, grid: { color: '#334155', drawBorder: false } },
    y: { min: 0, max: 100, ticks: { color: '#94A3B8' }, grid: { color: '#334155', drawBorder: false } }
  },
  plugins: { legend: { labels: { color: '#94A3B8' } } }
});

const radarChartData = computed(() => ({
  labels: entanglementData.value.labels,
  datasets: [
    {
      label: 'Entropy',
      backgroundColor: 'rgba(20, 184, 166, 0.3)',
      borderColor: '#14B8A6',
      pointBackgroundColor: '#14B8A6',
      data: entanglementData.value.perQubit
    }
  ]
}));

const radarChartOptions = ref({
  responsive: true, maintainAspectRatio: false,
  scales: {
    r: {
      min: 0, max: 1, grid: { color: '#334155' },
      pointLabels: { color: '#94A3B8' }, ticks: { color: '#94A3B8', backdropColor: 'transparent' }
    }
  },
  plugins: { legend: { display: false } }
});

// ── Computed Properties for SVG Gauges ──
const circSize = 180;
const circR = (circSize - 16) / 2;
const circCircumference = 2 * Math.PI * circR;
const circOffset = computed(() => circCircumference * (1 - Math.min(entanglementData.value.overall / 1, 1)));

const halfAngle = computed(() => Math.PI * featureOrthogonality.value);
const halfGaugeX = computed(() => 110 - 90 * Math.cos(halfAngle.value));
const halfGaugeY = computed(() => 130 - 90 * Math.sin(halfAngle.value));
const halfNeedleX = computed(() => 110 - 70 * Math.cos(halfAngle.value));
const halfNeedleY = computed(() => 130 - 70 * Math.sin(halfAngle.value));

// ── DataTable Helpers ──
const rowClass = (data) => data.highlight ? '!bg-[#6366F1]/5 font-medium' : '';
const formatExponential = (val) => val ? val.toExponential(2) : 'N/A';
</script>

<template>
  <div>
    
    <div class="px-4 pt-10 mx-auto mb-10 max-w-7xl sm:px-6 lg:px-8">
      <div class="flex items-center gap-3 mb-2">
        <FlaskConical class="w-8 h-8 text-[#14B8A6]" />
        <h1 class="text-3xl text-[#0F172A] dark:text-[#F8FAFC]">
          {{ isAr ? 'تقرير الميزة الكمية' : 'Quantum Advantage Report' }}
        </h1>
      </div>
      <p class="text-[#64748B] dark:text-[#94A3B8] mb-4">
        {{ isAr ? 'التحقق التجريبي من الآليات الكمية في البنية الهجينة.' : 'Empirical validation of quantum mechanisms in the hybrid architecture.' }}
      </p>
    </div>

    <div v-if="isLoading" class="flex flex-col items-center justify-center py-20">
      <ProgressSpinner style="width: 50px; height: 50px" strokeWidth="4" animationDuration=".5s" aria-label="Loading Metrics" />
      <p class="mt-4 text-[#64748B] dark:text-[#94A3B8] animate-pulse">Loading quantum metrics...</p>
    </div>

    <div v-else-if="error" class="px-4 mx-auto max-w-7xl sm:px-6 lg:px-8">
      <div class="flex flex-col items-center p-6 text-center border border-red-200 bg-red-50 dark:bg-red-900/20 dark:border-red-800 rounded-xl">
        <AlertCircle class="w-12 h-12 mb-3 text-red-500" />
        <h3 class="text-lg font-medium text-red-800 dark:text-red-300">Data Unavailable</h3>
        <p class="mt-1 text-red-600 dark:text-red-400">{{ error }}</p>
      </div>
    </div>

    <div v-else-if="qaData" class="px-4 mx-auto space-y-8 max-w-7xl sm:px-6 lg:px-8">
      
      <div class="flex flex-wrap gap-4 mb-6 text-sm">
        <span class="px-3 py-1.5 bg-white dark:bg-[#1E293B] border border-[#E2E8F0] dark:border-[#334155] rounded-lg text-[#475569] dark:text-[#94A3B8]">
          {{ isAr ? 'تاريخ التوليد' : 'Generated At' }}: {{ new Date(qaData.generated_at).toLocaleString() }}
        </span>
        <span class="flex items-center px-3 py-1.5 bg-white dark:bg-[#1E293B] border border-[#E2E8F0] dark:border-[#334155] rounded-lg text-[#475569] dark:text-[#94A3B8]">
          <Cpu class="w-3.5 h-3.5 mr-1" />{{ qaData.device }}
        </span>
        <span class="flex items-center px-3 py-1.5 bg-white dark:bg-[#1E293B] border border-[#E2E8F0] dark:border-[#334155] rounded-lg text-[#475569] dark:text-[#94A3B8]">
          <Atom class="w-3.5 h-3.5 mr-1" />{{ qaData.config.n_qubits }} Qubits, Depth {{ qaData.config.q_depth }}
        </span>
      </div>

      <div class="bg-white dark:bg-[#1E293B] rounded-xl border border-[#E2E8F0] dark:border-[#334155] p-6 shadow-sm">
        <div class="flex items-center gap-3 mb-6">
          <div class="p-2 bg-[#14B8A6]/10 rounded-lg text-[#14B8A6]"><TrendingUp class="w-5 h-5" /></div>
          <h2 class="text-xl text-[#0F172A] dark:text-[#F8FAFC]">{{ isAr ? 'الكسب الكمي (إزالة الفرع)' : 'The "Quantum Gain" (Branch Ablation)' }}</h2>
        </div>
        
        <div class="flex flex-wrap items-center gap-4 mb-6">
          <div class="inline-flex items-center gap-2 px-5 py-3 bg-[#22C55E]/10 border border-[#22C55E]/30 rounded-xl">
            <ArrowUp class="w-5 h-5 text-[#22C55E]" />
            <span class="text-2xl text-[#22C55E]">+{{ primaryQuantumGain }}%</span>
            <span class="text-[#22C55E]/80">{{ isAr ? 'كسب كمي' : 'Quantum Gain' }}</span>
          </div>
          <Info v-tooltip.top="'Quantum Gain = Full Accuracy − Classical Only Accuracy'" class="w-4 h-4 text-[#94A3B8] cursor-help" />
        </div>
        
        <div class="h-72">
          <Chart type="bar" :data="barChartData" :options="barChartOptions" class="w-full h-full" />
        </div>
      </div>

      <div class="bg-white dark:bg-[#1E293B] rounded-xl border border-[#E2E8F0] dark:border-[#334155] p-6 shadow-sm">
        <div class="flex items-center gap-3 mb-6">
          <div class="p-2 bg-[#14B8A6]/10 rounded-lg text-[#14B8A6]"><Activity class="w-5 h-5" /></div>
          <h2 class="text-xl text-[#0F172A] dark:text-[#F8FAFC]">{{ isAr ? 'تعامد الميزات' : 'Feature Orthogonality' }}</h2>
        </div>
        <div class="flex flex-col items-center gap-8 md:flex-row">
          <div class="flex justify-center flex-1">
            <div class="relative">
              <svg width="220" height="140" viewBox="0 0 220 140">
                <path d="M 20 130 A 90 90 0 0 1 200 130" fill="none" stroke="currentColor" class="text-[#E2E8F0] dark:text-[#334155]" stroke-width="14" stroke-linecap="round" />
                <path :d="`M 20 130 A 90 90 0 0 1 ${halfGaugeX} ${halfGaugeY}`" fill="none" stroke="#22C55E" stroke-width="14" stroke-linecap="round" />
                <line x1="110" y1="130" :x2="halfNeedleX" :y2="halfNeedleY" stroke="#EF4444" stroke-width="3" stroke-linecap="round" />
                <circle cx="110" cy="130" r="5" fill="#EF4444" />
                <text x="15" y="138" class="text-xs fill-[#94A3B8]">0.0</text>
                <text x="195" y="138" class="text-xs fill-[#94A3B8]">1.0</text>
              </svg>
            </div>
          </div>
          <div class="flex-1 space-y-3">
            <div class="text-4xl text-[#0F172A] dark:text-[#F8FAFC]">{{ featureOrthogonality.toFixed(3) }}</div>
            <div class="text-sm text-[#64748B] dark:text-[#94A3B8]">{{ isAr ? 'تشابه جيب التمام' : 'Cosine Similarity Score' }}</div>
            <div class="inline-block px-3 py-1 bg-[#22C55E]/10 text-[#22C55E] rounded-full text-sm">
              {{ isAr ? 'الهدف: قريب من 0.0 (متعامد/فريد)' : 'Target: Near 0.0 (Orthogonal / Unique)' }}
            </div>
            <p class="text-sm text-[#64748B] dark:text-[#94A3B8]">
              {{ isAr ? 'يشير التشابه المنخفض جداً إلى أن الفرع الكمي يتعلم ميزات فريدة ومكملة.' : 'Very low similarity confirms the quantum branch learns unique, complementary features distinct from the classical branch.' }}
            </p>
          </div>
        </div>
      </div>

      <div class="bg-white dark:bg-[#1E293B] rounded-xl border border-[#E2E8F0] dark:border-[#334155] p-6 shadow-sm">
        <div class="flex items-center gap-3 mb-6">
          <div class="p-2 bg-[#14B8A6]/10 rounded-lg text-[#14B8A6]"><Atom class="w-5 h-5" /></div>
          <h2 class="text-xl text-[#0F172A] dark:text-[#F8FAFC]">{{ isAr ? 'إنتروبيا التشابك' : 'Entanglement Entropy' }}</h2>
        </div>
        <div class="flex flex-col gap-8 lg:flex-row">
          <div class="flex flex-col items-center gap-2">
            <div class="relative flex items-center justify-center">
              <div class="flex flex-col items-center gap-2">
                <svg :width="circSize" :height="circSize" class="-rotate-90">
                  <circle :cx="circSize / 2" :cy="circSize / 2" :r="circR" fill="none" stroke="currentColor" class="text-[#E2E8F0] dark:text-[#334155]" stroke-width="10" />
                  <circle :cx="circSize / 2" :cy="circSize / 2" :r="circR" fill="none" stroke="#14B8A6" stroke-width="10" :stroke-dasharray="circCircumference" :stroke-dashoffset="circOffset" stroke-linecap="round" />
                </svg>
                <div class="absolute flex flex-col items-center justify-center" :style="{ width: circSize + 'px', height: circSize + 'px' }">
                  <span class="text-2xl text-[#0F172A] dark:text-[#F8FAFC]">{{ entanglementData.overall.toFixed(2) }}</span>
                </div>
                <span class="text-sm text-[#64748B] dark:text-[#94A3B8]">{{ isAr ? 'متوسط الإنتروبيا الكلي' : 'Overall Mean Entropy' }}</span>
              </div>
            </div>
          </div>
          <div class="flex-1 h-64">
            <Chart type="radar" :data="radarChartData" :options="radarChartOptions" class="w-full h-full" />
          </div>
        </div>
        <div class="mt-4 p-4 bg-[#14B8A6]/5 border border-[#14B8A6]/20 rounded-lg text-sm text-[#475569] dark:text-[#94A3B8]">
          <Info class="w-4 h-4 inline mr-1 text-[#14B8A6]" />
          {{ entanglementData.interpretation }}
        </div>
      </div>

      <div class="bg-white dark:bg-[#1E293B] rounded-xl border border-[#E2E8F0] dark:border-[#334155] p-6 shadow-sm">
        <div class="flex items-center gap-3 mb-6">
          <div class="p-2 bg-[#14B8A6]/10 rounded-lg text-[#14B8A6]"><Shield class="w-5 h-5" /></div>
          <h2 class="text-xl text-[#0F172A] dark:text-[#F8FAFC]">{{ isAr ? 'تباين التدرج (فحص هضبة البور)' : 'Gradient Variance (Barren Plateau Check)' }}</h2>
        </div>
        
        <DataTable :value="gradientVarianceRows" :rowClass="rowClass" class="text-sm p-datatable-sm overflow-hidden rounded-lg border border-[#E2E8F0] dark:border-[#334155]">
          <Column field="model" header="Model">
            <template #body="{ data }">
              <span class="text-[#0F172A] dark:text-[#F8FAFC]" :class="{ 'font-medium': data.highlight }">{{ data.model }}</span>
              <Tag v-if="data.highlight" value="Baseline" class="ml-2 bg-[#6366F1]/10 text-[#6366F1] !text-xs !py-0.5" />
            </template>
          </Column>
          <Column field="layer" header="Target Layer">
            <template #body="{ data }">
              <span class="font-mono text-xs text-[#475569] dark:text-[#94A3B8]">{{ data.layer }}</span>
            </template>
          </Column>
          <Column field="mean_var" header="Mean Grad Variance">
            <template #body="{ data }">
              <span class="font-mono text-[#0F172A] dark:text-[#F8FAFC]">{{ formatExponential(data.mean_var) }}</span>
            </template>
          </Column>
          <Column field="abs_mean" header="Abs Mean">
            <template #body="{ data }">
              <span class="font-mono text-[#0F172A] dark:text-[#F8FAFC]">{{ formatExponential(data.abs_mean) }}</span>
            </template>
          </Column>
          <Column field="batches" header="Batches" bodyClass="text-[#475569] dark:text-[#94A3B8]"></Column>
        </DataTable>

        <div class="mt-4 p-4 bg-[#22C55E]/5 border border-[#22C55E]/20 rounded-lg text-sm text-[#475569] dark:text-[#94A3B8]">
          <Info class="w-4 h-4 inline mr-1 text-[#22C55E]" />
          {{ gradientInterpretation }}
        </div>
      </div>

      <div class="bg-white dark:bg-[#1E293B] rounded-xl border border-[#E2E8F0] dark:border-[#334155] p-6 shadow-sm">
        <div class="flex items-center gap-3 mb-6">
          <div class="p-2 bg-[#14B8A6]/10 rounded-lg text-[#14B8A6]"><Zap class="w-5 h-5" /></div>
          <h2 class="text-xl text-[#0F172A] dark:text-[#F8FAFC]">{{ isAr ? 'مساهمة إعادة التحميل' : 'Re-upload Contribution' }}</h2>
        </div>
        <div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div class="p-4 bg-[#F8FAFC] dark:bg-[#0F172A] rounded-xl border border-[#E2E8F0] dark:border-[#334155] text-center">
            <div class="text-sm text-[#64748B] dark:text-[#94A3B8] mb-1">{{ isAr ? 'مع إعادة التحميل' : 'With Re-upload' }}</div>
            <div class="text-2xl text-[#0F172A] dark:text-[#F8FAFC]">{{ reuploadData.with }}%</div>
          </div>
          <div class="p-4 bg-[#F8FAFC] dark:bg-[#0F172A] rounded-xl border border-[#E2E8F0] dark:border-[#334155] text-center">
            <div class="text-sm text-[#64748B] dark:text-[#94A3B8] mb-1">{{ isAr ? 'بدون إعادة التحميل' : 'Without Re-upload' }}</div>
            <div class="text-2xl text-[#0F172A] dark:text-[#F8FAFC]">{{ reuploadData.without }}%</div>
          </div>
          <div class="p-4 bg-[#22C55E]/5 border border-[#22C55E]/20 rounded-xl text-center">
            <div class="text-sm text-[#64748B] dark:text-[#94A3B8] mb-1">{{ isAr ? 'المساهمة' : 'Contribution' }}</div>
            <div class="flex items-center justify-center gap-1">
              <ArrowUp class="w-4 h-4 text-[#22C55E]" />
              <span class="text-2xl text-[#22C55E]">+{{ reuploadData.contribution }}%</span>
            </div>
          </div>
        </div>
      </div>
 
      <div class="bg-white dark:bg-[#1E293B] rounded-xl border border-[#E2E8F0] dark:border-[#334155] p-6 shadow-sm">
        <div class="flex items-center gap-3 mb-6">
          <div class="p-2 bg-[#14B8A6]/10 rounded-lg text-[#14B8A6]"><FlaskConical class="w-5 h-5" /></div>
          <h2 class="text-xl text-[#0F172A] dark:text-[#F8FAFC]">{{ isAr ? 'ملاحظات المنهجية' : 'Experiment Notes & Methodology' }}</h2>
        </div>
        
        <Accordion :multiple="true">
          <AccordionTab v-for="(note, i) in methodologyNotes" :key="i" :header="note.title" 
            :pt="{ headerAction: '!bg-transparent !text-[#0F172A] dark:!text-[#F8FAFC] border-none', content: '!bg-transparent !text-[#475569] dark:!text-[#94A3B8] border-none pb-4' }">
            <p class="m-0 text-sm">
              {{ note.text }}
            </p>
          </AccordionTab>
        </Accordion>
      </div>
      
    </div>
  </div>
</template>

<style scoped>
:deep(.p-datatable) {
  @apply bg-transparent;
}
:deep(.p-datatable-thead > tr > th) {
  @apply bg-transparent text-[#64748B] dark:text-[#94A3B8] border-b border-[#E2E8F0] dark:border-[#334155] py-3;
}
:deep(.p-datatable-tbody > tr > td) {
  @apply border-b border-[#E2E8F0] dark:border-[#334155] py-3;
}
:deep(.p-accordion-header-link) {
  @apply border-none !important;
}
:deep(.p-accordion-content) {
  @apply border-none !important;
}
</style>