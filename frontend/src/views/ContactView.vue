<script setup>
import { ref, computed, watch, onMounted } from 'vue';
import { useToast } from 'primevue/usetoast';
import { useI18n } from 'vue-i18n'

const { t, locale } = useI18n()
const dir = computed(() => locale.value === 'AR' ? 'rtl' : 'ltr')
const isArabic = computed(() => locale.value === 'AR')
const toast = useToast();

const STORAGE_KEY = 'contactForm'

//  Form state 
const contact = ref({ name: '', subject: '', message: '' });
const touched = ref({ name: false, subject: false, message: false });
const isSending = ref(false);

onMounted(() => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      // Assign all at once → single watcher trigger, prevents mid-hydration wipe
      contact.value = {
        name: parsed.name ?? '',
        subject: parsed.subject ?? '',
        message: parsed.message ?? '',
      }
    }
  } catch {
    // ignore corrupted data
  }
})

// Always persist — resetEmail() handles explicit removal
watch(contact, (newVal) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(newVal))
}, { deep: true })

const errors = computed(() => ({
  name: touched.value.name && !contact.value.name.trim(),
  subject: touched.value.subject && !contact.value.subject.trim(),
  message: touched.value.message && !contact.value.message.trim(),
}));

const isValid = computed(
  () =>
    contact.value.name.trim() &&
    contact.value.subject.trim() &&
    contact.value.message.trim()
);

function touch(field) {
  touched.value[field] = true;
}

function resetEmail() {
  contact.value = { name: '', subject: '', message: '' };
  touched.value = { name: false, subject: false, message: false };
  localStorage.removeItem(STORAGE_KEY)
}

async function sendEmail() {
  // Mark all touched to surface validation errors
  touched.value = { name: true, subject: true, message: true };
  if (!isValid.value) return;

  isSending.value = true;

  try {
    const response = await fetch("api/v1/contact", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      // Send the data as a clean JSON payload
      body: JSON.stringify({
        name: contact.value.name,
        subject: contact.value.subject,
        message: contact.value.message
      })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    // Show Success Toast
    toast.add({
      severity: 'success',
      summary: t('contact.successTitle'),
      detail: t('contact.successMsg'),
      life: 4000
    });

    // Reset Form
    resetEmail()

  } catch (error) {
    console.error('Contact form error:', error);

    // Show Error Toast
    toast.add({
      severity: 'error',
      summary: t('contact.errorTitle'),
      detail: t('contact.errorMsg'),
      life: 5000
    });
  } finally {
    isSending.value = false;
  }
}
</script>

<template>
  <div
    class="contact-root"
    :class="{ 'contact-root--ar': isArabic }"
  >
    <Toast />

    <section class="contact-shell">
      <aside
        class="contact-side q-glass"
        :dir="locale === 'AR' ? 'rtl' : 'ltr'"
        aria-hidden="true"
      >
        <div class="contact-side__orb contact-side__orb--one" />
        <div class="contact-side__orb contact-side__orb--two" />
        <div class="contact-side__content">
          <span class="contact-side__eyebrow">{{ t('contact.sideEyebrow') }}</span>
          <h2 class="contact-side__title">
            {{ t('contact.sideTitle') }}
          </h2>
          <p class="contact-side__text">
            {{ t('contact.sideText') }}
          </p>
        </div>
      </aside>


      <main
        class="contact-card q-glass"
        :dir="dir"
      >
        <header class="form-header">
          <h1 class="form-title">
            {{ t('contact.pageTitle') }}
          </h1>
          <p class="form-subtitle">
            {{ t('contact.pageSubtitle') }}
          </p>
        </header>

        <form
          class="form-body"
          novalidate
          @submit.prevent="sendEmail"
        >
          <div class="field-group">
            <div class="field-label-row">
              <label
                class="field-label"
                for="contact-name"
              >{{ t('contact.name') }}</label>
              <Button
                v-if="contact.name || contact.subject || contact.message"
                type="button"
                icon="pi pi-times"
                :disabled="isSending"
                class="clear-fab"
                :aria-label="t('contact.clear')"
                @click="resetEmail"
              />
            </div>
            <InputText
              id="contact-name"
              v-model="contact.name"
              :placeholder="t('contact.namePlaceholder')"
              :class="['field-input', { 'p-invalid': errors.name }]"
              autocomplete="name"
              @blur="touch('name')"
            />
            <small
              v-if="errors.name"
              class="field-error"
            >
              <i class="pi pi-exclamation-circle" /> {{ t('contact.required') }}
            </small>
          </div>

          <div class="field-group">
            <label
              class="field-label"
              for="contact-subject"
            >{{ t('contact.subject') }}</label>
            <InputText
              id="contact-subject"
              v-model="contact.subject"
              :placeholder="t('contact.subjectPlaceholder')"
              :class="['field-input', { 'p-invalid': errors.subject }]"
              @blur="touch('subject')"
            />
            <small
              v-if="errors.subject"
              class="field-error"
            >
              <i class="pi pi-exclamation-circle" /> {{ t('contact.required') }}
            </small>
          </div>

          <div class="field-group">
            <label
              class="field-label"
              for="contact-message"
            >{{ t('contact.message') }}</label>
            <Textarea
              id="contact-message"
              v-model="contact.message"
              :placeholder="t('contact.messagePlaceholder')"
              :class="['field-input', { 'p-invalid': errors.message }]"
              rows="6"
              auto-resize
              @blur="touch('message')"
            />
            <small
              v-if="errors.message"
              class="field-error"
            >
              <i class="pi pi-exclamation-circle" /> {{ t('contact.required') }}
            </small>
          </div>

          <Button
            type="submit"
            :label="isSending ? t('contact.sending') : t('contact.send')"
            icon="pi pi-envelope"
            :loading="isSending"
            :disabled="isSending"
            class="submit-btn"
          />
        </form>
      </main>
    </section>
  </div>
</template>


<style scoped>
.contact-root {
  min-height: 100vh;
  padding: 1.5rem 1rem 3rem;
  color: var(--q-text);
}

.contact-root--ar .contact-side__title {
  font-size: clamp(2.2rem, 4.2vw, 3.5rem);
  line-height: 1.2;
}

.contact-root--ar .form-title {
  font-size: clamp(2.15rem, 3.2vw, 3rem);
  line-height: 1.25;
}

.contact-root--ar .form-subtitle,
.contact-root--ar .contact-side__text {
  font-size: 1.05rem;
  line-height: 2;
}

.contact-root--ar .field-label {
  font-size: 0.9rem;
  letter-spacing: 0;
}

.contact-root--ar :deep(.p-inputtext),
.contact-root--ar :deep(.p-textarea),
.contact-root--ar :deep(textarea),
.contact-root--ar :deep(input) {
  font-size: 1rem !important;
}

.lang-ar * {
  direction: rtl;
}

.lang-ar .contact-side__eyebrow {
  font-size: 0.95rem;
  letter-spacing: 0;
  text-transform: none;
}

.lang-ar .contact-side__title {
  font-size: clamp(2.3rem, 4.4vw, 3.7rem);
  line-height: 1.25;
}

.lang-ar .contact-side__text {
  font-size: 1.08rem;
  line-height: 2;
}

.lang-ar .form-title {
  font-size: clamp(2.15rem, 3.2vw, 3.1rem);
  line-height: 1.25;
}

.lang-ar .form-subtitle {
  font-size: 1.02rem;
  line-height: 1.95;
}

.lang-ar .field-label {
  font-size: 0.9rem;
  letter-spacing: 0;
}

.lang-ar .p-inputtext,
.lang-ar .p-button-label,
.lang-ar textarea {
  font-size: 1rem;
}

.lang-ar .form-title,
.lang-ar .contact-side__title,
.lang-ar .hero-title,
.lang-ar .hero-subtitle,
.lang-ar .section-title,
.lang-ar .team-group-title,
.lang-ar .site-footer__heading {
  font-family: var(--q-font-display);
}

.contact-shell {
  max-width: 1280px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: minmax(280px, 0.85fr) minmax(0, 1.15fr);
  gap: 1.5rem;
  align-items: stretch;
}

.contact-side,
.contact-card {
  border-radius: var(--q-radius-lg);
}

.contact-side {
  position: relative;
  overflow: hidden;
  min-height: 620px;
  padding: 2rem;
  display: flex;
  align-items: flex-end;
  background:
    radial-gradient(circle at top right, rgba(42, 184, 184, 0.2), transparent 30%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent),
    var(--q-surface);
}

.contact-side__content {
  position: relative;
  z-index: 1;
  max-width: 320px;
}

.contact-side__eyebrow {
  display: inline-block;
  margin-bottom: 0.75rem;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--q-teal);
}

.contact-side__title {
  margin: 0;
  font-family: var(--q-font-display);
  font-size: clamp(2rem, 4vw, 3.25rem);
  line-height: 1.05;
  color: var(--q-text);
}

.contact-side__text {
  margin-top: 1rem;
  color: var(--q-muted);
  line-height: 1.9;
}

.contact-side__orb {
  position: absolute;
  border-radius: 999px;
  filter: blur(4px);
}

.contact-side__orb--one {
  top: -40px;
  right: -60px;
  width: 240px;
  height: 240px;
  background: rgba(42, 184, 184, 0.22);
}

.contact-side__orb--two {
  bottom: 30px;
  left: -50px;
  width: 180px;
  height: 180px;
  background: rgba(42, 184, 184, 0.12);
}

.contact-card {
  padding: clamp(1.5rem, 3vw, 2.5rem);
}

.form-header {
  margin-bottom: 1.8rem;
}

.form-title {
  margin: 0 0 0.75rem;
  font-family: var(--q-font-display);
  font-size: clamp(2rem, 3vw, 2.8rem);
  color: var(--q-text);
  line-height: 1.1;
}

.form-subtitle {
  margin: 0;
  color: var(--q-muted);
  line-height: 1.8;
  max-width: 60ch;
}

.form-body {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.field-group {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.field-label {
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--q-muted);
}

:deep(.field-input.p-inputtext),
:deep(.field-input.p-textarea),
:deep(.field-input textarea),
:deep(.field-input input) {
  width: 100%;
  background: var(--q-surface-strong) !important;
  border: 1px solid var(--q-bar-border) !important;
  border-radius: 16px !important;
  color: var(--q-text) !important;
  font-family: var(--q-font-body) !important;
  padding: 0.9rem 1rem !important;
  box-shadow: none !important;
}

:deep(.field-input.p-inputtext:focus),
:deep(.field-input.p-textarea:focus),
:deep(.field-input textarea:focus),
:deep(.field-input input:focus) {
  border-color: var(--q-teal) !important;
  box-shadow: 0 0 0 3px rgba(42, 184, 184, 0.18) !important;
}

:deep(.p-inputtext::placeholder),
:deep(textarea::placeholder) {
  color: var(--q-muted) !important;
  opacity: 0.8;
}

:deep(.p-invalid),
:deep(.p-inputtext.p-invalid),
:deep(textarea.p-invalid) {
  border-color: var(--q-error) !important;
}

.field-error {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--q-error);
  font-size: 0.82rem;
}

:deep(.submit-btn.p-button) {
  margin-top: 0.5rem;
  width: 100%;
  justify-content: center;
  border: none !important;
  border-radius: 16px !important;
  padding: 0.95rem 1.25rem !important;
  background: var(--q-teal) !important;
  color: white !important;
  box-shadow: 0 14px 34px rgba(42, 184, 184, 0.22) !important;
}

:deep(.submit-btn.p-button:hover:not(:disabled)) {
  opacity: 0.92;
  transform: translateY(-1px);
}

@media (max-width: 960px) {
  .contact-shell {
    grid-template-columns: 1fr;
  }

  .contact-side {
    min-height: 280px;
    align-items: center;
  }
}

@media (max-width: 640px) {
  .contact-root {
    padding-inline: 0.75rem;
  }

  .contact-card,
  .contact-side {
    border-radius: 24px;
  }
}

.contact-card {
  position: relative;
  /* needed for the absolute fab */
}

.field-label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

:deep(.clear-fab.p-button) {
  width: 2.1rem !important;
  height: 2.1rem !important;
  padding: 0 !important;
  border-radius: 999px !important;
  background: var(--q-surface-strong) !important;
  border: 2px solid var(--q-bar-border) !important;
  color: var(--q-muted) !important;
  box-shadow: none !important;
}

:deep(.clear-fab.p-button:hover:not(:disabled)) {
  border-color: var(--q-error) !important;
  color: var(--q-error) !important;
  background: var(--q-surface-strong) !important;
  transform: rotate(90deg) !important;
}

:deep(.clear-fab .p-button-icon) {
  font-size: 0.75rem !important;
}
</style>