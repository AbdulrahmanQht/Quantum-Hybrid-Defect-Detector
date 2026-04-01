<script setup>
import { ref, computed } from 'vue';
import { useToast } from 'primevue/usetoast';
import Cookies from 'js-cookie';
import { useI18n } from 'vue-i18n'

const { t, locale } = useI18n()
const dir = computed(() => locale.value === 'AR' ? 'rtl' : 'ltr')
const toast = useToast();

const THEME_KEY = 'theme';
const isDark = ref(Cookies.get(THEME_KEY) === 'dark');

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

// ─── Form state ──────────────────────────────────────────────────────────────
const contact = ref({ name: '', subject: '', message: '' });
const touched = ref({ name: false, subject: false, message: false });
const isSending = ref(false);

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

async function sendEmail() {
  // Mark all touched to surface validation errors
  touched.value = { name: true, subject: true, message: true };
  if (!isValid.value) return;

  isSending.value = true;

  try {
    const response = await fetch(`${API_BASE}/api/contact`, {
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
    contact.value = { name: '', subject: '', message: '' };
    touched.value = { name: false, subject: false, message: false };

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
  <!-- Root wrapper — applies dark class based on isDark -->
  <div :class="['contact-root', { dark: isDark }]">
  <Toast />
    <!-- ── Page layout ──────────────────────────────────────────────────── -->
    <div class="contact-layout">

      <!-- Left accent panel (decorative) -->
      <aside class="accent-panel" aria-hidden="true">
        <div class="accent-circle accent-circle--1" />
        <div class="accent-circle accent-circle--2" />
        <div class="accent-tagline">
          <span class="tagline-word">Hello.</span>
          <span class="tagline-word">مرحباً.</span>
        </div>
      </aside>

      <!-- Main form card -->
      <main class="form-card" :dir="dir">
        <header class="form-header">
          <h1 class="form-title">{{ t('contact.pageTitle') }}</h1>
          <p class="form-subtitle">{{ t('contact.pageSubtitle') }}</p>
        </header>

        <form class="form-body" @submit.prevent="sendEmail" novalidate>

          <!-- Name -->
          <div class="field-group">
            <label class="field-label" for="contact-name">{{ t('contact.name') }}</label>
            <InputText
              id="contact-name"
              v-model="contact.name"
              :placeholder="t('contact.namePlaceholder')"
              :class="['field-input', { 'p-invalid': errors.name }]"
              autocomplete="name"
              @blur="touch('name')"
            />
            <small v-if="errors.name" class="field-error">
              <i class="pi pi-exclamation-circle" /> {{ t('contact.required') }}
            </small>
          </div>

          <!-- Subject -->
          <div class="field-group">
            <label class="field-label" for="contact-subject">{{ t('contact.subject') }}</label>
            <InputText
              id="contact-subject"
              v-model="contact.subject"
              :placeholder="t('contact.subjectPlaceholder')"
              :class="['field-input', { 'p-invalid': errors.subject }]"
              @blur="touch('subject')"
            />
            <small v-if="errors.subject" class="field-error">
              <i class="pi pi-exclamation-circle" /> {{ t('contact.required') }}
            </small>
          </div>

          <!-- Message -->
          <div class="field-group">
            <label class="field-label" for="contact-message">{{ t('contact.message') }}</label>
            <Textarea
              id="contact-message"
              v-model="contact.message"
              :placeholder="t('contact.messagePlaceholder')"
              :class="['field-input', { 'p-invalid': errors.message }]"
              rows="5"
              auto-resize
              @blur="touch('message')"
            />
            <small v-if="errors.message" class="field-error">
              <i class="pi pi-exclamation-circle" /> {{ t('contact.required') }}
            </small>
          </div>

          <!-- Submit -->
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
    </div>
  </div>
</template>

<style scoped>


/* ── Design tokens (light) ──────────────────────────────────────────────── */
.contact-root {
  --bg:          #f5f3ee;
  --bg-card:     #ffffff;
  --bg-panel:    #1a1a2e;
  --text-primary:#1c1b18;
  --text-muted:  #6b6860;
  --accent:      #0D9488;
  --accent-soft: #f0e6e0;
  --border:      #e4e0d8;
  --error:       #d93025;
  --radius:      16px;
  --shadow:      0 8px 40px rgba(0,0,0,.10);
  --font-display:'DM Serif Display', Georgia, serif;
  --font-body:   'DM Sans', system-ui, sans-serif;

  min-height: 100vh;
  background: var(--bg);
  font-family: var(--font-body);
  color: var(--text-primary);
  transition: background 0.3s, color 0.3s;
}

/* ── Dark tokens ────────────────────────────────────────────────────────── */
.contact-root.dark {
  --bg:          #0f0f14;
  --bg-card:     #1a1a24;
  --bg-panel:    #0a0a10;
  --text-primary:#ede9e0;
  --text-muted:  #8a8680;
  --accent:      #0D9488;
  --accent-soft: #095952;
  --border:      #2c2c38;
  --shadow:      0 8px 40px rgba(0,0,0,.4);
}

/* ── Controls bar ───────────────────────────────────────────────────────── */
.controls-bar {
  position: fixed;
  top: 1.25rem;
  right: 1.25rem;
  display: flex;
  gap: 0.5rem;
  z-index: 100;
}

[dir='rtl'] .controls-bar {
  right: auto;
  left: 1.25rem;
}

.ctrl-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  border: 1.5px solid var(--border);
  background: var(--bg-card);
  color: var(--text-primary);
  cursor: pointer;
  font-family: var(--font-body);
  font-size: 0.8rem;
  font-weight: 500;
  transition: background 0.2s, border-color 0.2s, transform 0.15s;
}
.ctrl-btn:hover { background: var(--accent-soft); border-color: var(--accent); transform: scale(1.06); }
.lang-btn { width: auto; padding: 0 0.9rem; border-radius: 999px; }

/* ── Layout ─────────────────────────────────────────────────────────────── */
.contact-layout {
  display: grid;
  grid-template-columns: 1fr 1.6fr;
  min-height: 100vh;
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem;
  gap: 2rem;
  align-items: center;
}

@media (max-width: 768px) {
  .contact-layout {
    grid-template-columns: 1fr;
    padding: 1.25rem;
    padding-top: 5rem;
  }
  .accent-panel { display: none; }
}

/* ── Accent panel ───────────────────────────────────────────────────────── */
.accent-panel {
  position: relative;
  background: var(--bg-panel);
  border-radius: var(--radius);
  min-height: 520px;
  overflow: hidden;
  display: flex;
  align-items: flex-end;
  padding: 2.5rem;
}

.accent-circle {
  position: absolute;
  border-radius: 50%;
  opacity: 0.18;
}
.accent-circle--1 {
  width: 320px; height: 320px;
  background: var(--accent);
  top: -60px; right: -80px;
}
.accent-circle--2 {
  width: 200px; height: 200px;
  background: #6a8fd8;
  bottom: 40px; left: -60px;
}

.accent-tagline {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  z-index: 1;
}
.tagline-word {
  font-family: var(--font-display);
  font-size: clamp(2.8rem, 5vw, 4rem);
  color: #ffffff;
  line-height: 1.1;
  letter-spacing: -0.02em;
}
.tagline-word:last-child { color: var(--accent); font-style: italic; }

/* ── Form card ──────────────────────────────────────────────────────────── */
.form-card {
  background: var(--bg-card);
  border-radius: var(--radius);
  padding: clamp(1.75rem, 4vw, 3rem);
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  transition: background 0.3s, border-color 0.3s;
}

.form-header { margin-bottom: 2rem; }

.form-title {
  font-family: var(--font-display);
  font-size: clamp(1.8rem, 3vw, 2.6rem);
  font-weight: 400;
  color: var(--text-primary);
  margin: 0 0 0.6rem;
  line-height: 1.15;
  letter-spacing: -0.02em;
}

.form-subtitle {
  font-size: 0.95rem;
  color: var(--text-muted);
  margin: 0;
  line-height: 1.6;
}

/* ── Form fields ────────────────────────────────────────────────────────── */
.form-body { display: flex; flex-direction: column; gap: 1.4rem; }

.field-group { display: flex; flex-direction: column; gap: 0.45rem; }

.field-label {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* Override PrimeVue input styling to match our tokens */
.field-input :deep(.p-inputtext),
.field-input:deep(textarea),
:deep(.field-input.p-inputtext),
:deep(.field-input textarea) {
  width: 100%;
  background: var(--bg) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.95rem !important;
  padding: 0.7rem 1rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  box-shadow: none !important;
}

:deep(.p-inputtext),
:deep(textarea.p-textarea) {
  width: 100%;
  background: var(--bg) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.95rem !important;
  padding: 0.7rem 1rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  box-shadow: none !important;
}

:deep(.p-inputtext:focus),
:deep(textarea.p-textarea:focus) {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent) !important;
  outline: none !important;
}

:deep(.p-invalid .p-inputtext),
:deep(.p-invalid),
:deep(.p-inputtext.p-invalid),
:deep(textarea.p-invalid) {
  border-color: var(--error) !important;
}

.field-error {
  font-size: 0.8rem;
  color: var(--error);
  display: flex;
  align-items: center;
  gap: 0.3rem;
}

/* ── Submit button ──────────────────────────────────────────────────────── */
:deep(.submit-btn.p-button) {
  background: var(--accent) !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: var(--font-body) !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  padding: 0.8rem 1.5rem !important;
  letter-spacing: 0.02em !important;
  transition: opacity 0.2s, transform 0.15s !important;
  box-shadow: 0 4px 16px color-mix(in srgb, var(--accent) 35%, transparent) !important;
  justify-content: center;
  width: 100%;
}

:deep(.submit-btn.p-button:hover:not(:disabled)) {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
}

:deep(.submit-btn.p-button:disabled) {
  opacity: 0.55 !important;
}
</style>