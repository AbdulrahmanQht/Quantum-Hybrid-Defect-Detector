<script setup>
import { ref } from 'vue'

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref(null)
const loading = ref(false)
const error = ref(null)
const response = ref(null)

const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png']
const MAX_SIZE_MB = 5

function onFileChange(event) {
  error.value = null
  response.value = null
  const file = event.target.files?.[0]
  if (!file) return

  if (!ALLOWED_TYPES.includes(file.type)) {
    error.value = 'Invalid file type. Please upload a JPG or PNG.'
    return
  }

  if (file.size > MAX_SIZE_MB * 1024 * 1024) {
    error.value = `File exceeds ${MAX_SIZE_MB}MB limit.`
    return
  }

  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
}

async function uploadImage() {
  if (!selectedFile.value) return

  loading.value = true
  error.value = null
  response.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const res = await fetch('http://127.0.0.1:8000/api/classify', {
      method: 'POST',
      body: formData
      // Do NOT set Content-Type manually — the browser sets it with the boundary
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err?.detail || `Server error: ${res.status}`)
    }

    const data = await res.json()
    response.value = `Received by backend: ${JSON.stringify(data)}`
  } catch (err) {
    error.value = err.message || 'Upload failed.'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="flex flex-col items-center gap-4 p-8">

    <!-- File Input -->
    <input
      ref="fileInput"
      type="file"
      accept=".jpg,.jpeg,.png"
      class="hidden"
      @change="onFileChange"
    />

    <!-- Upload Button -->
    <Button label="Choose Image" icon="pi pi-upload" @click="fileInput.click()" />

    <!-- Selected filename -->
    <p v-if="selectedFile" class="text-sm text-gray-600">
      Selected: {{ selectedFile.name }}
    </p>

    <!-- Image Preview -->
    <img
      v-if="previewUrl"
      :src="previewUrl"
      alt="Preview"
      class="border rounded max-h-64"
    />

    <!-- Send Button -->
    <Button
      v-if="selectedFile"
      label="Send to Backend"
      icon="pi pi-send"
      :loading="loading"
      @click="uploadImage"
    />

    <!-- Error -->
    <Message v-if="error" severity="error" :closable="false">{{ error }}</Message>

    <!-- Success response -->
    <Message v-if="response" severity="success" :closable="false">
      {{ response }}
    </Message>

  </div>
</template>