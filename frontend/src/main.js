import './style.css'
import App from './App.vue'
import { createApp } from 'vue'
import Button from 'primevue/button'
import PrimeVue from 'primevue/config'
import Aura from '@primevue/themes/aura'

const app = createApp(App)

// Use the new theme configuration
app.use(PrimeVue, {
    theme: {
        preset: Aura
    }
})

app.component('Button', Button)

app.mount('#app')