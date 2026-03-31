import './style.css'
import App from './App.vue'
import { createApp } from 'vue'
import 'primeicons/primeicons.css'
import Button from 'primevue/button'
import MenuBar from 'primevue/menubar'
import PrimeVue from 'primevue/config'
import Aura from '@primevue/themes/aura'
import router from './router'
import Cookies from 'js-cookie'

const app = createApp(App)

const savedTheme = Cookies.get('theme')
if (savedTheme === 'dark') {
    document.documentElement.classList.add('p-dark')
}

app.config.globalProperties.$cookies = Cookies

app.use(PrimeVue, {
    theme: {
        preset: Aura,
        options: {
            darkModeSelector: '.p-dark',
        }
    }
})

app.component('Button', Button)
app.component('MenuBar', MenuBar)
app.use(router)
app.mount('#app')