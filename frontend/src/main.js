import './style.css';
import App from './App.vue';
import router from './router';
import { createApp } from 'vue';
import Cookies from 'js-cookie';
import 'primeicons/primeicons.css';

// --- PrimeVue Core & Themes ---
import PrimeVue from 'primevue/config';
import Aura from '@primevue/themes/aura';

// --- PrimeVue Components ---
import Button from 'primevue/button';
import Message from 'primevue/message';
import MenuBar from 'primevue/menubar';
import FileUpload from 'primevue/fileupload';
import Card from 'primevue/card';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Tag from 'primevue/tag';
import Chart from 'primevue/chart';

const app = createApp(App)

// --- Theme Initialization ---
const savedTheme = Cookies.get('theme')
if (savedTheme === 'dark') {
    document.documentElement.classList.add('p-dark')
}

app.config.globalProperties.$cookies = Cookies

// --- Initialize PrimeVue ---
app.use(PrimeVue, {
    theme: {
        preset: Aura,
        options: {
            darkModeSelector: '.p-dark',
        }
    }
})

// --- Register Components Globally ---
app.component('Button', Button)
app.component('Message', Message)
app.component('MenuBar', MenuBar)
app.component('FileUpload', FileUpload)
app.component('Card', Card)
app.component('DataTable', DataTable)
app.component('Column', Column)
app.component('Tag', Tag)
app.component('Chart', Chart)

app.use(router)
app.mount('#app')