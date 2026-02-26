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
import ProgressBar from 'primevue/progressbar';
import Divider from 'primevue/divider';
import Badge from 'primevue/badge';
import Panel from 'primevue/panel';
import Image from 'primevue/image';
import Chip from 'primevue/chip';
import Toast from 'primevue/toast';
import ToastService from 'primevue/toastservice';
import Skeleton from 'primevue/skeleton';
import InlineMessage from 'primevue/inlinemessage';

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
app.use(ToastService)

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
app.component('ProgressBar', ProgressBar)
app.component('Divider', Divider)
app.component('Badge', Badge)
app.component('Panel', Panel)
app.component('Image', Image)
app.component('Chip', Chip)
app.component('Toast', Toast)
app.component('Skeleton', Skeleton)
app.component('InlineMessage', InlineMessage)

app.use(router)
app.mount('#app')