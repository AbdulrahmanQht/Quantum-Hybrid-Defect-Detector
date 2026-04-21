// src/i18n.js
import { createI18n } from 'vue-i18n';
import Cookies from 'js-cookie';

const messages = {
  EN: {
    navbar: {
      home: 'Home',
      classify: 'Classify',
      benchmark: 'Benchmark',
      quantum_advantage: "Quantum Advantage",
      about: 'About',
      contact: 'Contact'
    },
    home: {
      hero: {
        title: 'Quantum-Hybrid Defect Detector',
        subtitle: 'Revolutionizing Industrial Defect Detection',
        description: 'Our hybrid quantum-classical system identifies corrosion, cracks, and leaks in pipelines and industrial equipment with high accuracy and efficiency.',
        button: 'Start Classification'
      },
      slider: [
        { title: 'Project Overview', text: 'This project explores the integration of classical CNNs with Quantum Machine Learning to improve pipeline defect detection.' },
        { title: 'Hybrid Quantum Approach', text: 'A hybrid model with a quantum layer within a classical CNN enhances robustness, reduces noise sensitivity, and accelerates processing.' },
        { title: 'Industrial Application', text: 'Designed for the oil and gas industry, detecting defects such as corrosion, cracks, and leaks to improve safety and efficiency.' }
      ],
      credits: {
        value: 'Our values: Academically demonstrates hybrid quantum-classical integration. Practically improves inspection accuracy, noise robustness, and processing speed.',
        team: 'Supervised by Dr. Mustafa Youldash and Dr. Naya Nagy. Developed by a team of six students from Imam Abdulrahman Bin Faisal University.'
      }
    },
    classify: {
      title: 'Defect Detection',
      subtitle: 'Upload an industrial image to run inference across all three models simultaneously',
      dropzone: 'Drop image here or browse files',
      reset: 'Reset',
      run: 'Run Classification',
      validating: 'Validating...',
      running: 'Running Models...',
      max_size: 'Maximum file size is 5MB.',
      max_dimensions: 'Up to 4096×4096px.',
      choose: 'Choose Image',
      top_prediction: 'Top Prediction',
      results_defect: 'Defect Detected',
      "Deformation": "Deformation",
      "Deposition": "Deposition",
      "Disconnect": "Disconnect",
      "Misalignment": "Misalignment",
      "Obstacle": "Obstacle",
      "Rupture": "Rupture",
      model_comparison: 'Model Comparison',
      confidence_scores: 'Confidence Scores',
      inference_latency: 'Inference Latency',
      distribution: 'Confidence Distribution',
      export_title: 'Export Results',
      export_subtitle: 'Download classification results for further analysis',
      confidence: 'Confidence',
      latency: 'Latency',
      model: 'Model',
      prediction: 'Prediction',
      safe: 'SAFE',
      defect: 'DEFECT',
      err_format: 'Invalid format. Please upload a PNG, JPG, JPEG, or WEBP image.',
      err_size: 'File size exceeds 5MB. Please upload a smaller image.',
      err_spoof: 'File content does not match its extension.',
      err_upload: 'Upload failed. Please try again.',
      noise_toggle: "Inject Noise",
      noise_toggle_sub: "Simulate degraded inspection conditions",
      noise_severity: "Severity",
      noise_low: "Mild",
      noise_medium: "Degraded",
      noise_high: "Severe",
      noisy_results: "Noisy Inference",
      noisy_top_prediction: "Top Prediction Under Noise",
      noisy_label: "Noisy"
    },
    benchmark: {},
    quantum_advantage: {},
    contact: {
      pageTitle: 'Get in Touch',
      pageSubtitle: "We'd love to hear from you. Send us a message and we'll respond as soon as possible.",
      name: 'Your Name',
      namePlaceholder: 'John Doe',
      subject: 'Subject',
      subjectPlaceholder: 'How can we help?',
      message: 'Message',
      messagePlaceholder: 'Write your message here...',
      send: 'Send Message',
      sending: 'Opening...',
      required: 'This field is required',
      successTitle: 'Success',
      successMsg: 'Your message has been sent successfully.',
      errorTitle: 'Error',
      errorMsg: 'Failed to send message. Please try again later.'
    }
  },
  AR: {
    navbar: {
      home: 'الرئيسية',
      classify: 'تصنيف',
      benchmark: 'مقارنة',
      quantum_advantage: "تفوق الحوسبة الكمّية",
      about: 'من نحن',
      contact: 'اتصل بنا'
    },
    home: {
      hero: {
        title: 'كاشف العيوب الكمّي-الكلاسيكي',
        subtitle: ' تعريف عصري لكشف العيوب الصناعية',
        description: 'نظامنا الهجين الكمّي-الكلاسيكي يكتشف التآكل، التشققات، والعيوب في خطوط الأنابيب والمعدات الصناعية بدقة وكفاءة عالية',
        button: 'ابدأ التصنيف'
      },
      slider: [
        { title: 'نظرة عامة على المشروع', text: 'يستكشف هذا المشروع دمج الشبكات العصبية التقليدية مع التعلم الكمّي لتحسين كشف العيوب في خطوط الأنابيب' },
        { title: 'النهج الكمّي الهجين', text: 'يستخدم نموذج هجين طبقة كمّية ضمن شبكة عصبية تقليدية لتعزيز المتانة، تقليل الحساسية التشويش، وتسريع المعالجة' },
        { title: 'التطبيق الصناعي', text: 'مصمم لصناعات النفط والغاز، لاكتشاف عيوب الأنابيب مثل التآكل والتشققات لتحسين السلامة والكفاءة' }
      ],
      credits: {
        value: '.القيم: أكاديميا يوضح دمج النماذج الكمّية والكلاسيكية. عملياً يحسن دقة الفحص، مقاومة التشويش الصور ، وسرعة المعالجة',
        team:'.اشراف الدكتور مصطفى يولداش والدكتورة نايا ناجي. تم تطويره بواسطة فريق من ستة طلاب من جامعة الإمام عبدالرحمن بن فيصل'
      }
    },
    classify: {
      title: 'كشف العيوب',
      subtitle: 'قم بتحميل صورة صناعية لتشغيل الاستدلال عبر النماذج الثلاثة في وقت واحد',
      dropzone: 'أفلت الصورة هنا أو تصفح الملفات',
      reset: 'إعادة ضبط',
      run: 'تشغيل التصنيف',
      validating: 'جاري التحقق...',
      running: 'جاري تشغيل النماذج...',
      max_size: 'الحد الأقصى لحجم الملف هو 5 ميجابايت.',
      max_dimensions: 'حتى 4096×4096 بكسل.',
      choose: 'اختر صورة',
      top_prediction: 'التوقع الأفضل',
      results_defect: 'تم اكتشاف خلل',
      "Deformation": "تشوه",
      "Deposition": "ترسبات",
      "Disconnect": "انفصال",
      "Misalignment": "عدم محاذاة",
      "Obstacle": "عائق",
      "Rupture": "تمزق / كسر",
      model_comparison: 'مقارنة النماذج',
      confidence_scores: 'درجات الثقة',
      inference_latency: 'وقت الاستجابة',
      distribution: 'توزيع الثقة',
      export_title: 'تصدير النتائج',
      export_subtitle: 'تحميل نتائج التصنيف لمزيد من التحليل',
      confidence: 'الثقة',
      latency: 'وقت الاستجابة',
      model: 'النموذج',
      prediction: 'التوقع',
      safe: 'سليم',
      defect: 'خلل',
      err_format: 'صيغة غير صالحة. يرجى تحميل صورة PNG أو JPG أو WEBP.',
      err_size: 'حجم الملف يتجاوز 5 ميجابايت.',
      err_spoof: 'محتوى الملف لا يتطابق مع امتداده.',
      err_upload: 'فشل التحميل. يرجى المحاولة مرة أخرى.',
      noise_toggle: "تفعيل التشويش",
      noise_toggle_sub: "محاكاة ظروف الفحص الرديئة",
      noise_severity: "مستوى التشويش",
      noise_low: "منخفض",
      noise_medium: "متوسط",
      noise_high: "عالي",
      noisy_results: "نتائج مشوشة",
      noisy_top_prediction: "أفضل توقع مع التشويش",
      noisy_label: "مشوش"
    },
    benchmark: {},
    quantum_advantage: {},
    contact: {
      pageTitle: 'تواصل معنا',
      pageSubtitle: 'يسعدنا سماع رأيك. أرسل لنا رسالة وسنرد عليك في أقرب وقت ممكن.',
      name: 'اسمك',
      namePlaceholder: 'محمد أحمد',
      subject: 'الموضوع',
      subjectPlaceholder: 'كيف يمكننا مساعدتك؟',
      message: 'الرسالة',
      messagePlaceholder: 'اكتب رسالتك هنا...',
      send: 'إرسال الرسالة',
      sending: 'جاري الإرسال...',
      required: 'هذا الحقل مطلوب',
      successTitle: 'نجاح',
      successMsg: 'تم أرسال رسالتك بنجاح.',
      errorTitle: 'خطأ',
      errorMsg: 'فشل إرسال الرسالة. يرجى المحاولة مرة أخرى لاحقاً.'
    }
  }
}

export const i18n = createI18n({
  legacy: false, 
  locale: Cookies.get('app_lang') || 'EN',
  fallbackLocale: 'EN',
  messages,
});