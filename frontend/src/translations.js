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
   footer: {
    navigationTitle: 'Navigation',
    teamTitle: 'Research Team',
    contactTitle: 'Contact',
    emailLabel: 'Contact Us',
    emailHref: 'mailto:your-email@example.com',
    githubLabel: 'Project Repository',
    githubHref: 'https://github.com/AbdulrahmanQht/Quantum-Hybrid-Defect-Detector',
    copy: '© 2026 Quantum-Hybrid Defect Detector. All rights reserved.',
    teamLinks: [
      { name: 'Fawaz altahini', linkedin: 'https://linkedin.com/in/your-link' },
      { name: 'Abdulrahman alqahtani', linkedin: 'https://www.linkedin.com/in/abdulrahmanqht/' },
      { name: 'Ali alhushayyish', linkedin: 'https://linkedin.com/in/your-link' },
      { name: 'Azzam alzahrani', linkedin: 'https://www.linkedin.com/in/azzam-alzahrani-52926b36b/' },
      { name: 'Talal altowijri', linkedin: 'https://linkedin.com/in/your-link' },
      { name: 'Omar almalki', linkedin: 'https://linkedin.com/in/your-link' }
    ]
  },
    home: {
  hero: {
    badge: 'Quantum-Classical Industrial AI',
    title: 'Quantum-Hybrid Defect Detector',
    subtitle: 'A stronger landing experience for industrial inspection and quantum-enhanced analysis.',
    description: 'Our hybrid quantum-classical system identifies corrosion, cracks, leaks, and structural defects in pipelines and industrial equipment with a design focused on clarity, credibility, and technical depth.',
        button: 'Start Classification',
        secondaryButton: 'View Benchmark',
    tertiaryButton: 'See Quantum Advantage'

  },
  quickLinks: [
    { label: 'Go to Classify', to: '/classify' },
    { label: 'Benchmark', to: '/benchmark' },
    { label: 'Quantum Advantage', to: '/quantum-advantage' }
  ],
  stats: [
    { value: 'Hybrid AI', label: 'Quantum + Classical' },
    { value: 'Industrial', label: 'Pipeline Defect Focus' },
    { value: 'Research-led', label: 'Academic Project' }
  ],
  slider: [
    { title: 'Project Overview', text: 'This project explores the integration of classical CNNs with Quantum Machine Learning to improve pipeline defect detection.' },
    { title: 'Hybrid Quantum Approach', text: 'A hybrid model with a quantum layer inside a classical CNN improves robustness, supports noisy conditions, and strengthens model behavior in difficult inspection settings.' },
    { title: 'Industrial Application', text: 'Designed for oil and gas inspection scenarios, the system targets practical defect categories such as corrosion, cracks, leaks, and structural anomalies.' }
  ],
  highlights: [
    {
      title: 'Inspection-Oriented Design',
      text: 'Built for a serious industrial use case instead of a generic ML demo, with a landing page that explains the actual value of the system.'
    },
    {
      title: 'Hybrid Quantum Pipeline',
      text: 'The project presents a clear quantum-classical architecture that supports experimentation, benchmarking, and comparison.'
    },
    {
      title: 'Usable Product Flow',
      text: 'Visitors can move directly to classification, benchmark results, or the quantum advantage page without friction.'
    }
  ],
  sections: {
    overviewEyebrow: 'Platform',
    overviewTitle: 'Built for modern defect detection',
    overviewText: 'This landing page positions the project as both a serious academic effort and a practical industrial system. It gives the user a clear path into the core product experience while preserving your existing visual identity.',
    sliderEyebrow: 'Highlights',
    highlightsEyebrow: 'Strengths',
    highlightsTitle: 'What this homepage should communicate'
  },
  team: {
    eyebrow: 'Team',
    title: 'Project Supervisors and Research Team',
    supervisorsTitle: 'Project Supervisors',
    researchersTitle: 'Research Team',
    github: 'GitHub',
    linkedin: 'LinkedIn',
    supervisors: [
      { name: 'Dr. Mustafa Youldash', role: 'Principal Supervisor' },
      { name: 'Dr. Naya Nagy', role: 'Co-Supervisor' }
    ],
    researchers: [
      {
        name: 'Fawaz altahini',
        role: 'Team Leader',
        github: 'https://github.com/Tafawaz',
        linkedin: 'https://linkedin.com/in/your-link'
      },
      {
        name: 'Abdulrahman alqahtani',
        role: 'Team Member',
        github: 'https://github.com/AbdulrahmanQht',
        linkedin: 'https://www.linkedin.com/in/abdulrahmanqht/'
      },
      {
        name: 'Ali alhushayyish',
        role: 'Team Member',
        github: 'https://github.com/AliHamad13',
        linkedin: 'https://linkedin.com/in/your-link'
      },
      {
        name: 'Azzam alzahrani',
        role: 'Team Member',
        github: 'https://github.com/xAzzamx',
        linkedin: 'https://www.linkedin.com/in/azzam-alzahrani-52926b36b/'
      },
      {
        name: 'Talal altowijri',
        role: 'Team Member',
        github: 'https://github.com/TalalAlt',
        linkedin: 'https://linkedin.com/in/your-link'
      },
      {
        name: 'Omar almalki',
        role: 'Team Member',
        github: 'https://github.com/your-link',
        linkedin: 'https://linkedin.com/in/your-link'
      }
    ]
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
      noisy_label: "Noisy",
      eyebrow: 'Quantum · Classical · Hybrid',
      models: {
        cnn: 'CNN',
        qnn_cpu: 'QNN-CPU',
        qnn_gpu: 'QNN-GPU'
      },
export_csv_success: 'Results exported to CSV successfully!',
export_json_success: 'Results exported to JSON successfully!',
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
      errorMsg: 'Failed to send message. Please try again later.',
      sideEyebrow: 'Contact',
  sideTitle: 'Let’s talk about the project.',
  sideText: 'Use the form to reach the team for questions, collaboration, or feedback about the Quantum-Hybrid Defect Detector.',
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
      footer: {
    navigationTitle: 'التنقل',
    teamTitle: 'الفريق البحثي',
    contactTitle: 'التواصل',
    emailLabel: 'تواصل معنا',
    emailHref: 'mailto:your-email@example.com',
    githubLabel: 'مستودع المشروع',
    githubHref: 'https://github.com/AbdulrahmanQht/Quantum-Hybrid-Defect-Detector',
    copy: '© 2026 كاشف العيوب الكمّي-الكلاسيكي. جميع الحقوق محفوظة.',
    teamLinks: [
      { name: 'فواز الطحيني', linkedin: 'https://linkedin.com/in/your-link' },
      { name: 'عبدالرحمن القحطاني', linkedin: 'https://linkedin.com/in/your-link' },
      { name: 'علي الحشيّش', linkedin: 'https://linkedin.com/in/your-link' },
      { name: 'عزام الزهراني', linkedin: 'https://linkedin.com/in/your-link' },
      { name: 'طلال النويجري', linkedin: 'https://linkedin.com/in/your-link' },
      { name: 'عمر المالكي', linkedin: 'https://linkedin.com/in/your-link' }
    ]
  },

    home: {
  hero: {
    badge: 'ذكاء صناعي كمّي-كلاسيكي',
    title: 'كاشف العيوب الكمّي-الكلاسيكي',
    subtitle: 'واجهة رئيسية أقوى للفحص الصناعي والتحليل المعزز بالحوسبة الكمّية.',
    description: 'نظامنا الهجين الكمّي-الكلاسيكي يكتشف التآكل والتشققات والتسربات والعيوب الهيكلية في خطوط الأنابيب والمعدات الصناعية ضمن تجربة أكثر وضوحاً واحترافية.',
    button: 'ابدأ التصنيف',
        secondaryButton: 'عرض المقارنة',
    tertiaryButton: 'استكشف التفوق الكمّي'
  },
  quickLinks: [
    { label: 'الانتقال إلى التصنيف', to: '/classify' },
    { label: 'المقارنة', to: '/benchmark' },
    { label: 'التفوق الكمّي', to: '/quantum-advantage' }
  ],
  stats: [
    { value: 'ذكاء هجين', label: 'كمّي + كلاسيكي' },
    { value: 'صناعي', label: 'يركز على عيوب الأنابيب' },
    { value: 'بحثي', label: 'مشروع أكاديمي' }
  ],
  slider: [
    { title: 'نظرة عامة على المشروع', text: 'يستكشف هذا المشروع دمج الشبكات العصبية التقليدية مع التعلم الآلي الكمّي لتحسين كشف العيوب في خطوط الأنابيب.' },
    { title: 'النهج الكمّي الهجين', text: 'يستخدم النموذج طبقة كمّية داخل شبكة عصبية تقليدية لتحسين المتانة ودعم الحالات المشوشة وتعزيز الأداء في ظروف الفحص الصعبة.' },
    { title: 'التطبيق الصناعي', text: 'تم تصميم النظام لسيناريوهات الفحص في قطاع النفط والغاز، مع التركيز على فئات العيوب العملية مثل التآكل والتشققات والتسربات والاضطرابات الهيكلية.' }
  ],
  highlights: [
    {
      title: 'تصميم موجه للفحص',
      text: 'الواجهة تعرض المشروع كنظام صناعي وبحثي فعلي بدلاً من كونه عرضاً عاماً لتعلم الآلة فقط.'
    },
    {
      title: 'مسار كمّي هجين',
      text: 'يعرض المشروع بنية واضحة تدمج بين الحوسبة التقليدية والكمّية وتدعم التجربة والمقارنة والشرح.'
    },
    {
      title: 'تدفق استخدام واضح',
      text: 'يمكن للزائر الانتقال مباشرة إلى التصنيف أو المقارنة أو صفحة التفوق الكمّي بسهولة.'
    }
  ],
  sections: {
    overviewEyebrow: 'المنصة',
    overviewTitle: 'مصمم لكشف العيوب الحديث',
    overviewText: 'تجعل هذه الواجهة الرئيسية المشروع يبدو كمجهود أكاديمي جاد ونظام صناعي عملي في الوقت نفسه، مع المحافظة على هويتكم البصرية الحالية.',
    sliderEyebrow: 'أبرز النقاط',
    highlightsEyebrow: 'نقاط القوة',
    highlightsTitle: 'ما الذي يجب أن توضحه هذه الصفحة'
  },
  team: {
    eyebrow: 'الفريق',
    title: 'المشرفون على المشروع والفريق البحثي',
    supervisorsTitle: 'المشرفون على المشروع',
    researchersTitle: 'الفريق البحثي',
    github: 'قيت هب',
    linkedin: 'لينكد إن',
    supervisors: [
      { name: 'Dr. Mustafa Youldash', role: 'المشرف الرئيسي' },
      { name: 'Dr. Naya Nagy', role: 'المشرف المشارك' }
    ],
    researchers: [
      {
        name: 'فواز الطحيني',
        role: 'قائد الفريق',
        github: 'https://github.com/Tafawaz',
        linkedin: 'https://linkedin.com/in/your-link'
      },
      {
        name: 'عبدالرحمن القحطاني',
        role: 'عضو فريق',
        github: 'https://github.com/AbdulrahmanQht',
        linkedin: 'https://www.linkedin.com/in/abdulrahmanqht/'
      },
      {
        name: 'علي الحشيّش',
        role: 'عضو فريق',
        github: 'https://github.com/AliHamad13',
        linkedin: 'https://linkedin.com/in/your-link'
      },
      {
        name: 'عزام الزهراني',
        role: 'عضو فريق',
        github: 'https://github.com/xAzzamx',
        linkedin: 'https://www.linkedin.com/in/azzam-alzahrani-52926b36b/'
      },
      {
        name: 'طلال النويجري',
        role: 'عضو فريق',
        github: 'https://github.com/TalalAlt',
        linkedin: 'https://linkedin.com/in/your-link'
      },
      {
        name: 'عمر المالكي',
        role: 'عضو فريق',
        github: 'https://github.com/your-link',
        linkedin: 'https://linkedin.com/in/your-link'
      }
    ]
  },
  credits: {
    value: 'قيمنا: إظهار التكامل بين النماذج الكمّية والكلاسيكية أكاديمياً، وتحسين دقة الفحص والموثوقية وتجربة الاستخدام عملياً.',
    team: 'بإشراف الدكتور مصطفى يولداش والدكتورة نايا ناجي. تم تطوير المشروع بواسطة فريق بحثي من جامعة الإمام عبدالرحمن بن فيصل.'
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
      noisy_label: "مشوش",
      eyebrow: 'كمّي · كلاسيكي · هجين',
      models: {
        cnn: 'CNN',
        qnn_cpu: 'QNN-CPU',
        qnn_gpu: 'QNN-GPU'
      },
export_csv_success: 'تم تصدير النتائج إلى CSV بنجاح!',
export_json_success: 'تم تصدير النتائج إلى JSON بنجاح!',
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
      errorMsg: 'فشل إرسال الرسالة. يرجى المحاولة مرة أخرى لاحقاً.',
      sideEyebrow: 'تواصل',
      
  sideTitle: 'لنتحدث عن المشروع.',
  sideText: 'استخدم النموذج للتواصل مع الفريق بخصوص الأسئلة أو التعاون أو الملاحظات حول كاشف العيوب الكمّي-الكلاسيكي.',

    }
  }
}

export const i18n = createI18n({
  legacy: false, 
  locale: Cookies.get('app_lang') || 'EN',
  fallbackLocale: 'EN',
  messages,
});