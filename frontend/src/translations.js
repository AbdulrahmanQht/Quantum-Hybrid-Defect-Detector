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
      restored_hint: "Results restored from last session — reset or re-select image to run again."
    },
    benchmark: {},
    qa: {
      title: 'Quantum Advantage Report',
      subtitle: 'Empirical validation of quantum mechanisms in the hybrid architecture.',
      generatedAt: 'Generated',
      qubits: 'Qubits',
      depth: 'Depth',
      loading: 'Loading quantum metrics...',
      errorTitle: 'Data Unavailable',
      errorMsg: 'Failed to load quantum advantage data. Please ensure the backend is running.',
      model: 'Model',
      confirmed: 'Advantage Confirmed',
      notConfirmed: 'Not Confirmed',
      exp2: {
        title: 'Quantum Gain (Branch Ablation)',
        desc: 'How much accuracy does the quantum branch add over classical-only inference?',
        quantumGain: 'Quantum Gain',
        fullModel: 'Full Model',
        classicalOnly: 'Classical Only',
        quantumOnly: 'Quantum Only',
        tooltip: 'Quantum Gain = Full Accuracy − Classical Only Accuracy'
      },
      exp6: {
        title: 'Noise Robustness',
        desc: 'Does the quantum branch become more valuable as input quality degrades?',
        gaussian: 'Gaussian',
        blur: 'Blur',
        contrast: 'Contrast',
        salt_pepper: 'Salt & Pepper',
        insight: 'A rising quantum gain under increasing noise confirms the quantum branch improves robustness in degraded inspection conditions.'
      },
      exp1: {
        title: 'Feature Orthogonality',
        desc: 'Are the quantum and classical branches learning different things?',
        score: 'Cosine Similarity Score',
        target: 'Target: Near 0.0 (Orthogonal)',
        explanation: 'Very low similarity confirms the quantum branch learns unique, complementary features distinct from the classical branch.'
      },
      exp13: {
        title: 'Linear CKA',
        desc: 'Are the learned representations structurally different?',
        scale: '0 = orthogonal, 1 = identical',
        insight: 'Low CKA confirms the quantum branch learns complementary representational structure, invariant to rotation and isotropic scaling — a stronger claim than cosine similarity.'
      },
      exp3: {
        title: 'Re-upload Contribution',
        desc: 'How much does the data re-uploading technique improve accuracy?',
        with: 'With Re-upload',
        without: 'Without Re-upload',
        contribution: 'Contribution'
      },
      exp4: {
        title: 'Entanglement Entropy',
        desc: 'Is the quantum circuit generating real quantum correlations?',
        overallMean: 'Overall Mean Entropy'
      },
      exp5: {
        title: 'Gradient Variance (Barren Plateau Check)',
        desc: 'Can the quantum circuit still learn, or have gradients vanished?',
        target: 'Target Layer',
        meanVar: 'Mean Grad Variance',
        absMean: 'Abs Mean',
        batches: 'Batches'
      },
      exp7: {
        title: 'VQC Expressibility',
        desc: 'How well does the quantum circuit explore the available Hilbert space?',
        klDiv: 'KL Divergence from Haar',
        ref: 'Haar Reference',
        insight: 'Lower KL divergence means the circuit explores the Hilbert space more uniformly. Values below 0.05 indicate near-Haar expressibility.'
      },
      exp9: {
        title: 'Geometric Difference',
        desc: 'Does the quantum kernel span directions the classical kernel cannot?',
        insight: 'g > 1 means the quantum kernel spans feature directions the classical RBF kernel cannot represent (Huang et al. 2021). This is a rigorous, data-driven quantum advantage claim.'
      },
      exp8: {
        title: 'Kernel Target Alignment',
        desc: 'Which kernel better aligns with the classification labels?',
        quantum: 'KTA Quantum',
        classical: 'KTA Classical',
        diff: 'Difference',
        insight: 'KTA measures how well a kernel aligns with the label structure. A positive difference means the quantum kernel is better aligned to the task.'
      },
      exp10: {
        title: 'Fisher Effective Dimension',
        desc: 'How efficiently does each model use its parameters?',
        params: 'Parameters',
        dEff1000: 'd_eff (n=1000)',
        dEffPerParam: 'd_eff / param',
        insight: 'Higher d_eff per parameter means the model uses its parameters more efficiently. The QNN achieves comparable effective dimension with far fewer parameters.'
      },
      exp11: {
        title: 'Feature Effective Rank',
        desc: 'How much of the embedding space is each branch actually using?',
        classical: 'Classical',
        quantum: 'Quantum',
        insight: 'Utilisation = effective rank / embedding dim. The comparison shows how efficiently each branch uses its available dimensions.'
      },
      exp12: {
        title: 'Intrinsic Dimension',
        desc: 'How much information compression does each branch achieve?',
        classical: 'Classical (z)',
        quantum: 'Quantum (q_emb)',
        insight: 'Lower intrinsic dimension on q_emb vs z with competitive accuracy means the quantum circuit compresses class-relevant information more efficiently (TwoNN, Facco 2017).'
      },
      exp14: {
        title: 'Class Separability',
        desc: 'How well does each embedding separate the six defect classes?',
        advantage: 'Q. Advantage',
        insight: 'Fisher criterion J = tr(S_W⁻¹ S_B). Higher J means tighter within-class clusters and wider between-class margins.'
      },
      methodology: {
        title: 'Experiment Notes & Methodology'
      }
    },
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
      restored_hint: ".تم استعادة النتائج من الجلسة السابقة. أعد الاختيار أو انقر إعادة تعيين للتشغيل من جديد"

    },
    benchmark: {},
    qa: {
      title: 'تقرير التفوق الكمّي',
      subtitle: 'التحقق التجريبي من الآليات الكمّية في البنية الهجينة.',
      generatedAt: 'تاريخ التوليد',
      qubits: 'كيوبت',
      depth: 'العمق',
      loading: 'جاري تحميل المقاييس الكمّية...',
      errorTitle: 'البيانات غير متوفرة',
      errorMsg: 'فشل تحميل بيانات التفوق الكمّي. تأكد من تشغيل الخادم.',
      model: 'النموذج',
      confirmed: 'تفوق مؤكّد',
      notConfirmed: 'غير مؤكّد',
      exp2: {
        title: 'الكسب الكمّي (إزالة الفرع)',
        desc: 'كم تضيف الدقة عند استخدام الفرع الكمّي مقارنة بالفرع الكلاسيكي فقط؟',
        quantumGain: 'كسب كمّي',
        fullModel: 'النموذج الكامل',
        classicalOnly: 'كلاسيكي فقط',
        quantumOnly: 'كمّي فقط',
        tooltip: 'الكسب الكمّي = دقة النموذج الكامل − دقة الكلاسيكي فقط'
      },
      exp6: {
        title: 'المتانة ضد التشويش',
        desc: 'هل يزداد دور الفرع الكمّي عند تدهور جودة المدخلات؟',
        gaussian: 'ضوضاء غاوسية',
        blur: 'ضبابية',
        contrast: 'تباين',
        salt_pepper: 'ملح وفلفل',
        insight: 'ارتفاع الكسب الكمّي مع زيادة التشويش يؤكد أن الفرع الكمّي يحسّن المتانة في ظروف الفحص الصعبة.'
      },
      exp1: {
        title: 'تعامد الميزات',
        desc: 'هل يتعلم الفرعان الكمّي والكلاسيكي أشياء مختلفة؟',
        score: 'درجة تشابه جيب التمام',
        target: 'الهدف: قريب من 0.0 (متعامد)',
        explanation: 'التشابه المنخفض جداً يؤكد أن الفرع الكمّي يتعلم ميزات فريدة ومكملة مختلفة عن الفرع الكلاسيكي.'
      },
      exp13: {
        title: 'تحليل CKA الخطي',
        desc: 'هل التمثيلات المتعلّمة مختلفة هيكلياً؟',
        scale: '0 = متعامد، 1 = متطابق',
        insight: 'انخفاض CKA يؤكد أن الفرع الكمّي يتعلم بنية تمثيلية مكملة، وهذا أقوى من تشابه جيب التمام لأنه ثابت تحت الدوران والقياس.'
      },
      exp3: {
        title: 'مساهمة إعادة التحميل',
        desc: 'كم تحسّن تقنية إعادة تحميل البيانات من الدقة؟',
        with: 'مع إعادة التحميل',
        without: 'بدون إعادة التحميل',
        contribution: 'المساهمة'
      },
      exp4: {
        title: 'إنتروبيا التشابك',
        desc: 'هل تولّد الدائرة الكمّية ارتباطات كمّية حقيقية؟',
        overallMean: 'متوسط الإنتروبيا الكلي'
      },
      exp5: {
        title: 'تباين التدرج (فحص الهضبة)',
        desc: 'هل لا تزال الدائرة الكمّية قادرة على التعلم أم اختفت التدرجات؟',
        target: 'الطبقة المستهدفة',
        meanVar: 'متوسط تباين التدرج',
        absMean: 'المتوسط المطلق',
        batches: 'الدفعات'
      },
      exp7: {
        title: 'قابلية تعبير الدائرة الكمّية',
        desc: 'ما مدى تغطية الدائرة الكمّية لفضاء هيلبرت المتاح؟',
        klDiv: 'تباعد KL من هار',
        ref: 'مرجع هار',
        insight: 'انخفاض تباعد KL يعني أن الدائرة تستكشف فضاء هيلبرت بشكل أكثر انتظاماً. القيم أقل من 0.05 تشير إلى قابلية تعبير قريبة من هار.'
      },
      exp9: {
        title: 'الفرق الهندسي',
        desc: 'هل تمتد نواة الكمّية لاتجاهات لا تستطيع النواة الكلاسيكية تمثيلها؟',
        insight: 'g > 1 يعني أن النواة الكمّية تمتد لاتجاهات لا تستطيع نواة RBF الكلاسيكية تمثيلها (Huang et al. 2021). هذا إثبات صارم للتفوق الكمّي.'
      },
      exp8: {
        title: 'محاذاة النواة المستهدفة',
        desc: 'أي نواة أفضل في التوافق مع تصنيفات البيانات؟',
        quantum: 'KTA كمّي',
        classical: 'KTA كلاسيكي',
        diff: 'الفرق',
        insight: 'يقيس KTA مدى توافق النواة مع بنية التصنيفات. الفرق الموجب يعني أن النواة الكمّية أفضل توافقاً مع المهمة.'
      },
      exp10: {
        title: 'البعد الفعّال لفيشر',
        desc: 'ما مدى كفاءة كل نموذج في استخدام معاملاته؟',
        params: 'المعاملات',
        dEff1000: 'd_eff (n=1000)',
        dEffPerParam: 'd_eff / معامل',
        insight: 'ارتفاع d_eff لكل معامل يعني أن النموذج يستخدم معاملاته بكفاءة أعلى. يحقق QNN بُعداً فعّالاً مماثلاً بعدد معاملات أقل بكثير.'
      },
      exp11: {
        title: 'الرتبة الفعّالة للميزات',
        desc: 'كم من فضاء التضمين يستخدمه كل فرع فعلياً؟',
        classical: 'كلاسيكي',
        quantum: 'كمّي',
        insight: 'الاستخدام = الرتبة الفعّالة / بُعد التضمين. المقارنة توضح كفاءة كل فرع في استخدام أبعاده المتاحة.'
      },
      exp12: {
        title: 'البعد الجوهري',
        desc: 'ما مقدار ضغط المعلومات الذي يحققه كل فرع؟',
        classical: 'كلاسيكي (z)',
        quantum: 'كمّي (q_emb)',
        insight: 'انخفاض البعد الجوهري في q_emb مقارنة بـ z مع دقة تنافسية يعني أن الدائرة الكمّية تضغط المعلومات المتعلقة بالفئات بكفاءة أعلى.'
      },
      exp14: {
        title: 'قابلية فصل الفئات',
        desc: 'ما مدى فصل كل تضمين لفئات العيوب الست؟',
        advantage: 'تفوق كمّي',
        insight: 'معيار فيشر J = tr(S_W⁻¹ S_B). ارتفاع J يعني تجمعات أضيق داخل الفئة وفواصل أوسع بين الفئات.'
      },
      methodology: {
        title: 'ملاحظات المنهجية والتجارب'
      }
    },
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