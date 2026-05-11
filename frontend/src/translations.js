// src/i18n.js
import { createI18n } from "vue-i18n";
import Cookies from "js-cookie";

const messages = {
  EN: {
    notFound: {
      title: "Page Not Found",
      subtitle:
        "The route you're looking for collapsed into superposition — it exists neither here nor anywhere else.",
      whereLabel: "Where would you like to go?",
      cta: "Back to Home",
      footerNote: "Think something is broken?",
      footerLink: "Let us know",
    },
    navbar: {
      home: "Home",
      classify: "Classify",
      benchmark: "Benchmark",
      quantum_advantage: "Quantum Advantage",
      about: "About",
      contact: "Contact",
    },
    footer: {
      navigationTitle: "Navigation",
      teamTitle: "Research Team",
      contactTitle: "Contact",
      emailLabel: "Contact Us",
      emailHref: "mailto:your-email@example.com",
      githubLabel: "Project Repository",
      githubHref:
        "https://github.com/AbdulrahmanQht/Quantum-Hybrid-Defect-Detector",
      copy: "© 2026 Quantum-Hybrid Defect Detector. All rights reserved.",
      teamLinks: [
        {
          name: "Fawaz altahini",
          github: "https://github.com/Tafawaz",
        },
        {
          name: "Abdulrahman alqahtani",
          github: "https://github.com/AbdulrahmanQht",
        },
        {
          name: "Ali alhushayyish",
          github: "https://github.com/AliHamad13",
        },
        {
          name: "Azzam alzahrani",
          github: "https://github.com/xAzzamx",
        },
        {
          name: "Talal altowijri",
          github: "https://github.com/TalalAlt",
        },
        {
          name: "Omar almalki",
          github: "https://github.com/Sgingertig",
        },
      ],
    },
    home: {
      hero: {
        badge: "Quantum-Classical Industrial AI",
        title: "Quantum-Hybrid Defect Detector",
        subtitle:
          "Advanced industrial inspection powered by quantum-enhanced image analysis.",
        description:
          "A comparative research platform evaluating classical CNNs against Hybrid Quantum-Classical Neural Networks (HQCNNs) for detecting structural defects in industrial pipelines.",
        button: "Start Classification",
        secondaryButton: "View Benchmark",
        tertiaryButton: "See Quantum Advantage",
      },
      quickLinks: [
        { label: "Go to Classify", to: "/classify" },
        { label: "Benchmark", to: "/benchmark" },
        { label: "Quantum Advantage", to: "/quantum-advantage" },
      ],
      stats: [
        { value: "Hybrid AI", label: "Quantum + Classical" },
        { value: "Industrial", label: "Pipeline Defect Focus" },
        { value: "Research-led", label: "Academic Project" },
      ],
      slider: [
        {
          title: "Project Overview",
          text: "This project explores the integration of classical CNNs with Quantum Machine Learning to improve pipeline defect detection.",
        },
        {
          title: "Hybrid Quantum Approach",
          text: "A hybrid model with a quantum layer inside a classical CNN improves robustness, supports noisy conditions, and strengthens model behavior in difficult inspection settings.",
        },
        {
          title: "Industrial Application",
          text: "Designed for oil and gas inspection scenarios, the system targets practical defect categories such as corrosion, cracks, leaks, and structural anomalies.",
        },
      ],
      highlights: [
        {
          title: "Empirical Quantum Advantage",
          text: "Evaluating feature orthogonality, kernel target alignment, and barren plateau resilience to prove the quantum branch's mathematical contribution.",
        },
        {
          title: "Noise Robustness Analysis",
          text: "Rigorous stress-testing of models under various perturbations (Gaussian, Blur, Contrast) to validate quantum resilience in degraded environments.",
        },
        {
          title: "Comprehensive Benchmarking",
          text: "Side-by-side evaluation of inference latency, confidence calibration, and class separability between standard and quantum-enhanced models.",
        },
      ],
      sections: {
        overviewEyebrow: "Research Context",
        overviewTitle: "Bridging Quantum Computing and Industrial AI",
        overviewText:
          "This project addresses the critical need for reliable automated defect detection in oil and gas infrastructure. By integrating Variational Quantum Circuits (VQCs) into classical pipelines, we explore whether quantum embeddings can extract richer geometric features and maintain higher accuracy under severe industrial noise.",
        sliderEyebrow: "Highlights",
        highlightsEyebrow: "Strengths",
        highlightsTitle: "Core Research Objectives",
      },
      team: {
        eyebrow: "Team",
        title: "Project Supervisors and Research Team",
        supervisorsTitle: "Project Supervisors",
        researchersTitle: "Research Team",
        github: "GitHub",
        linkedin: "LinkedIn",
        supervisors: [
          { name: "Dr. Mustafa Youldash", role: "Principal Supervisor" },
          { name: "Dr. Naya Nagy", role: "Co-Supervisor" },
        ],
        researchers: [
          {
            name: "Fawaz altahini",
            role: "Team Leader",
            github: "https://github.com/Tafawaz",
            linkedin: "https://linkedin.com/in/your-link",
          },
          {
            name: "Abdulrahman alqahtani",
            role: "Team Member",
            github: "https://github.com/AbdulrahmanQht",
            linkedin: "https://www.linkedin.com/in/abdulrahmanqht/",
          },
          {
            name: "Ali alhushayyish",
            role: "Team Member",
            github: "https://github.com/AliHamad13",
            linkedin: "https://linkedin.com/in/your-link",
          },
          {
            name: "Azzam alzahrani",
            role: "Team Member",
            github: "https://github.com/xAzzamx",
            linkedin: "https://www.linkedin.com/in/azzam-alzahrani-52926b36b/",
          },
          {
            name: "Talal altowijri",
            role: "Team Member",
            github: "https://github.com/TalalAlt",
            linkedin: "https://linkedin.com/in/your-link",
          },
          {
            name: "Omar almalki",
            role: "Team Member",
            github: "https://github.com/Sgingertig",
            linkedin: "https://linkedin.com/in/your-link",
          },
        ],
      },
    },
    classify: {
      title: "Defect Detection",
      subtitle:
        "Upload an industrial image to run inference across all three models simultaneously",
      dropzone: "Drop image here or browse files",
      reset: "Reset",
      run: "Run Classification",
      validating: "Validating...",
      running: "Running Models...",
      max_size: "Maximum file size is 5MB.",
      max_dimensions: "Up to 4096×4096px.",
      choose: "Choose Image",
      top_prediction: "Top Prediction",
      results_defect: "Defect Detected",
      Deformation: "Deformation",
      Deposition: "Deposition",
      Disconnect: "Disconnect",
      Misalignment: "Misalignment",
      Obstacle: "Obstacle",
      Rupture: "Rupture",
      model_comparison: "Model Comparison",
      confidence_scores: "Confidence Scores",
      inference_latency: "Inference Latency",
      distribution: "Confidence Distribution",
      export_title: "Export Results",
      export_subtitle: "Download classification results for further analysis",
      confidence: "Confidence",
      confidence_percent: "Confidence (%)",
      latency: "Latency",
      latency_ms: "Latency (ms)",
      model: "Model",
      prediction: "Prediction",
      safe: "SAFE",
      defect: "DEFECT",
      err_format:
        "Invalid format. Please upload a PNG, JPG, JPEG, or WEBP image.",
      err_size: "File size exceeds 5MB. Please upload a smaller image.",
      err_spoof: "File content does not match its extension.",
      err_upload: "Upload failed. Please try again.",
      clean: "Clean",
      noisy: "Noisy",
      noise_toggle: "Inject Noise",
      noise_toggle_sub: "Simulate degraded inspection conditions",
      noise_severity: "Severity",
      noise_low: "Mild",
      noise_medium: "Degraded",
      noise_high: "Severe",
      noisy_results: "Noisy Inference",
      noisy_top_prediction: "Top Prediction Under Noise",
      noisy_label: "Noisy Label (Conf.)",
      compare_label: "Compare",
      clean_label: "Clean Label (Conf.)",
      noisy_confidence_scores: "Noisy Confidence Scores",
      noisy_inference_latency: "Noisy Inference Latency",
      noisy_distribution: "Noisy Confidence Distribution",
      clean_vs_noisy: "Clean vs Noisy",
      clean_noisy_delta: "Clean vs Noisy Delta",
      delta_confidence: "Confidence Δ",
      delta_latency: "Latency Δ",
      eyebrow: "Classical · Quantum Hybrid (CPU) · Quantum Hybrid (GPU)",
      noisy_pending: "Run the model to see the noisy preview",
      models: {
        cnn: "CNN",
        qnn_cpu: "QNN CPU",
        qnn_gpu: "QNN GPU",
      },
      export_csv_success: "Results exported to CSV successfully!",
      export_json_success: "Results exported to JSON successfully!",
      restored_hint:
        "Results restored from last session — reset or re-select image to run again.",
      prediction_match: "Verdict",
      match: "Match",
      mismatch: "Mismatch",
      tooltips: {
        model_comparison:
          "Side-by-side comparison of all models. CNN is the classical baseline; QNN CPU and QNN GPU are hybrid quantum-classical variants trained on the same dataset.",
        confidence_scores:
          "Each bar shows how certain a model is about its top prediction. Higher confidence is not always correct — a high-confidence wrong answer is a calibration failure, not a strength.",
        inference_latency:
          "Time taken by each model to process the image and return a prediction, measured in milliseconds. Lower values mean faster inference.",
        distribution:
          "Confidence share across models for the predicted class. A large slice toward one model means that model is most certain about its prediction.",
        noisy_confidence:
          "Under severe noise, CNN may report higher confidence than the quantum models but still predict incorrectly. This is a known overconfidence artifact — see the Benchmark page for evidence.",
        noisy_top_prediction:
          "The highest-confidence model under noise. CNN confidence under noise can be misleading — our benchmark shows CNN can be wrong at high confidence levels under severe noise conditions.",
        clean_vs_noisy:
          "Side-by-side confidence bars before and after noise injection. A drop in quantum model confidence shows better calibration; CNN often holds high confidence even when its prediction is incorrect.",
        clean_noisy_delta:
          "Δ (delta) is the change in each metric after noise injection. A positive CNN confidence delta under severe noise may reflect overconfidence rather than a genuine performance improvement.",
        clean_image:
          "The 384x384 preprocessed tensor fed directly into the models.",
        noisy_image:
          "Simulates industrial camera degradation using combined Gaussian, Motion Blur, Contrast reduction, and Lens Occlusion algorithms.",
      },
      cnn_overconfidence_warning:
        "Under severe noise, CNN may show the highest confidence here — but our benchmark results confirm it can still predict incorrectly at high confidence levels. Do not interpret high CNN confidence as superior reliability under noise.",
    },
    benchmark: {
      hero: {
        badge: "Research Benchmark",
        title: "Benchmarking the hybrid defect detection stack",
        subtitle:
          "Evaluation across clean accuracy, noise robustness, latency, and class diagnostics.",
        description:
          "This benchmark page compares the CNN baseline with the QNN CPU and QNN GPU variants using the same dataset setup, giving a consistent view of reliability and deployment tradeoffs.",
        test_samples: "Test Samples",
        fault_classes: "Fault Classes",
        models_evaluated: "Models Evaluated",
        accuracy_overview: "Accuracy Overview",
        loaded: "Results Loaded",
        live: "Live",
      },
      dataset: {
        eyebrow: "Dataset",
        fault_classes: "Fault Classes",
        title: "Dataset description and representative samples",
        description:
          "The benchmark starts with a visual snapshot of the inspection dataset. Each carousel slide shows one real sample from a defect class used during evaluation.",
        carousel_badge: "Sample Carousel",
        source_label: "Source:",
        source_name: "Kaggle Pipeline Defect Dataset",
        source_url:
          "https://www.kaggle.com/datasets/simplexitypipeline/pipeline-defect-dataset/data",
        transform_title: "Classification conversion:",
        transform_text:
          "The original dataset was designed for image detection, then transformed into a classification dataset by keeping only images whose label file contained exactly one unique class.",
        pipeline_title: "Dataset preparation workflow",
        facts: [
          { label: "Split", value: "Benchmark Test Set" },
          { label: "Domain", value: "Industrial Pipeline Defects" },
          { label: "Coverage", value: "6 Defect Classes" },
          { label: "Use", value: "Comparison + Diagnostics" },
        ],
        pipeline_steps: [
          "Read all label files and verify that the corresponding image file exists.",
          "Keep a sample only when its label file contains exactly one unique class ID, then store the image and label under that class.",
          "Discard any sample whose label file is empty or contains multiple unique class IDs by excluding it from the exported dataset.",
          "Split the remaining valid samples into train, validation, and test sets using a 70% / 15% / 15% ratio.",
          "Copy the valid images and their matching label files into a new directory structure grouped by split and class name.",
        ],
        samples: [
          {
            key: "Deformation",
            label: "Deformation",
            title: "Deformation sample",
            text: "A representative deformation case from the benchmark split used to stress shape-sensitive features.",
          },
          {
            key: "Deposition",
            label: "Deposition",
            title: "Deposition sample",
            text: "A sample showing accumulated material patterns that challenge texture-focused feature extraction.",
          },
          {
            key: "Disconnect",
            label: "Disconnect",
            title: "Disconnect sample",
            text: "A discontinuity example used to evaluate how clearly each model separates structural breaks.",
          },
          {
            key: "Misalignment",
            label: "Misalignment",
            title: "Misalignment sample",
            text: "A spatial shift example that tests sensitivity to positional inconsistency and geometric cues.",
          },
          {
            key: "Obstacle",
            label: "Obstacle",
            title: "Obstacle sample",
            text: "A sample containing obstructive interference that can reduce visual clarity in inspection scenes.",
          },
          {
            key: "Rupture",
            label: "Rupture",
            title: "Rupture sample",
            text: "A rupture case used to measure how reliably each model identifies severe structural failure.",
          },
        ],
      },
      summary: {
        batch_size: "Batch Size",
        epochs: "Training Epochs",
        qubits: "Qubits",
        q_depth: "Quantum Depth",
        resolution: "Image Resolution",
        device: "Device",
      },
      performance: {
        eyebrow: "Performance",
        title: "Clean performance snapshot",
        description:
          "Clean-data performance metrics across all three model configurations, including accuracy, weighted F1-score, and latency.",
        clean_accuracy: "Clean Accuracy",
        weighted_f1: "Weighted F1",
        latency: "Inference Latency",
        samples: "Evaluated Samples",
        best_accuracy: "Best Accuracy",
        fastest: "Fastest",
        benchmark: "Benchmark",
      },
      robustness: {
        eyebrow: "Robustness",
        title: "Noise robustness analysis",
        description:
          "Switch between supported perturbations to compare how each model behaves as inspection conditions degrade.",
        filter_label: "Noise Type",
        points: "points",
        level: "Level",
        gaussian: "Gaussian",
        blur: "Blur",
        contrast: "Contrast",
        salt_pepper: "Salt & Pepper",
        motion_blur: "Motion Blur",
        jpeg_compression: "JPEG Compression",
        lens_occlusion: "Lens Occlusion",
      },
      latency: {
        eyebrow: "Latency",
        title: "Inference speed",
        description:
          "Inference latency measures how long a model takes to process a single input, with lower values indicating faster and more efficient performance.",
      },
      diagnostics: {
        eyebrow: "Diagnostics",
        title: "Model diagnostics",
        description:
          "Inspect the confusion matrix and per-class metrics for the selected model using the shared benchmark payload.",
        model: "Model",
        actual: "Actual",
        confusion_matrix: "Confusion Matrix",
        per_class_metrics: "Per-Class Metrics",
        class: "Class",
        precision: "Precision",
        recall: "Recall",
        f1: "F1",
        support: "Support",
        average: "Average",
      },
      config: {
        eyebrow: "Configuration",
        title: "Benchmark configuration snapshot",
      },
      states: {
        loading: "Loading benchmark results...",
        error: "Failed to load benchmark data.",
        empty: "No benchmark data is available.",
      },
      models: {
        CNN: "CNN",
        QNN_CPU: "QNN CPU",
        QNN_GPU: "QNN GPU",
      },
      charts: {
        efficiency_title: "Efficiency Frontier",
        efficiency_desc: "Accuracy vs. Latency trade-off analysis.",
        radar_desc: "Multi-metric comparison across models.",
        robustness_curve: "Noise Robustness Curve",
        reliability_desc: "Comparison of accuracy vs. model confidence.",
        latency_axis: "Latency (ms)",
        accuracy_axis: "Accuracy (%)",
        confidence_label: "Model Mean Confidence",
        actual_acc_label: "Actual Accuracy",
        radar_title: "Multi-Metric Model Comparison",
        radar_sub:
          "Accuracy, F1, Precision, Recall and Noise Robustness across all three models",
        scatter_title: "Accuracy vs Inference Latency",
        scatter_sub:
          "One point per model — higher accuracy and lower latency is better",
        robustness_title: "Accuracy Under Increasing Noise",
        robustness_sub: "How each model degrades as noise intensity grows",
        reliability_title: "Reliability Curve",
        reliability_sub:
          "Actual accuracy vs mean confidence at each noise level for the selected model",
        radar_tooltip:
          "Each axis is a metric. A larger covered area means a better overall model profile across all measured dimensions.",
        scatter_tooltip:
          "Each point is one model. The ideal position is top-left — highest accuracy with the lowest inference latency.",
        robustness_tooltip:
          "Shows how each model's accuracy drops as noise severity increases. A flatter line means the model is more noise-robust.",
        reliability_tooltip:
          "Plots actual accuracy against average model confidence at each noise level. A well-calibrated model's line hugs the diagonal — overconfident models sit above it.",
      },
      metrics: {
        overall_maun: "Overall MAUN",
        mean_conf: "Mean Confidence",
        robustness_score: "Robustness Score",
        accuracy: "Accuracy",
        f1: "F1 Score",
        precision: "Precision",
        recall: "Recall",
        accuracy_tooltip:
          "Percentage of test images correctly classified across all six defect categories on clean (unperturbed) data.",
        f1_tooltip:
          "Harmonic mean of Precision and Recall. Accounts for both false positives and false negatives — a balanced quality measure.",
        maun_tooltip:
          "Mean Accuracy Under Noise — average accuracy across all noise types and severity levels evaluated in the benchmark.",
        robustness_score_tooltip:
          "Composite score reflecting how consistently the model maintains accuracy under all tested noise conditions.",
        mean_conf_tooltip:
          "Average prediction confidence across all test samples. Compare with actual accuracy to detect overconfidence.",
      },
    },
    qa: {
      title: "Quantum Advantage Report",
      subtitle:
        "Empirical validation of quantum mechanisms in the hybrid architecture.",
      generatedAt: "Generated",
      qubits: "Qubits",
      depth: "Depth",
      loading: "Loading quantum metrics...",
      errorTitle: "Data Unavailable",
      errorMsg:
        "Failed to load quantum advantage data. Please ensure the backend is running.",
      model: "Model",
      confirmed: "Advantage Confirmed",
      notConfirmed: "Not Confirmed",
      exp2: {
        title: "Quantum Gain (Branch Ablation)",
        desc: "How much accuracy does the quantum branch add over classical-only inference?",
        quantumGain: "Quantum Gain",
        fullModel: "Full Model",
        classicalOnly: "Classical Only",
        quantumOnly: "Quantum Only",
        tooltip: "Quantum Gain = Full Accuracy − Classical Only Accuracy",
      },
      exp6: {
        title: "Noise Robustness",
        desc: "Does the quantum branch become more valuable as input quality degrades?",
        gaussian: "Gaussian",
        blur: "Blur",
        contrast: "Contrast",
        salt_pepper: "Salt & Pepper",
        motion_blur: "Motion Blur",
        jpeg_compression: "JPEG Compression",
        lens_occlusion: "Lens Occlusion",
        insight:
          "A rising quantum gain under increasing noise confirms the quantum branch improves robustness in degraded inspection conditions.",
      },
      exp1: {
        title: "Feature Orthogonality",
        desc: "Are the quantum and classical branches learning different things?",
        score: "Cosine Similarity Score",
        target: "Target: Near 0.0 (Orthogonal)",
        explanation:
          "Very low similarity confirms the quantum branch learns unique, complementary features distinct from the classical branch.",
        tooltip:
          "Cosine similarity measures the angle between two feature vectors. A value near 0.0 means the quantum and classical branches are nearly orthogonal — they are learning different information, which is the desired outcome.",
      },
      exp13: {
        title: "Linear CKA",
        desc: "Are the learned representations structurally different?",
        scale: "0 = orthogonal, 1 = identical",
        insight:
          "Low CKA confirms the quantum branch learns complementary representational structure, invariant to rotation and isotropic scaling — a stronger claim than cosine similarity.",
        tooltip:
          "Linear CKA measures structural similarity between learned representations. Lower values indicate the quantum branch learns unique, orthogonal features.",
      },
      exp3: {
        title: "Re-upload Contribution",
        desc: "How much does the data re-uploading technique improve accuracy?",
        with: "With Re-upload",
        without: "Without Re-upload",
        contribution: "Contribution",
        tooltip:
          "Data re-uploading encodes classical inputs into the quantum circuit multiple times across different layers. This gives the circuit greater expressive power — the contribution metric shows the accuracy gain this technique alone provides.",
      },
      exp4: {
        title: "Entanglement Entropy",
        desc: "Is the quantum circuit generating real quantum correlations?",
        overallMean: "Overall Mean Entropy",
        tooltip:
          "Von Neumann entropy measures quantum entanglement per qubit. A value above 0 means the circuit is generating real quantum correlations between qubits — not just classical mixing. Higher entropy indicates richer quantum information processing.",
      },
      exp5: {
        title: "Gradient Variance (Barren Plateau Check)",
        desc: "Can the quantum circuit still learn, or have gradients vanished?",
        target: "Target Layer",
        meanVar: "Mean Grad Variance",
        absMean: "Abs Mean",
        batches: "Batches",
        tooltip:
          "Barren plateaus are regions where gradients vanish exponentially, making quantum circuits untrainable. A non-zero gradient variance confirms the quantum circuit is actively learning and has not fallen into this failure mode.",
      },
      exp7: {
        title: "VQC Expressibility",
        desc: "How well does the quantum circuit explore the available Hilbert space?",
        klDiv: "KL Divergence from Haar",
        ref: "Haar Reference",
        insight:
          "Lower KL divergence means the circuit explores the Hilbert space more uniformly. Values below 0.05 indicate near-Haar expressibility.",
        tooltip:
          "A Haar-random circuit samples the full quantum state space uniformly. KL divergence from Haar measures how close our trained circuit comes to that ideal coverage. Lower values mean the circuit can represent a wider variety of quantum states.",
      },
      exp9: {
        title: "Geometric Difference",
        desc: "Does the quantum kernel span directions the classical kernel cannot?",
        insight:
          "g > 1 means the quantum kernel spans feature directions the classical RBF kernel cannot represent (Huang et al. 2021). This is a rigorous, data-driven quantum advantage claim.",
        tooltip:
          "The geometric difference g quantifies how much extra feature space the quantum kernel covers compared to a classical RBF kernel. g > 1 is a formal, data-driven proof of quantum advantage, as defined by Huang et al. (2021).",
      },
      exp8: {
        title: "Kernel Target Alignment",
        desc: "Which kernel better aligns with the classification labels?",
        quantum: "KTA Quantum",
        classical: "KTA Classical",
        diff: "Difference",
        insight:
          "KTA measures how well a kernel aligns with the label structure. A positive difference means the quantum kernel is better aligned to the task.",
        tooltip:
          "Kernel Target Alignment (KTA) measures how well a kernel's similarity structure matches the ground-truth labels. A higher KTA means the kernel naturally groups similar classes together, making classification easier. A positive quantum–classical difference means the quantum kernel is better suited to this specific task.",
      },
      exp10: {
        title: "Fisher Effective Dimension",
        desc: "How efficiently does each model use its parameters?",
        params: "Parameters",
        dEff1000: "d_eff (n=1000)",
        dEffPerParam: "d_eff / param",
        insight:
          "Higher d_eff per parameter means the model uses its parameters more efficiently. The QNN achieves comparable effective dimension with far fewer parameters.",
        tooltip:
          "Effective dimension (d_eff) from Fisher Information Matrix analysis measures how many of a model's parameters are meaningfully contributing to its predictions on this dataset. d_eff per parameter normalises this by model size — a higher ratio means each parameter is doing more useful work.",
      },
      exp11: {
        title: "Feature Effective Rank",
        desc: "How much of the embedding space is each branch actually using?",
        classical: "Classical",
        quantum: "Quantum",
        insight:
          "Utilisation = effective rank / embedding dim. The comparison shows how efficiently each branch uses its available dimensions.",
        tooltip:
          "Effective rank measures how many embedding dimensions carry meaningful signal (vs. noise or redundancy). A high utilisation percentage means the branch is making full use of its available representational capacity rather than wasting dimensions.",
      },
      exp12: {
        title: "Intrinsic Dimension",
        desc: "How much information compression does each branch achieve?",
        classical: "Classical (z)",
        quantum: "Quantum (q_emb)",
        insight:
          "Lower intrinsic dimension on q_emb vs z with competitive accuracy means the quantum circuit compresses class-relevant information more efficiently (TwoNN, Facco 2017).",
        tooltip:
          "Intrinsic dimension estimates the true degrees of freedom needed to describe the data distribution in each embedding space, using the TwoNN estimator (Facco 2017). A lower intrinsic dimension with similar accuracy means the branch has found a more compact and efficient representation of the relevant information.",
      },
      exp14: {
        title: "Class Separability",
        desc: "How well does each embedding separate the six defect classes?",
        advantage: "Q. Advantage",
        insight:
          "Fisher criterion J = tr(S_W⁻¹ S_B). Higher J means tighter within-class clusters and wider between-class margins.",
        tooltip:
          "The Fisher criterion J measures class separability: it compares how spread out classes are from each other (between-class scatter S_B) versus how spread out samples are within each class (within-class scatter S_W). A higher J means the embedding is forming tighter, more distinct clusters per defect type.",
      },
      methodology: {
        title: "Methodology Notes",
        notes: {
          sample_size: "Sample Size",
          train_test_split: "Train / Test Split",
          augmentation: "Augmentation",
          optimizer: "Optimizer",
          loss_function: "Loss Function",
          epochs: "Epochs",
          batch_size: "Batch Size",
          learning_rate: "Learning Rate",
          quantum_backend: "Quantum Backend",
          n_qubits: "Number of Qubits",
          q_depth: "Circuit Depth",
          entanglement_entropy: "Entanglement Entropy",
          expressibility: "Expressibility",
          kernel_experiments: "Kernel Experiments",
          fim: "Fisher Information Matrix",
          parameter_matched_ablation: "Parameter-Matched Ablation",
        },
      },
    },
    contact: {
      pageTitle: "Get in Touch",
      pageSubtitle:
        "We'd love to hear from you. Send us a message and we'll respond as soon as possible.",
      name: "Your Name",
      namePlaceholder: "John Doe",
      subject: "Subject",
      subjectPlaceholder: "How can we help?",
      message: "Message",
      messagePlaceholder: "Write your message here...",
      send: "Send Message",
      sending: "Sending...",
      required: "This field is required",
      successTitle: "Success",
      successMsg: "Your message has been sent successfully.",
      errorTitle: "Failed to Send",
      errorMsg: "Failed to send message. Please try again later.",
      sideEyebrow: "Contact",
      sideTitle: "Let’s talk about the project.",
      sideText:
        "Use the form to reach the team for questions, collaboration, or feedback about the Quantum-Hybrid Defect Detector.",
      clear: "Clear current input",
    },
  },
  AR: {
    notFound: {
      title: "الصفحة غير موجودة",
      subtitle:
        "المسار الذي تبحث عنه انهار في تراكب كمومي — لا وجود له هنا ولا في أي مكان آخر.",
      whereLabel: "إلى أين تريد الذهاب؟",
      cta: "العودة إلى الرئيسية",
      footerNote: "هل تعتقد أن هناك خطأ ما؟",
      footerLink: "أخبرنا",
    },
    navbar: {
      home: "الرئيسية",
      classify: "تصنيف",
      benchmark: "مقارنة",
      quantum_advantage: "تفوق الحوسبة الكمّية",
      about: "من نحن",
      contact: "اتصل بنا",
    },
    footer: {
      navigationTitle: "التنقل",
      teamTitle: "الفريق البحثي",
      contactTitle: "التواصل",
      emailLabel: "تواصل معنا",
      emailHref: "mailto:your-email@example.com",
      githubLabel: "مستودع المشروع",
      githubHref:
        "https://github.com/AbdulrahmanQht/Quantum-Hybrid-Defect-Detector",
      copy: "© 2026 كاشف العيوب الكمّي-الكلاسيكي. جميع الحقوق محفوظة.",
      teamLinks: [
        {
          name: "فواز الطحيني",
          github: "https://github.com/Tafawaz",
        },
        {
          name: "عبدالرحمن القحطاني",
          github: "https://github.com/AbdulrahmanQht",
        },
        {
          name: "علي الحشيّش",
          github: "https://github.com/AliHamad13",
        },
        {
          name: "عزام الزهراني",
          github: "https://github.com/xAzzamx",
        },
        {
          name: "طلال التويجري",
          github: "https://github.com/TalalAlt",
        },
        {
          name: "عمر المالكي",
          github: "https://github.com/Sgingertig",
        },
      ],
    },
    home: {
      hero: {
        badge: "ذكاء صناعي كمّي-كلاسيكي",
        title: "كاشف العيوب الكمّي-الكلاسيكي",
        subtitle: "فحص صناعي متقدم مدعوم بتحليل الصور المعزز بالحوسبة الكمّية.",
        description:
          "منصة بحثية مقارنة لتقييم الشبكات العصبية الالتفافية (CNNs) مقابل الشبكات العصبية الكمّية-الكلاسيكية الهجينة (HQCNNs) في كشف العيوب الهيكلية لخطوط الأنابيب الصناعية.",
        button: "ابدأ التصنيف",
        secondaryButton: "عرض المقارنة",
        tertiaryButton: "استكشف التفوق الكمّي",
      },
      quickLinks: [
        { label: "الانتقال إلى التصنيف", to: "/classify" },
        { label: "المقارنة", to: "/benchmark" },
        { label: "التفوق الكمّي", to: "/quantum-advantage" },
      ],
      stats: [
        { value: "ذكاء هجين", label: "كمّي + كلاسيكي" },
        { value: "صناعي", label: "يركز على عيوب الأنابيب" },
        { value: "بحثي", label: "مشروع أكاديمي" },
      ],
      slider: [
        {
          title: "نظرة عامة على المشروع",
          text: "يستكشف هذا المشروع دمج الشبكات العصبية التقليدية مع التعلم الآلي الكمّي لتحسين كشف العيوب في خطوط الأنابيب.",
        },
        {
          title: "النهج الكمّي الهجين",
          text: "يستخدم النموذج طبقة كمّية داخل شبكة عصبية تقليدية لتحسين المتانة ودعم الحالات المشوشة وتعزيز الأداء في ظروف الفحص الصعبة.",
        },
        {
          title: "التطبيق الصناعي",
          text: "تم تصميم النظام لسيناريوهات الفحص في قطاع النفط والغاز، مع التركيز على فئات العيوب العملية مثل التآكل والتشققات والتسربات والاضطرابات الهيكلية.",
        },
      ],
      highlights: [
        {
          title: "ميزة كمّية تجريبية",
          text: "تقييم تعامد الميزات، ومحاذاة النواة مع الهدف، ومقاومة الهضاب الجرداء لإثبات المساهمة الرياضية للفرع الكمّي.",
        },
        {
          title: "تحليل المتانة ضد التشويش",
          text: "اختبار إجهاد دقيق للنماذج تحت تشويشات متنوعة (غاوسي، تمويه، تباين) للتحقق من صمود النماذج الكمّية في بيئات الفحص الرديئة.",
        },
        {
          title: "مقارنة معيارية شاملة",
          text: "تقييم مباشر وشامل لزمن الاستدلال، ومعايرة الثقة، وقابلية فصل الفئات بين النماذج التقليدية والمعززة كمّياً.",
        },
      ],
      sections: {
        overviewEyebrow: "السياق البحثي",
        overviewTitle: "الربط بين الحوسبة الكمّية والذكاء الاصطناعي الصناعي",
        overviewText:
          "يعالج هذا المشروع الحاجة الملحّة لكشف العيوب آلياً وبموثوقية في البنية التحتية لقطاع النفط والغاز. من خلال دمج الدوائر الكمّية المتغيرة (VQCs) في المسارات الكلاسيكية، نستكشف قدرة التضمين الكمّي على استخراج ميزات هندسية أغنى والحفاظ على دقة أعلى تحت ظروف التشويش الصناعي الشديد.",
        sliderEyebrow: "أبرز النقاط",
        highlightsEyebrow: "نقاط القوة",
        highlightsTitle: "أهداف البحث الرئيسية",
      },
      team: {
        eyebrow: "الفريق",
        title: "المشرفون على المشروع والفريق البحثي",
        supervisorsTitle: "المشرفون على المشروع",
        researchersTitle: "الفريق البحثي",
        github: "قيت هب",
        linkedin: "لينكد إن",
        supervisors: [
          { name: "د. مصطفى يولداش", role: "المشرف الرئيسي" },
          { name: "د. نايا ناجي", role: "المشرف المشارك" },
        ],
        researchers: [
          {
            name: "فواز الطحيني",
            role: "قائد الفريق",
            github: "https://github.com/Tafawaz",
            linkedin: "https://linkedin.com/in/your-link",
          },
          {
            name: "عبدالرحمن القحطاني",
            role: "عضو فريق",
            github: "https://github.com/AbdulrahmanQht",
            linkedin: "https://www.linkedin.com/in/abdulrahmanqht/",
          },
          {
            name: "علي الحشيّش",
            role: "عضو فريق",
            github: "https://github.com/AliHamad13",
            linkedin: "https://linkedin.com/in/your-link",
          },
          {
            name: "عزام الزهراني",
            role: "عضو فريق",
            github: "https://github.com/xAzzamx",
            linkedin: "https://www.linkedin.com/in/azzam-alzahrani-52926b36b/",
          },
          {
            name: "طلال التويجري",
            role: "عضو فريق",
            github: "https://github.com/TalalAlt",
            linkedin: "https://linkedin.com/in/your-link",
          },
          {
            name: "عمر المالكي",
            role: "عضو فريق",
            github: "https://github.com/Sgingertig",
            linkedin: "https://linkedin.com/in/your-link",
          },
        ],
      },
    },
    classify: {
      title: "كشف العيوب",
      subtitle:
        "قم بتحميل صورة صناعية لتشغيل الاستدلال عبر النماذج الثلاثة في وقت واحد",
      dropzone: "أفلت الصورة هنا أو تصفح الملفات",
      reset: "إعادة ضبط",
      run: "تشغيل التصنيف",
      validating: "جاري التحقق...",
      running: "جاري تشغيل النماذج...",
      max_size: "الحد الأقصى لحجم الملف هو 5 ميجابايت.",
      max_dimensions: "حتى 4096×4096 بكسل.",
      choose: "اختر صورة",
      top_prediction: "التوقع الأفضل",
      results_defect: "تم اكتشاف خلل",
      Deformation: "تشوه",
      Deposition: "ترسبات",
      Disconnect: "انفصال",
      Misalignment: "عدم محاذاة",
      Obstacle: "عائق",
      Rupture: "تمزق / كسر",
      model_comparison: "مقارنة النماذج",
      confidence_scores: "درجات الثقة",
      inference_latency: "وقت الاستجابة",
      distribution: "توزيع الثقة",
      export_title: "تصدير النتائج",
      export_subtitle: "تحميل نتائج التصنيف لمزيد من التحليل",
      confidence: "الثقة",
      confidence_percent: "الثقة (%)",
      latency: "وقت الاستجابة",
      latency_ms: "وقت الاستجابة (ms)",
      model: "النموذج",
      prediction: "التوقع",
      safe: "سليم",
      defect: "خلل",
      err_format: "صيغة غير صالحة. يرجى تحميل صورة PNG أو JPEG أو JPG أو WEBP.",
      err_size: "حجم الملف يتجاوز 5 ميجابايت. يرجى تحميل صورة أصغر حجماً.",
      err_spoof: "محتوى الملف لا يتطابق مع امتداده.",
      err_upload: "فشل التحميل. يرجى المحاولة مرة أخرى.",
      clean: "نظيف",
      noisy: "مشوش",
      noise_toggle: "تفعيل التشويش",
      noise_toggle_sub: "محاكاة ظروف الفحص الرديئة",
      noise_severity: "مستوى التشويش",
      noise_low: "منخفض",
      noise_medium: "متوسط",
      noise_high: "عالي",
      noisy_results: "نتائج مشوشة",
      noisy_top_prediction: "أفضل توقع مع التشويش",
      noisy_label: "التصنيف المشوش (الثقة)",
      compare_label: "مقارنة",
      clean_label: "التصنيف النظيف (الثقة)",
      noisy_confidence_scores: "درجات الثقة للبيانات المشوشة",
      noisy_inference_latency: "زمن الاستدلال للبيانات المشوشة",
      noisy_distribution: "توزيع الثقة للبيانات المشوشة",
      clean_vs_noisy: "النظيف مقابل المشوش",
      clean_noisy_delta: "فرق النظيف والمشوش",
      delta_confidence: "فرق الثقة",
      delta_latency: "فرق زمن الاستدلال",
      eyebrow: "كلاسيكي · كمّي هجين (CPU) · كمّي هجين (GPU)",
      noisy_pending: "شغّل النموذج لمعاينة الصورة المشوَّشة",
      models: {
        cnn: "CNN",
        qnn_cpu: "QNN CPU",
        qnn_gpu: "QNN GPU",
      },
      export_csv_success: "CSV تم تصدير النتائج بنجاح إلى!",
      export_json_success: "JSON تم تصدير النتائج بنجاح إلى",
      restored_hint:
        "تمت استعادة النتائج من الجلسة السابقة — أعد التعيين أو اختر صورة أخرى للتشغيل مجدداً.",
      prediction_match: "النتيجة",
      match: "متطابق",
      mismatch: "غير متطابق",
      tooltips: {
        model_comparison:
          "مقارنة مباشرة بين جميع النماذج. CNN هو النموذج الكلاسيكي الأساسي، بينما QNN CPU وQNN GPU هما النموذجان الهجينان الكمّيان-الكلاسيكيان المدربان على نفس البيانات.",
        confidence_scores:
          "يوضح كل شريط مدى يقين النموذج من توقعه الأول. الثقة العالية ليست دليلاً على الصحة دائماً — فالإجابة الخاطئة بثقة عالية هي فشل في المعايرة وليست ميزة.",
        inference_latency:
          "الوقت الذي يستغرقه كل نموذج لمعالجة الصورة وإعادة التوقع، بالميلي ثانية. القيم الأقل تعني استجابة أسرع.",
        distribution:
          "توزيع الثقة بين النماذج للفئة المتوقعة. الحصة الكبيرة لنموذج معين تعني أنه الأكثر يقيناً من توقعه.",
        noisy_confidence:
          "عند مستويات التشويش الشديدة، قد يُظهر CNN ثقة أعلى من النماذج الكمّية مع بقاء توقعه خاطئاً. هذه ظاهرة ثقة زائفة موثقة — راجع صفحة المقارنة لمزيد من الأدلة.",
        noisy_top_prediction:
          "النموذج الأعلى ثقة في وجود التشويش. ثقة CNN تحت التشويش قد تكون مضللة — أثبتت نتائج المقارنة أن CNN يمكن أن يُخطئ بثقة عالية تحت تشويش شديد.",
        clean_vs_noisy:
          "أعمدة الثقة قبل وبعد إضافة التشويش. انخفاض ثقة النماذج الكمّية يعكس معايرة أفضل، بينما CNN يحتفظ بثقة عالية حتى حين يكون توقعه خاطئاً.",
        clean_noisy_delta:
          "Δ (دلتا) هو الفرق في كل مقياس بعد إضافة التشويش. ارتفاع ثقة CNN تحت تشويش شديد قد يعكس ثقة زائدة وليس تحسناً حقيقياً في الأداء.",
        clean_image:
          "الموتر المُعالج مسبقًا بحجم 384×384 والمُمرَّر مباشرة إلى النماذج",
        noisy_image:
          "يحاكي تدهور جودة الكاميرات الصناعية باستخدام مزيج من خوارزميات التشويش الغاوسي (Gaussian)، وضبابية الحركة، وتقليل التباين، وانسداد العدسة.",
      },
      cnn_overconfidence_warning:
        "تحت التشويش الشديد، قد يُظهر CNN أعلى ثقة هنا — لكن نتائج المقارنة تؤكد أنه قد لا يزال يُخطئ في توقعاته رغم الثقة العالية. لا تعتبر ثقة CNN العالية دليلاً على تفوق أدائه تحت التشويش.",
    },
    benchmark: {
      hero: {
        badge: "مقارنة بحثية",
        title: "مقارنة شاملة لمنظومة كشف العيوب الهجينة",
        subtitle:
          "تقييم يشمل الدقة في البيانات النظيفة، ومقاومة الضوضاء، وزمن الاستدلال، وتحليل أداء الفئات.",
        description:
          "تعرض هذه الصفحة مقارنة مباشرة بين نموذج CNN الأساسي ونموذجي QNN CPU وQNN GPU باستخدام نفس إعدادات البيانات حتى يتمكن الفريق من فهم الاعتمادية ومفاضلات النشر بوضوح.",
        test_samples: "عينات الاختبار",
        fault_classes: "فئات العيوب",
        models_evaluated: "النماذج المقيمة",
        accuracy_overview: "نظرة عامة على الدقة",
        loaded: "تم تحميل النتائج",
        live: "مباشر",
      },
      dataset: {
        eyebrow: "البيانات",
        fault_classes: "فئات الأعطال",
        title: "وصف مجموعة البيانات وعينات ممثلة",
        description:
          "تبدأ صفحة المقارنة بعرض بصري لمجموعة البيانات. كل شريحة في الكاروسيل تعرض عينة حقيقية من إحدى الفئات المستخدمة في التقييم.",
        carousel_badge: "كاروسيل العينات",
        source_label: "المصدر:",
        source_name: "مجموعة بيانات عيوب الأنابيب من كاجل",
        source_url:
          "https://www.kaggle.com/datasets/simplexitypipeline/pipeline-defect-dataset/data",
        transform_title: "التحويل إلى مهمة تصنيف:",
        transform_text:
          "كانت مجموعة البيانات الأصلية مخصصة لاكتشاف الأجسام داخل الصور، ثم تم تحويلها إلى مهمة تصنيف عبر الاحتفاظ فقط بالصور التي يحتوي ملف الوسم الخاص بها على فئة واحدة فريدة.",
        pipeline_title: "خطوات تجهيز البيانات",
        facts: [
          { label: "الجزء", value: "مجموعة الاختبار" },
          { label: "المجال", value: "عيوب الأنابيب الصناعية" },
          { label: "التغطية", value: "6 فئات عيوب" },
          { label: "الاستخدام", value: "مقارنة وتشخيص" },
        ],
        pipeline_steps: [
          "قراءة جميع ملفات الوسوم والتحقق من وجود ملف الصورة المقابل لكل ملف.",
          "الاحتفاظ بالعينة فقط إذا كان ملف الوسم يحتوي على فئة واحدة فريدة، ثم تخزين مسار الصورة والوسم تحت تلك الفئة.",
          "استبعاد أي عينة يكون ملف الوسم الخاص بها فارغاً أو يحتوي على أكثر من فئة فريدة، لذلك لا يتم نسخها إلى المجلدات الجديدة.",
          "تقسيم العينات الصالحة إلى مجموعات تدريب وتحقق واختبار بنسبة 70% و15% و15%.",
          "نسخ الصور الصالحة وملفات الوسوم المطابقة لها إلى بنية مجلدات جديدة مرتبة حسب الجزء واسم الفئة.",
        ],
        samples: [
          {
            key: "Deformation",
            label: "تشوه",
            title: "عينة تشوه",
            text: "عينة ممثلة لفئة التشوه ضمن مجموعة المقارنة لقياس حساسية النماذج تجاه التغيرات الشكلية.",
          },
          {
            key: "Deposition",
            label: "ترسبات",
            title: "عينة ترسبات",
            text: "عينة توضح تراكم المواد على السطح وهو نمط يختبر قدرة النماذج على التقاط الملمس والتفاصيل الدقيقة.",
          },
          {
            key: "Disconnect",
            label: "انفصال",
            title: "عينة انفصال",
            text: "مثال على الانقطاع البنيوي يستخدم لقياس وضوح الفصل بين المناطق السليمة ومناطق الانفصال.",
          },
          {
            key: "Misalignment",
            label: "عدم محاذاة",
            title: "عينة عدم محاذاة",
            text: "عينة لانحراف موضعي تختبر حساسية النماذج تجاه التغيرات الهندسية والإزاحة المكانية.",
          },
          {
            key: "Obstacle",
            label: "عائق",
            title: "عينة عائق",
            text: "عينة تحتوي على عائق بصري يقلل وضوح المشهد ويمثل سيناريو فحص أكثر صعوبة.",
          },
          {
            key: "Rupture",
            label: "تمزق",
            title: "عينة تمزق",
            text: "عينة لفئة التمزق تستخدم لقياس قدرة النماذج على اكتشاف الأعطال الهيكلية الشديدة بثبات.",
          },
        ],
      },
      summary: {
        batch_size: "حجم الدفعة",
        epochs: "عصور التدريب",
        qubits: "الكيوبتات",
        q_depth: "عمق الدارة الكمية",
        resolution: "دقة الصورة",
        device: "الجهاز",
      },
      performance: {
        eyebrow: "الأداء",
        title: "ملخص الأداء النظيف",
        description:
          "مقاييس الأداء على البيانات النظيفة عبر جميع إعدادات النماذج الثلاثة، وتشمل: الدقة، ومتوسط F1 المُوزَّن، وزمن الاستدلال.",
        clean_accuracy: "الدقة النظيفة",
        weighted_f1: "درجة F1 الموزونة",
        latency: "زمن الاستدلال",
        samples: "العينات المقيمة",
        best_accuracy: "أفضل دقة",
        fastest: "الأسرع",
        benchmark: "مقارنة",
      },
      robustness: {
        eyebrow: "التحمل",
        title: "تحليل تحمل التشويش",
        description:
          "بدل بين أنواع التشويش المدعومة لمقارنة سلوك كل نموذج عند تدهور ظروف الفحص.",
        filter_label: "نوع التشويش",
        points: "نقاط",
        level: "المستوى",
        gaussian: "غاوسي",
        blur: "تمويه",
        contrast: "التباين",
        salt_pepper: "ملح وفلفل",
        motion_blur: "تمويه حركي",
        jpeg_compression: "ضغط JPEG",
        lens_occlusion: "حجب العدسة",
      },
      latency: {
        eyebrow: "السرعة",
        title: "سرعة الاستدلال",
        description:
          "يُقاس زمن الاستدلال بالوقت الذي يستغرقه النموذج لمعالجة مدخل واحد، حيث تشير القيم الأقل إلى أداء أسرع وأكثر كفاءة.",
      },
      diagnostics: {
        eyebrow: "التشخيص",
        title: "تشخيص النموذج",
        description:
          "افحص مصفوفة الالتباس ومقاييس كل فئة للنموذج المحدد باستخدام نفس بيانات المقارنة.",
        model: "النموذج",
        actual: "الحقيقة",
        confusion_matrix: "مصفوفة الالتباس",
        per_class_metrics: "مقاييس كل فئة",
        class: "الفئة",
        precision: "الدقة",
        recall: "الاسترجاع",
        f1: "F1",
        support: "الدعم",
        average: "المتوسط",
      },
      config: {
        eyebrow: "الإعدادات",
        title: "لقطة من إعدادات المقارنة",
      },
      states: {
        loading: "جارٍ تحميل نتائج المقارنة...",
        error: "فشل تحميل بيانات المقارنة.",
        empty: "لا توجد بيانات مقارنة حالياً.",
      },
      models: {
        CNN: "CNN",
        QNN_CPU: "QNN CPU",
        QNN_GPU: "QNN GPU",
      },
      charts: {
        efficiency_title: "حد الكفاءة",
        efficiency_desc: "تحليل المقايضة بين الدقة ووقت الاستجابة.",
        radar_title: "مقارنة النماذج متعددة المقاييس",
        radar_desc: "مقارنة متعددة المقاييس عبر النماذج.",
        robustness_curve: "منحنى متانة الضوضاء",
        reliability_title: "منحنى الموثوقية",
        reliability_desc: "مقارنة الدقة مقابل ثقة النموذج.",
        latency_axis: "وقت الاستجابة (ملي ثانية)",
        accuracy_axis: "الدقة (%)",
        confidence_label: "متوسط ثقة النموذج",
        actual_acc_label: "الدقة الفعلية",
        radar_sub:
          "الدقة، F1، الضبط، الاسترجاع، ومتانة الضوضاء عبر النماذج الثلاثة",
        scatter_title: "الدقة مقابل زمن الاستدلال",
        scatter_sub:
          "نقطة واحدة لكل نموذج — الدقة الأعلى والزمن الأقل هما الأفضل",
        robustness_title: "الدقة تحت ضوضاء متصاعدة",
        robustness_sub: "كيف يتراجع أداء كل نموذج مع ازدياد شدة الضوضاء",
        reliability_sub:
          "الدقة الفعلية مقابل متوسط الثقة عند كل مستوى ضوضاء للنموذج المحدد",
        radar_tooltip:
          "كل محور يمثل مقياساً مختلفاً. المساحة الأكبر داخل المضلع تعني أداءً أشمل وأكثر توازناً عبر جميع الأبعاد المقيّسة.",
        scatter_tooltip:
          "كل نقطة تمثل نموذجاً واحداً. الموضع المثالي هو أعلى اليسار — أعلى دقة مع أقل زمن استدلال.",
        robustness_tooltip:
          "يوضح كيف تتراجع دقة كل نموذج مع تصاعد شدة التشويش. الخط الأكثر استواءً يعني أن النموذج أكثر صموداً أمام الضوضاء.",
        reliability_tooltip:
          "يرسم الدقة الفعلية في مقابل متوسط ثقة النموذج عند كل مستوى ضوضاء. النموذج المعاير جيداً يقترب من القطر — النماذج مفرطة الثقة تقع فوقه.",
      },
      metrics: {
        overall_maun: "متوسط الدقة تحت الضوضاء",
        mean_conf: "متوسط الثقة",
        robustness_score: "درجة المتانة",
        accuracy: "الدقة",
        f1: "F1",
        precision: "الضبط",
        recall: "الاسترجاع",
        accuracy_tooltip:
          "نسبة الصور المصنّفة بشكل صحيح من جميع فئات العيوب الست على البيانات النظيفة غير المشوهة.",
        f1_tooltip:
          "المتوسط التوافقي بين الضبط والاسترجاع. يراعي كلاً من الإيجابيات الخاطئة والسلبيات الخاطئة — مقياس جودة متوازن.",
        maun_tooltip:
          "متوسط الدقة تحت الضوضاء — متوسط الدقة عبر جميع أنواع التشويش ومستويات الشدة المقيّسة في المقارنة.",
        robustness_score_tooltip:
          "درجة مركبة تعكس مدى ثبات دقة النموذج تحت جميع ظروف التشويش المختبرة.",
        mean_conf_tooltip:
          "متوسط ثقة التنبؤ عبر جميع عينات الاختبار. قارنه بالدقة الفعلية للكشف عن الثقة الزائدة.",
      },
    },
    qa: {
      title: "تقرير التفوق الكمّي",
      subtitle: "التحقق التجريبي من الآليات الكمّية في البنية الهجينة.",
      generatedAt: "تاريخ التوليد",
      qubits: "كيوبت",
      depth: "العمق",
      loading: "جاري تحميل المقاييس الكمّية...",
      errorTitle: "البيانات غير متاحة",
      errorMsg: "فشل تحميل بيانات التفوق الكمّي.",
      model: "النموذج",
      confirmed: "تفوق مؤكّد",
      notConfirmed: "غير مؤكّد",
      exp2: {
        title: "الكسب الكمّي (إزالة الفرع)",
        desc: "ما مقدار الدقة التي يضيفها الفرع الكمّي مقارنة بالاستدلال الكلاسيكي فقط؟",
        quantumGain: "كسب كمّي",
        fullModel: "النموذج الكامل",
        classicalOnly: "كلاسيكي فقط",
        quantumOnly: "كمّي فقط",
        tooltip: "الكسب الكمّي = دقة النموذج الكامل − دقة الكلاسيكي فقط",
      },
      exp6: {
        title: "المتانة ضد التشويش",
        desc: "هل يزداد دور الفرع الكمّي عند تدهور جودة المدخلات؟",
        gaussian: "ضوضاء غاوسية",
        blur: "ضبابية",
        contrast: "تباين",
        salt_pepper: "ملح وفلفل",
        motion_blur: "تمويه حركي",
        jpeg_compression: "ضغط JPEG",
        lens_occlusion: "حجب العدسة",
        insight:
          "ارتفاع الكسب الكمّي مع زيادة التشويش يؤكد أن الفرع الكمّي يحسّن المتانة في ظروف الفحص الصعبة.",
      },
      exp1: {
        title: "تعامد الميزات",
        desc: "هل يتعلم الفرعان الكمّي والكلاسيكي أشياء مختلفة؟",
        score: "درجة تشابه جيب التمام",
        target: "الهدف: قريب من 0.0 (متعامد)",
        explanation:
          "التشابه المنخفض جداً يؤكد أن الفرع الكمّي يتعلم ميزات فريدة ومكملة مختلفة عن الفرع الكلاسيكي.",
        tooltip:
          "يقيس تشابه جيب التمام الزاوية بين متجهي الميزات. قيمة قريبة من 0.0 تعني أن الفرعين شبه متعامدين — كل منهما يتعلم معلومات مختلفة، وهو الهدف المطلوب.",
      },
      exp13: {
        title: "تحليل CKA الخطي",
        desc: "هل التمثيلات المتعلّمة مختلفة هيكلياً؟",
        scale: "0 = متعامد، 1 = متطابق",
        insight:
          "انخفاض CKA يؤكد أن الفرع الكمّي يتعلم بنية تمثيلية مكملة، وهذا أقوى من تشابه جيب التمام لأنه ثابت تحت الدوران والقياس.",
        tooltip:
          "يقارن محاذاة النواة المتمركزة (CKA) هندسة فضاءي التمثيل. على خلاف تشابه جيب التمام، فهو ثابت تحت الدوران والقياس — لذا فالدرجة المنخفضة هي دليل أقوى وأكثر موثوقية على أن الفرعين تعلّما تمثيلات مختلفة هيكلياً.",
      },
      exp3: {
        title: "مساهمة إعادة التحميل",
        desc: "كم تحسّن تقنية إعادة تحميل البيانات من الدقة؟",
        with: "مع إعادة التحميل",
        without: "بدون إعادة التحميل",
        contribution: "المساهمة",
        tooltip:
          "تُشفّر تقنية إعادة تحميل البيانات المدخلات الكلاسيكية في الدائرة الكمّية عدة مرات عبر طبقات مختلفة. هذا يمنح الدائرة قدرة تعبيرية أكبر — ومقياس المساهمة يُظهر زيادة الدقة التي تحققها هذه التقنية وحدها.",
      },
      exp4: {
        title: "إنتروبيا التشابك",
        desc: "هل تولّد الدائرة الكمّية ارتباطات كمّية حقيقية؟",
        overallMean: "متوسط الإنتروبيا الكلي",
        tooltip:
          "تقيس إنتروبيا فون نيومان التشابك الكمّي لكل كيوبت. قيمة أعلى من 0 تعني أن الدائرة تولّد ارتباطات كمّية حقيقية بين الكيوبتات — وليس مجرد مزج كلاسيكي. الإنتروبيا الأعلى تشير إلى معالجة كمّية أكثر ثراءً.",
      },
      exp5: {
        title: "تباين التدرج (فحص الهضبة الجرداء)",
        desc: "هل لا تزال الدائرة الكمّية قادرة على التعلم أم اختفت التدرجات؟",
        target: "الطبقة المستهدفة",
        meanVar: "متوسط تباين التدرج",
        absMean: "المتوسط المطلق",
        batches: "الدفعات",
        tooltip:
          "الهضاب الجرداء هي مناطق تتلاشى فيها التدرجات أسياً مما يجعل الدوائر الكمّية غير قابلة للتدريب. تباين التدرج غير الصفري يؤكد أن الدائرة لا تزال تتعلم بفاعلية ولم تقع في هذا الفخ.",
      },
      exp7: {
        title: "قابلية تعبير الدائرة الكمّية",
        desc: "ما مدى تغطية الدائرة الكمّية لفضاء هيلبرت المتاح؟",
        klDiv: "تباعد KL عن توزيع هار",
        ref: "مرجع هار",
        insight:
          "انخفاض تباعد KL يعني أن الدائرة تستكشف فضاء هيلبرت بشكل أكثر انتظاماً. القيم أقل من 0.05 تشير إلى قابلية تعبير قريبة من هار.",
        tooltip:
          "دائرة هار العشوائية تأخذ عيّنات من فضاء الحالات الكمّية بالتساوي. يقيس تباعد KL مدى اقتراب دائرتنا المدرّبة من هذا التغطية المثالية. القيم المنخفضة تعني أن الدائرة تستطيع تمثيل مجموعة أوسع من الحالات الكمّية.",
      },
      exp9: {
        title: "الفرق الهندسي",
        desc: "هل تمتد النواة الكمّية لاتجاهات لا تستطيع النواة الكلاسيكية تمثيلها؟",
        insight:
          "g > 1 يعني أن النواة الكمّية تمتد لاتجاهات لا تستطيع نواة RBF الكلاسيكية تمثيلها (Huang et al. 2021). هذا إثبات صارم للتفوق الكمّي.",
        tooltip:
          "يحدّد الفرق الهندسي g مقدار فضاء الميزات الإضافي الذي تغطيه النواة الكمّية مقارنة بنواة RBF الكلاسيكية. g > 1 هو إثبات رسمي قائم على البيانات للتفوق الكمّي وفق تعريف Huang et al. (2021).",
      },
      exp8: {
        title: "محاذاة النواة مع الهدف",
        desc: "أي نواة أفضل في التوافق مع تصنيفات البيانات؟",
        quantum: "KTA كمّي",
        classical: "KTA كلاسيكي",
        diff: "الفرق",
        insight:
          "يقيس KTA مدى توافق النواة مع بنية التصنيفات. الفرق الموجب يعني أن النواة الكمّية أفضل توافقاً مع المهمة.",
        tooltip:
          "تقيس محاذاة النواة المستهدفة (KTA) مدى توافق بنية تشابه النواة مع التصنيفات الحقيقية. KTA أعلى يعني أن النواة تُجمّع الفئات المتشابهة معاً بشكل طبيعي مما يسهّل التصنيف. الفرق الموجب للكمّي على الكلاسيكي يعني أن النواة الكمّية أنسب لهذه المهمة.",
      },
      exp10: {
        title: "البعد الفعّال لفيشر",
        desc: "ما مدى كفاءة كل نموذج في استخدام معاملاته؟",
        params: "المعاملات",
        dEff1000: "d_eff (n=1000)",
        dEffPerParam: "d_eff / معامل",
        insight:
          "ارتفاع d_eff لكل معامل يعني أن النموذج يستخدم معاملاته بكفاءة أعلى. يحقق QNN بُعداً فعّالاً مماثلاً بعدد معاملات أقل بكثير.",
        tooltip:
          "يقيس البعد الفعّال من تحليل مصفوفة معلومات فيشر عدد معاملات النموذج التي تُسهم فعلاً في توقعاته على هذه البيانات. d_eff لكل معامل يعيّر هذا بحجم النموذج — النسبة الأعلى تعني أن كل معامل يؤدي عملاً مفيداً أكثر.",
      },
      exp11: {
        title: "الرتبة الفعّالة للميزات",
        desc: "كم من فضاء التضمين يستخدمه كل فرع فعلياً؟",
        classical: "كلاسيكي",
        quantum: "كمّي",
        insight:
          "الاستخدام = الرتبة الفعّالة / بُعد التضمين. المقارنة توضح كفاءة كل فرع في استخدام أبعاده المتاحة.",
        tooltip:
          "تقيس الرتبة الفعّالة عدد أبعاد التضمين التي تحمل إشارة ذات معنى مقارنة بالضوضاء أو التكرار. نسبة استخدام عالية تعني أن الفرع يستغل طاقته التمثيلية الكاملة بدلاً من هدر الأبعاد.",
      },
      exp12: {
        title: "البعد الجوهري",
        desc: "ما مقدار ضغط المعلومات الذي يحققه كل فرع؟",
        classical: "كلاسيكي (z)",
        quantum: "كمّي (q_emb)",
        insight:
          "انخفاض البعد الجوهري في q_emb مقارنة بـ z مع دقة تنافسية يعني أن الدائرة الكمّية تضغط المعلومات المتعلقة بالفئات بكفاءة أعلى.",
        tooltip:
          "يُقدّر البعد الجوهري درجات الحرية الحقيقية اللازمة لوصف توزيع البيانات في كل فضاء تضمين، باستخدام مُقدِّر TwoNN (Facco 2017). بُعد جوهري أقل مع دقة مشابهة يعني أن الفرع وجد تمثيلاً أكثر إحكاماً وكفاءة للمعلومات ذات الصلة.",
      },
      exp14: {
        title: "قابلية فصل الفئات",
        desc: "ما مدى فصل كل تضمين لفئات العيوب الست؟",
        advantage: "تفوق كمّي",
        insight:
          "معيار فيشر J = tr(S_W⁻¹ S_B). ارتفاع J يعني تجمعات أضيق داخل الفئة وفواصل أوسع بين الفئات.",
        tooltip:
          "يقيس معيار فيشر J قابلية الفصل بين الفئات: يقارن تشتت الفئات عن بعضها (تشتت ما بين الفئات S_B) مقابل تشتت العيّنات داخل كل فئة (تشتت داخل الفئة S_W). J أعلى يعني أن التضمين يكوّن تجمعات أكثر إحكاماً وتمايزاً لكل نوع عيب.",
      },
      methodology: {
        title: "ملاحظات منهجية",
        notes: {
          sample_size: "حجم العينة",
          train_test_split: "تقسيم التدريب / الاختبار",
          augmentation: "تعزيز البيانات",
          optimizer: "المحسِّن",
          loss_function: "دالة الخسارة",
          epochs: "الحقب",
          batch_size: "حجم الدُّفعة",
          learning_rate: "معدل التعلم",
          quantum_backend: "البنية الكمّية",
          n_qubits: "عدد الكيوبتات",
          q_depth: "عمق الدائرة",
          entanglement_entropy: "إنتروبيا التشابك",
          expressibility: "قابلية التعبير",
          kernel_experiments: "تجارب النواة",
          fim: "مصفوفة معلومات فيشر",
          parameter_matched_ablation: "اختبار الاستئصال بمطابقة المعلمات",
        },
      },
    },
    contact: {
      pageTitle: "تواصل معنا",
      pageSubtitle:
        "يسعدنا سماع رأيك. أرسل لنا رسالة وسنرد عليك في أقرب وقت ممكن.",
      name: "اسمك",
      namePlaceholder: "عبدالرحمن أحمد",
      subject: "الموضوع",
      subjectPlaceholder: "كيف يمكننا مساعدتك؟",
      message: "الرسالة",
      messagePlaceholder: "اكتب رسالتك هنا...",
      send: "إرسال الرسالة",
      sending: "جاري الإرسال...",
      required: "هذا الحقل مطلوب",
      successTitle: "نجاح",
      successMsg: "تم إرسال رسالتك بنجاح.",
      errorTitle: "خطأ",
      errorMsg: "فشل إرسال الرسالة. يرجى المحاولة مرة أخرى لاحقاً.",
      sideEyebrow: "تواصل",

      sideTitle: "لنتحدث عن المشروع.",
      sideText:
        "استخدم النموذج للتواصل مع الفريق بخصوص الأسئلة أو التعاون أو الملاحظات حول كاشف العيوب الكمّي-الكلاسيكي.",
      clear: "إمسح النص المدخل الحالي",
    },
  },
};

export const i18n = createI18n({
  legacy: false,
  locale: Cookies.get("app_lang") || "EN",
  fallbackLocale: "EN",
  messages,
});
