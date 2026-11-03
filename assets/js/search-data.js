// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-home",
    title: "Home",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "Publications",
          description: "Publications by Boxun Xu in efficient generative modeling, machine learning systems, and hardware co-design.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-blog",
          title: "Blog",
          description: "More to come.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "news-our-paper-on-a-multi-modal-iot-soc-with-on-chip-mram-was-accepted-for-publication-in-jssc",
          title: 'Our paper on a multi-modal IoT SoC with on-chip MRAM was accepted for...',
          description: "",
          section: "News",},{id: "news-started-my-summer-internship-at-meta-ai-working-on-multi-teacher-knowledge-distillation-for-multimodal-multitask-foundation-models",
          title: 'Started my summer internship at Meta AI, working on multi-teacher knowledge distillation for...',
          description: "",
          section: "News",},{id: "news-our-papers-on-3d-spiking-transformer-accelerators-and-llm-guided-analog-design-were-accepted-to-iccad-2024",
          title: 'Our papers on 3D spiking transformer accelerators and LLM-guided analog design were accepted...',
          description: "",
          section: "News",},{id: "news-our-paper-on-3d-spiking-transformer-accelerators-was-nominated-for-the-william-j-mccalla-best-paper-award-at-iccad-2024",
          title: 'Our paper on 3D spiking transformer accelerators was nominated for the William J....',
          description: "",
          section: "News",},{id: "news-joining-meta-superintelligence-labs-this-summer-in-seattle-to-work-on-efficient-video-generation",
          title: 'Joining Meta Superintelligence Labs this summer in Seattle to work on efficient video...',
          description: "",
          section: "News",},{id: "news-our-paper-on-network-hardware-co-optimization-for-sparse-snn-accelerators-was-accepted-for-publication-in-tcad-as-a-long-paper",
          title: 'Our paper on network-hardware co-optimization for sparse SNN accelerators was accepted for publication...',
          description: "",
          section: "News",},{id: "news-our-paper-on-heterogeneous-core-acceleration-of-spiking-transformers-with-error-constrained-pruning-was-accepted-to-isca-2025",
          title: 'Our paper on heterogeneous-core acceleration of spiking transformers with error-constrained pruning was accepted...',
          description: "",
          section: "News",},{id: "news-our-paper-on-heterogeneous-quantization-for-spiking-vision-transformers-was-accepted-to-asap-2025",
          title: 'Our paper on heterogeneous quantization for spiking vision transformers was accepted to ASAP...',
          description: "",
          section: "News",},{id: "news-our-paper-on-transfer-learning-for-vmin-prediction-in-advanced-nodes-was-accepted-to-itc-2025",
          title: 'Our paper on transfer learning for Vmin prediction in advanced nodes was accepted...',
          description: "",
          section: "News",},{id: "news-our-paper-on-3d-moe-spiking-transformer-acceleration-was-accepted-to-iccad-2025",
          title: 'Our paper on 3D MoE spiking transformer acceleration was accepted to ICCAD 2025....',
          description: "",
          section: "News",},{id: "news-our-paper-on-3d-moe-spiking-transformers-was-nominated-for-the-william-j-mccalla-best-paper-award-at-iccad-2025-for-the-second-consecutive-year",
          title: 'Our paper on 3D MoE spiking transformers was nominated for the William J....',
          description: "",
          section: "News",},{id: "news-our-papers-on-adaptive-kv-caching-for-visual-autoregressive-models-and-kan-based-graph-contrastive-learning-were-accepted-to-aaai-2026",
          title: 'Our papers on adaptive KV caching for visual autoregressive models and KAN-based graph...',
          description: "",
          section: "News",},{id: "news-sparse-forcing-native-trainable-sparse-attention-for-real-time-autoregressive-video-generation-in-collaboration-with-meta-superintelligence-labs-preprint",
          title: 'Sparse Forcing: native trainable sparse attention for real-time autoregressive video generation, in collaboration...',
          description: "",
          section: "News",},{id: "news-completed-my-internship-at-meta-superintelligence-labs-working-on-efficient-video-generation-for-moviegen",
          title: 'Completed my internship at Meta Superintelligence Labs, working on efficient video generation for...',
          description: "",
          section: "News",},{id: "news-vegas-our-work-on-mitigating-vlm-hallucinations-was-accepted-to-cvpr-2026-findings",
          title: 'VEGAS, our work on mitigating VLM hallucinations, was accepted to CVPR 2026 Findings....',
          description: "",
          section: "News",},{id: "news-received-the-radhakrishnan-nagarajan-family-fellowship-at-ucsb",
          title: 'Received the Radhakrishnan Nagarajan Family Fellowship at UCSB.',
          description: "",
          section: "News",},{id: "news-named-an-icml-2026-gold-reviewer",
          title: 'Named an ICML 2026 Gold Reviewer.',
          description: "",
          section: "News",},{id: "news-won-second-place-at-the-dac-ph-d-forum",
          title: 'Won second place at the DAC Ph.D. Forum.',
          description: "",
          section: "News",},{id: "news-defended-my-ph-d-dissertation-co-architecting-models-and-systems-for-efficient-perception-and-generation-at-uc-santa-barbara",
          title: 'Defended my Ph.D. dissertation, “Co-Architecting Models and Systems for Efficient Perception and Generation,”...',
          description: "",
          section: "News",},{id: "news-var-q-our-work-on-tuning-free-kv-cache-quantization-for-visual-autoregressive-generation-was-accepted-to-neurips-2026-paper",
          title: 'VAR-Q, our work on tuning-free KV cache quantization for visual autoregressive generation, was...',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%62%6F%78%75%6E%78%75@%75%63%73%62.%65%64%75", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/boxun-x-865232154", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0009-0003-2896-6632", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=MU2fk-kAAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
