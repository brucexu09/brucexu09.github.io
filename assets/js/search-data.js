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
  },{id: "nav-research",
          title: "Research",
          description: "Publications grouped by research pillar.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-from-ppo-to-search-r1-the-design-space-of-reasoning-and-agentic-rl",
      
        title: "From PPO to Search-R1: The Design Space of Reasoning and Agentic RL",
      
      description: "A component-by-component decomposition of PPO through GRPO, verifiers, and retrieved-token masking to a complete instance of Agentic RL.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2026/from-ppo-to-search-r1/";
        
      },
    },{id: "news-paper-on-a-multi-modal-iot-soc-with-on-chip-mram-accepted-at-jssc",
          title: 'Paper on a multi-modal IoT SoC with on-chip MRAM accepted at JSSC.',
          description: "",
          section: "News",},{id: "news-started-summer-internship-at-meta-ai-working-on-multi-teacher-knowledge-distillation-of-multi-modal-multi-task-foundation-models",
          title: 'Started summer internship at Meta AI, working on multi-teacher knowledge distillation of multi-modal...',
          description: "",
          section: "News",},{id: "news-papers-on-3d-spiking-transformer-accelerators-and-llm-guided-analog-design-accepted-at-iccad-2024",
          title: 'Papers on 3D spiking transformer accelerators and LLM-guided analog design accepted at ICCAD...',
          description: "",
          section: "News",},{id: "news-paper-on-3d-spiking-transformer-accelerators-nominated-for-the-william-j-mccalla-best-paper-award-at-iccad-2024",
          title: '🏆 Paper on 3D spiking transformer accelerators nominated for the William J. McCalla...',
          description: "",
          section: "News",},{id: "news-joining-meta-superintelligence-labs-this-summer-in-seattle-working-on-efficient-movie-generation",
          title: 'Joining Meta Superintelligence Labs this summer in Seattle, working on efficient movie generation....',
          description: "",
          section: "News",},{id: "news-paper-on-network-hardware-co-optimization-for-sparse-snn-accelerators-accepted-at-tcad-as-a-long-paper",
          title: 'Paper on network-hardware co-optimization for sparse SNN accelerators accepted at TCAD as a...',
          description: "",
          section: "News",},{id: "news-paper-on-heterogeneous-core-acceleration-of-spiking-transformers-with-error-constrained-pruning-accepted-at-isca-2025",
          title: 'Paper on heterogeneous-core acceleration of spiking transformers with error-constrained pruning accepted at ISCA...',
          description: "",
          section: "News",},{id: "news-paper-on-heterogeneous-quantization-for-spiking-vision-transformers-accepted-at-asap-2025",
          title: 'Paper on heterogeneous quantization for spiking vision transformers accepted at ASAP 2025.',
          description: "",
          section: "News",},{id: "news-paper-on-transfer-learning-for-vmin-prediction-in-advanced-nodes-accepted-at-itc-2025",
          title: 'Paper on transfer learning for Vmin prediction in advanced nodes accepted at ITC...',
          description: "",
          section: "News",},{id: "news-paper-on-3d-moe-spiking-transformer-acceleration-accepted-at-iccad-2025",
          title: 'Paper on 3D MoE spiking transformer acceleration accepted at ICCAD 2025.',
          description: "",
          section: "News",},{id: "news-paper-on-3d-moe-spiking-transformers-nominated-for-the-william-j-mccalla-best-paper-award-at-iccad-2025-second-consecutive-year",
          title: '🏆 Paper on 3D MoE spiking transformers nominated for the William J. McCalla...',
          description: "",
          section: "News",},{id: "news-papers-on-adaptive-kv-caching-for-visual-autoregressive-models-and-kan-based-graph-contrastive-learning-accepted-at-aaai-2026",
          title: 'Papers on adaptive KV caching for visual autoregressive models and KAN-based graph contrastive...',
          description: "",
          section: "News",},{id: "news-paper-on-vlm-hallucination-mitigation-vegas-accepted-at-cvpr-2026-findings",
          title: 'Paper on VLM hallucination mitigation (VEGAS) accepted at CVPR 2026 Findings.',
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
