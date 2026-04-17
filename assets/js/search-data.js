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
        },{id: "post-从-ppo-到-search-r1-reasoning-与-agentic-rl-的设计空间",
      
        title: "从 PPO 到 Search-R1：Reasoning 与 Agentic RL 的设计空间",
      
      description: "按组件拆解 PPO，然后沿 GRPO、Verifier、Retrieved Token Masking 一路梳理到 Agentic RL 的完整实例",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2026/from-ppo-to-search-r1-v2/";
        
      },
    },{id: "post-google-gemini-updates-flash-1-5-gemma-2-and-project-astra",
      
        title: 'Google Gemini updates: Flash 1.5, Gemma 2 and Project Astra <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
      
      description: "We’re sharing updates across our Gemini family of models and a glimpse of Project Astra, our vision for the future of AI assistants.",
      section: "Posts",
      handler: () => {
        
          window.open("https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/", "_blank");
        
      },
    },{id: "post-displaying-external-posts-on-your-al-folio-blog",
      
        title: 'Displaying External Posts on Your al-folio Blog <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.open("https://medium.com/@al-folio/displaying-external-posts-on-your-al-folio-blog-b60a1d241a0a?source=rss-17feae71c3c4------2", "_blank");
        
      },
    },{id: "news-one-paper-is-accepted-by-jssc",
          title: 'One paper is accepted by JSSC.',
          description: "",
          section: "News",},{id: "news-i-started-my-summer-internship-at-working-on-knowledge-distillation-of-multi-modal-foundation-models",
          title: 'I started my summer internship at , working on Knowledge Distillation of Multi-modal...',
          description: "",
          section: "News",},{id: "news-two-papers-are-accepted-by-iccad-2024",
          title: 'Two papers are accepted by ICCAD 2024!',
          description: "",
          section: "News",},{id: "news-our-work-spiking-transformer-accelerators-in-3d-integration-is-nominated-as-william-j-mccalla-best-paper-award-at-iccad-24",
          title: 'Our Work “Spiking Transformer Accelerators in 3D Integration” is nominated as William J....',
          description: "",
          section: "News",},{id: "news-this-summer-i-will-join-working-on-efficient-movie-generation-in-seattle",
          title: 'This summer, I will join , working on Efficient Movie Generation in Seattle!...',
          description: "",
          section: "News",},{id: "news-our-work-has-been-accepted-by-ieee-transactions-on-computer-aided-design-of-integrated-circuits-and-systems-tcad-as-a-long-paper",
          title: 'Our work has been accepted by IEEE Transactions on Computer-Aided Design of Integrated...',
          description: "",
          section: "News",},{id: "news-one-paper-is-accepted-by-international-symposium-on-computer-architecture-isca-25-see-you-in-tokyo",
          title: 'One paper is accepted by International Symposium on Computer Architecture (ISCA’25)! See you...',
          description: "",
          section: "News",},{id: "news-one-paper-is-accepted-by-asap-2025",
          title: 'One paper is accepted by ASAP 2025!',
          description: "",
          section: "News",},{id: "news-one-paper-is-accepted-by-itc-2025",
          title: 'One paper is accepted by ITC 2025!',
          description: "",
          section: "News",},{id: "news-one-paper-is-accepted-by-iccad-2025",
          title: 'One paper is accepted by ICCAD 2025!',
          description: "",
          section: "News",},{id: "news-our-work-has-been-nominated-for-the-william-j-mccalla-best-paper-award-at-iccad-2025-for-the-second-consecutive-year",
          title: 'Our work has been nominated for the William J. McCalla Best Paper Award...',
          description: "",
          section: "News",},{id: "news-two-papers-are-accepted-by-aaai-2026",
          title: 'Two papers are accepted by AAAI 2026!',
          description: "",
          section: "News",},{id: "news-one-paper-is-accepted-by-cvpr-2026-findings",
          title: 'One paper is accepted by CVPR 2026 Findings!',
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
