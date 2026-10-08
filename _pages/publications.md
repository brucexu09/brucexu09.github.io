---
layout: page
permalink: /publications/
title: Research
description: Research in efficient visual generation, KV-cache compression, sparse attention, and hardware–algorithm co-design.
nav: true
nav_order: 2
---

<!-- _pages/publications.md -->

<!-- Bibsearch Feature -->

{% include bib_search.liquid %}

<div class="pub-legend">
  <span><span class="representative-star">★</span> Representative work</span>
  <span><sup>*</sup> Co-first authors</span>
  <span>🏆 Award / nomination</span>
</div>

<div class="publications">

<h2 class="pillar">Efficient Generative Modeling</h2>
<p class="pillar-intro">Adaptive KV caching, tuning-free quantization, and trainable sparse attention for scalable image and video autoregressive generation.</p>
{% bibliography --query @*[category=generative]* --group_by none %}

<h2 class="pillar">Hardware/Algorithm Co-design and EDA</h2>
<p class="pillar-intro">Spiking transformers, 3D accelerators, and LLM-assisted EDA — from algorithm down to silicon.</p>
{% bibliography --query @*[category=codesign]* --group_by none %}

<h2 class="category">Other Publications</h2>
{% bibliography --query @*[category=others]* --group_by none %}

</div>
