---
layout: page
permalink: /publications/
title: Research
description: Publications grouped by research pillar.
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
<p class="pillar-intro">KV caching, sparse attention, and quantization for scalable visual & video autoregressive models.</p>
{% bibliography --query @*[category=generative]* --group_by none %}

<h2 class="pillar">Hardware/Algorithm Co-design and EDA</h2>
<p class="pillar-intro">Spiking transformers, 3D accelerators, and LLM-assisted EDA — from algorithm down to silicon.</p>
{% bibliography --query @*[category=codesign]* --group_by none %}

<h2 class="category">Other Publications</h2>
{% bibliography --query @*[category=others]* --group_by none %}

</div>
