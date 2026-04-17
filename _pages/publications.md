---
layout: page
permalink: /publications/
title: publications
description: publications grouped by research pillar.
nav: true
nav_order: 2
---

<!-- _pages/publications.md -->

<!-- Bibsearch Feature -->

{% include bib_search.liquid %}

<div class="publications">

<h2 class="category">Efficient Generative Modeling</h2>
{% bibliography --query @*[category=generative]* %}

<h2 class="category">Hardware/Algorithm Co-design and EDA</h2>
{% bibliography --query @*[category=codesign]* %}

<h2 class="category">Others</h2>
{% bibliography --query @*[category=others]* %}

</div>
