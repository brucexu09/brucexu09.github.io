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

<h2 class="pillar">Efficient Generative Modeling</h2>
{% bibliography --query @*[category=generative]* --group_by none %}

<h2 class="pillar">Hardware/Algorithm Co-design and EDA</h2>
{% bibliography --query @*[category=codesign]* --group_by none %}

<h2 class="category">Other Publications</h2>
{% bibliography --query @*[category=others]* --group_by none %}

</div>
