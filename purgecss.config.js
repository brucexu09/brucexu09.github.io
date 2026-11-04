module.exports = {
  content: ["_site/**/*.html", "_site/**/*.js"],
  css: ["_site/assets/css/*.css"],
  output: "_site/assets/css/",
  skippedContentGlobs: ["_site/assets/**/*.html"],
  // These attributes are set through dataset at runtime and are absent from static HTML.
  dynamicAttributes: ["data-paused", "data-at-start", "data-at-end", "data-overflow", "data-dragging", "data-scrollbar-ready"],
};
