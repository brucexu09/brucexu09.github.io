(() => {
  const root = document.getElementById("bxp");
  if (!root) return;
  const pairedVideos = Array.from(root.querySelectorAll("[data-comparison-video]"));
  const views = Array.from(root.querySelectorAll("[data-view]"));
  const navigation = Array.from(root.querySelectorAll(".bx-topnav [data-view-link]"));
  const showView = (name, shouldScroll) => {
    if (!views.some((view) => view.dataset.view === name)) return;
    views.forEach((view) => {
      view.hidden = view.dataset.view !== name;
    });
    navigation.forEach((link) => {
      if (link.dataset.viewLink === name) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    root.dataset.currentView = name;
    root.dispatchEvent(new Event("bx:viewchange"));
    if (name !== "home") pairedVideos.forEach((video) => video.pause());
    if (shouldScroll) root.scrollIntoView({ behavior: "auto", block: "start" });
  };
  root.querySelectorAll("[data-view-link]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      showView(link.dataset.viewLink, true);
      const destination = link.getAttribute("href");
      if (window.location.pathname + window.location.hash !== destination) {
        window.history.pushState(null, "", destination);
      }
    });
  });
  const restoreRoute = (shouldScroll) => {
    const anchorId = window.location.hash.slice(1);
    const target = Array.from(root.querySelectorAll("[id]")).find((element) => element.id === anchorId);
    const view = target ? target.closest("[data-view]") : null;
    if (view) showView(view.dataset.view, false);
    else {
      const path = window.location.pathname.replace(/\/+$/, "") || "/";
      const name = path === "/publications" ? "publications" : path === "/blog" ? "blog" : "home";
      showView(name, false);
    }
    if (shouldScroll && target) target.scrollIntoView({ behavior: "auto", block: "start" });
  };
  window.addEventListener("hashchange", () => restoreRoute(true));
  window.addEventListener("popstate", () => restoreRoute(false));
  showView(root.dataset.initialView || "home", false);
  restoreRoute(false);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const pairs = Array.from(root.querySelectorAll("[data-hd-pair]"));
  pairs.forEach((pair) => {
    const videos = Array.from(pair.querySelectorAll("video"));
    const [leader, follower] = videos;
    const toggle = pair.querySelector("[data-video-toggle]");
    const playbackRate = Number(pair.dataset.playbackRate) || 1;
    let visible = false;
    let manualPlay = false;
    let starting = false;
    let autoplayBlocked = false;
    let generation = 0;
    const eligible = () =>
      visible &&
      !document.hidden &&
      root.dataset.currentView === "home" &&
      pair.dataset.userPaused !== "true" &&
      (!reducedMotion.matches || manualPlay);
    const ready = () => videos.every((video) => video.readyState >= 3 && !video.error);
    const update = () => {
      const paused = videos.every((video) => video.paused);
      toggle.dataset.paused = String(paused);
      toggle.setAttribute("aria-label", (paused ? "Play " : "Pause ") + pair.dataset.playbackLabel);
    };
    const pause = () => {
      generation += 1;
      starting = false;
      videos.forEach((video) => {
        video.autoplay = false;
        video.pause();
        video.playbackRate = playbackRate;
      });
      update();
    };
    const play = (startLoading = false) => {
      // A direct play request starts loading on mobile, even before canplay fires.
      if (startLoading === true) autoplayBlocked = false;
      if (
        !eligible() ||
        autoplayBlocked ||
        (startLoading !== true && !ready()) ||
        videos.some((video) => video.error) ||
        starting ||
        videos.every((video) => !video.paused)
      )
        return;
      starting = true;
      const request = ++generation;
      Promise.allSettled(
        videos.map((video) => {
          video.autoplay = true;
          video.playbackRate = playbackRate;
          return video.play();
        })
      ).then((results) => {
        if (request !== generation) return;
        starting = false;
        if (!eligible()) pause();
        else if (results.some((result) => result.status === "rejected")) {
          // A browser rejection is not an explicit pause. Retry on re-entry.
          autoplayBlocked = true;
          pause();
        }
        update();
      });
    };
    const refresh = () => {
      if (eligible()) play(true);
      else pause();
    };
    videos.forEach((video) => {
      video.controls = false;
      video.autoplay = false;
      video.defaultMuted = true;
      video.muted = true;
      video.playsInline = true;
      video.loop = true;
      video.defaultPlaybackRate = playbackRate;
      video.playbackRate = playbackRate;
      video.addEventListener("play", () => {
        if (!eligible()) pause();
        else update();
      });
      video.addEventListener("pause", update);
      video.addEventListener("canplay", () => play());
      video.addEventListener("waiting", () => {
        // Initial buffering must not cancel the play request that triggered it.
        if (!starting && !ready()) pause();
      });
      video.addEventListener("error", () => {
        pause();
        const fallback = pair.parentElement.querySelector(".bx-media-fallback");
        if (fallback) fallback.hidden = false;
      });
    });
    // Correct small timing differences through speed, without seeking across frames.
    leader.addEventListener("timeupdate", () => {
      if (!eligible() || !ready() || videos.some((video) => video.paused || video.seeking)) return;
      const duration = Math.min(...videos.map((video) => video.duration));
      if (!Number.isFinite(duration) || duration <= 0) return;
      let drift = follower.currentTime - leader.currentTime;
      if (drift > duration / 2) drift -= duration;
      if (drift < -duration / 2) drift += duration;
      const correction = Math.abs(drift) < 0.035 ? 0 : Math.max(-0.08, Math.min(0.08, drift * 0.3));
      follower.playbackRate = playbackRate * (1 - correction);
    });
    toggle.hidden = false;
    toggle.addEventListener("click", () => {
      if (starting || videos.some((video) => !video.paused)) {
        pair.dataset.userPaused = "true";
        manualPlay = false;
        pause();
      } else {
        pair.dataset.userPaused = "false";
        manualPlay = true;
        play(true);
      }
    });
    const observer = new IntersectionObserver(
      (entries) => {
        visible = entries[0].isIntersecting;
        if (visible)
          videos.forEach((video) => {
            video.preload = "auto";
          });
        refresh();
      },
      { threshold: 0.1 }
    );
    observer.observe(pair);
    document.addEventListener("visibilitychange", refresh);
    window.addEventListener("pageshow", refresh);
    root.addEventListener("bx:viewchange", refresh);
    reducedMotion.addEventListener("change", () => {
      manualPlay = false;
      refresh();
    });
    pause();
  });

  // Native content scrolling with a persistent, draggable scrollbar.
  const newsWindow = root.querySelector(".bx-news-window");
  const newsScroll = root.querySelector(".bx-news-scroll");
  const newsBar = root.querySelector(".bx-news-scrollbar");
  const newsThumb = root.querySelector(".bx-news-scroll-thumb");
  if (newsWindow && newsScroll && newsBar && newsThumb) {
    let dragPointer = null;
    let grabOffset = 0;
    const maximumScroll = () => Math.max(0, newsScroll.scrollHeight - newsScroll.clientHeight);
    const updateNewsScrollbar = () => {
      if (!newsScroll.clientHeight) return;
      const maximum = maximumScroll();
      const position = Math.max(0, Math.min(maximum, newsScroll.scrollTop));
      const overflow = maximum > 2;
      newsWindow.dataset.overflow = String(overflow);
      newsWindow.dataset.atStart = String(position <= 2);
      newsWindow.dataset.atEnd = String(position >= maximum - 2);
      newsWindow.dataset.scrollbarReady = "true";
      newsBar.hidden = !overflow;
      newsBar.setAttribute("aria-valuemax", String(Math.round(maximum)));
      newsBar.setAttribute("aria-valuenow", String(Math.round(position)));
      if (!overflow) return;
      const trackHeight = newsBar.getBoundingClientRect().height;
      const thumbHeight = Math.min(trackHeight, Math.max(36, (trackHeight * newsScroll.clientHeight) / newsScroll.scrollHeight));
      const thumbTop = ((trackHeight - thumbHeight) * position) / maximum;
      newsThumb.style.height = thumbHeight + "px";
      newsThumb.style.transform = "translateY(" + thumbTop + "px)";
    };
    const moveThumb = (clientY) => {
      const track = newsBar.getBoundingClientRect();
      const travel = track.height - newsThumb.getBoundingClientRect().height;
      if (travel <= 0) return;
      const fraction = Math.max(0, Math.min(1, (clientY - track.top - grabOffset) / travel));
      newsScroll.scrollTop = fraction * maximumScroll();
      updateNewsScrollbar();
    };
    newsBar.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || maximumScroll() <= 2) return;
      event.preventDefault();
      newsBar.focus({ preventScroll: true });
      const thumb = newsThumb.getBoundingClientRect();
      grabOffset = event.clientY >= thumb.top && event.clientY <= thumb.bottom ? event.clientY - thumb.top : thumb.height / 2;
      dragPointer = event.pointerId;
      newsBar.dataset.dragging = "true";
      newsBar.setPointerCapture(dragPointer);
      moveThumb(event.clientY);
    });
    newsBar.addEventListener("pointermove", (event) => {
      if (event.pointerId === dragPointer) moveThumb(event.clientY);
    });
    const stopDragging = (event) => {
      if (event.pointerId !== dragPointer) return;
      dragPointer = null;
      newsBar.dataset.dragging = "false";
      if (newsBar.hasPointerCapture(event.pointerId)) newsBar.releasePointerCapture(event.pointerId);
    };
    newsBar.addEventListener("pointerup", stopDragging);
    newsBar.addEventListener("pointercancel", stopDragging);
    newsBar.addEventListener("lostpointercapture", stopDragging);
    newsBar.addEventListener("keydown", (event) => {
      const maximum = maximumScroll();
      const page = newsScroll.clientHeight * 0.9;
      const targets = {
        ArrowDown: newsScroll.scrollTop + 40,
        ArrowUp: newsScroll.scrollTop - 40,
        PageDown: newsScroll.scrollTop + page,
        PageUp: newsScroll.scrollTop - page,
        Home: 0,
        End: maximum,
      };
      if (!(event.key in targets)) return;
      event.preventDefault();
      newsScroll.scrollTop = Math.max(0, Math.min(maximum, targets[event.key]));
      updateNewsScrollbar();
    });
    newsBar.addEventListener(
      "wheel",
      (event) => {
        if (!event.deltaY || maximumScroll() <= 2) return;
        event.preventDefault();
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? newsScroll.clientHeight : 1;
        newsScroll.scrollTop = Math.max(0, Math.min(maximumScroll(), newsScroll.scrollTop + event.deltaY * unit));
        updateNewsScrollbar();
      },
      { passive: false }
    );
    newsScroll.addEventListener("scroll", updateNewsScrollbar, { passive: true });
    if ("ResizeObserver" in window) {
      const newsResize = new ResizeObserver(updateNewsScrollbar);
      newsResize.observe(newsScroll);
      newsResize.observe(newsScroll.querySelector(".bx-news-list"));
    }
    window.addEventListener("resize", updateNewsScrollbar);
    updateNewsScrollbar();
  }
})();
