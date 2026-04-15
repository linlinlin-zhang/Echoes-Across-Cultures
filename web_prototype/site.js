const ambientCanvases = [...document.querySelectorAll("[data-ambient-canvas]")];
const flowJourneys = [...document.querySelectorAll("[data-flow-journey]")];
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (!prefersReducedMotion && window.Lenis) {
  initializeSmoothScroll();
}

if (ambientCanvases.length && !prefersReducedMotion) {
  ambientCanvases.forEach((canvas) => {
    initializeAmbientCanvas(canvas);
  });
}

if (flowJourneys.length) {
  initializeFlowJourneys();
}

function initializeSmoothScroll() {
  if (window.__echoLenis) {
    return window.__echoLenis;
  }

  const lenis = new window.Lenis({
    lerp: 0.08,
    smoothWheel: true,
    wheelMultiplier: 0.92,
    touchMultiplier: 1,
  });

  window.__echoLenis = lenis;

  if (window.gsap && window.ScrollTrigger) {
    lenis.on("scroll", () => {
      window.ScrollTrigger.update();
    });

    window.gsap.ticker.add((time) => {
      lenis.raf(time * 1000);
    });
    window.gsap.ticker.lagSmoothing(0);
  } else {
    const raf = (time) => {
      lenis.raf(time);
      window.requestAnimationFrame(raf);
    };

    window.requestAnimationFrame(raf);
  }

  return lenis;
}

function initializeAmbientCanvas(canvas) {
  const host = canvas.parentElement;
  const ctx = canvas.getContext("2d");
  const mode = canvas.dataset.ambientCanvas || "landing";
  if (!ctx || !host) {
    return;
  }

  let width = 0;
  let height = 0;
  let rafId = 0;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const resizeObserver = new ResizeObserver(() => resize());
  resizeObserver.observe(host);
  resize();

  function resize() {
    const rect = host.getBoundingClientRect();
    width = Math.max(1, rect.width);
    height = Math.max(1, rect.height);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function frame(timestamp) {
    const time = timestamp * 0.001;
    ctx.clearRect(0, 0, width, height);

    if (mode === "orbit") {
      drawOrbitField(ctx, width, height, time);
    } else if (mode === "pipeline") {
      drawPipelineField(ctx, width, height, time);
    } else if (mode === "detail") {
      drawDetailField(ctx, width, height, time);
    } else if (mode === "journey") {
      drawJourneyField(ctx, width, height, time);
    } else {
      drawLandingField(ctx, width, height, time);
    }

    rafId = window.requestAnimationFrame(frame);
  }

  rafId = window.requestAnimationFrame(frame);

  window.addEventListener("beforeunload", () => {
    window.cancelAnimationFrame(rafId);
    resizeObserver.disconnect();
  }, { once: true });
}

function drawLandingField(ctx, width, height, time) {
  const lines = 6;
  for (let index = 0; index < lines; index += 1) {
    const y = height * (0.22 + index * 0.1);
    const amplitude = 18 + index * 6;
    const phase = time * 0.18 + index * 0.9;
    const warm = index % 2 === 0;
    const stroke = warm ? "rgba(255, 171, 125, 0.16)" : "rgba(137, 156, 255, 0.14)";
    const pulse = warm ? "rgba(255, 190, 142, 0.32)" : "rgba(168, 192, 255, 0.28)";

    ctx.beginPath();
    ctx.moveTo(-40, y + Math.sin(phase) * 8);
    ctx.bezierCurveTo(
      width * 0.22,
      y - amplitude,
      width * 0.58,
      y + amplitude,
      width + 40,
      y + Math.sin(phase + Math.PI * 0.4) * 10,
    );
    ctx.lineWidth = 1.25 + index * 0.08;
    ctx.strokeStyle = stroke;
    ctx.stroke();

    const progress = (time * (0.04 + index * 0.006) + index * 0.16) % 1;
    const x = -30 + progress * (width + 60);
    const py = y + Math.sin(progress * Math.PI * 3 + phase) * amplitude * 0.36;
    drawGlowDot(ctx, x, py, 4 + index * 0.3, pulse);
  }
}

function drawOrbitField(ctx, width, height, time) {
  const cx = width * 0.53;
  const cy = height * 0.46;
  const ellipses = [
    { rx: width * 0.36, ry: height * 0.22, alpha: 0.16 },
    { rx: width * 0.3, ry: height * 0.18, alpha: 0.12 },
    { rx: width * 0.24, ry: height * 0.14, alpha: 0.1 },
  ];

  ellipses.forEach((ellipse, index) => {
    ctx.beginPath();
    ctx.ellipse(cx, cy, ellipse.rx, ellipse.ry, 0.1 + index * 0.18, 0, Math.PI * 2);
    ctx.lineWidth = 1;
    ctx.strokeStyle = `rgba(17, 17, 17, ${ellipse.alpha})`;
    ctx.stroke();

    const pulseAngle = time * (0.55 + index * 0.08) + index;
    const px = cx + Math.cos(pulseAngle) * ellipse.rx;
    const py = cy + Math.sin(pulseAngle) * ellipse.ry;
    drawGlowDot(ctx, px, py, 4 + index, index === 1 ? "rgba(255, 185, 116, 0.34)" : "rgba(157, 177, 255, 0.28)");
  });

  drawSignalRibbon(ctx, width, height, time, {
    y: height * 0.58,
    amplitude: 10,
    color: "rgba(255, 163, 115, 0.18)",
    speed: 0.28,
  });
  drawSignalRibbon(ctx, width, height, time + 0.8, {
    y: height * 0.52,
    amplitude: 7,
    color: "rgba(134, 159, 255, 0.16)",
    speed: 0.24,
  });
}

function drawPipelineField(ctx, width, height, time) {
  const rows = 4;
  for (let index = 0; index < rows; index += 1) {
    const y = height * (0.24 + index * 0.18);
    const wobble = Math.sin(time * 0.8 + index * 0.6) * 6;
    ctx.beginPath();
    ctx.moveTo(28, y);
    ctx.bezierCurveTo(width * 0.28, y - wobble, width * 0.72, y + wobble, width - 28, y);
    ctx.lineWidth = index === 1 ? 2.4 : 1.4;
    ctx.strokeStyle = index % 2 === 0 ? "rgba(140, 156, 255, 0.16)" : "rgba(255, 170, 123, 0.22)";
    ctx.stroke();

    const progress = (time * (0.22 + index * 0.04) + index * 0.18) % 1;
    const x = 28 + progress * (width - 56);
    drawGlowDot(ctx, x, y + Math.sin(progress * Math.PI * 4 + time) * 3, index === 1 ? 7 : 5, index === 1 ? "rgba(255, 170, 123, 0.36)" : "rgba(154, 170, 255, 0.28)");
  }

  for (let column = 0; column < 6; column += 1) {
    const x = width * (0.12 + column * 0.14);
    ctx.beginPath();
    ctx.moveTo(x, 24);
    ctx.lineTo(x, height - 24);
    ctx.strokeStyle = "rgba(17, 17, 17, 0.035)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function drawDetailField(ctx, width, height, time) {
  const cx = width * 0.5;
  const cy = height * 0.46;
  const rings = [
    { radius: Math.min(width, height) * 0.24, start: time * 0.2, length: Math.PI * 1.15, color: "rgba(255, 174, 128, 0.18)" },
    { radius: Math.min(width, height) * 0.3, start: time * -0.18 + 0.8, length: Math.PI * 0.96, color: "rgba(147, 162, 255, 0.16)" },
    { radius: Math.min(width, height) * 0.36, start: time * 0.14 + 1.4, length: Math.PI * 0.72, color: "rgba(17, 17, 17, 0.08)" },
  ];

  rings.forEach((ring, index) => {
    ctx.beginPath();
    ctx.arc(cx, cy, ring.radius, ring.start, ring.start + ring.length);
    ctx.lineWidth = index === 0 ? 2.8 : 1.5;
    ctx.strokeStyle = ring.color;
    ctx.lineCap = "round";
    ctx.stroke();

    const px = cx + Math.cos(ring.start + ring.length) * ring.radius;
    const py = cy + Math.sin(ring.start + ring.length) * ring.radius;
    drawGlowDot(ctx, px, py, 4 + index, index === 0 ? "rgba(255, 181, 134, 0.34)" : "rgba(155, 176, 255, 0.26)");
  });

  for (let index = 0; index < 5; index += 1) {
    const angle = time * 0.4 + index * 1.26;
    const radius = Math.min(width, height) * (0.17 + index * 0.025);
    drawGlowDot(
      ctx,
      cx + Math.cos(angle) * radius,
      cy + Math.sin(angle) * radius,
      2.6,
      index % 2 === 0 ? "rgba(255, 180, 132, 0.2)" : "rgba(153, 174, 255, 0.18)",
    );
  }
}

function drawJourneyField(ctx, width, height, time) {
  const baseY = height * 0.55;
  const gradients = [
    { x: width * 0.14, y: height * 0.24, r: 120, color: "rgba(255, 205, 174, 0.08)" },
    { x: width * 0.46, y: height * 0.34, r: 132, color: "rgba(194, 210, 255, 0.06)" },
    { x: width * 0.82, y: height * 0.48, r: 144, color: "rgba(255, 214, 188, 0.07)" },
  ];

  gradients.forEach((gradient, index) => {
    const radial = ctx.createRadialGradient(
      gradient.x + Math.sin(time * (0.18 + index * 0.03)) * 14,
      gradient.y + Math.cos(time * (0.16 + index * 0.02)) * 10,
      0,
      gradient.x,
      gradient.y,
      gradient.r,
    );
    radial.addColorStop(0, gradient.color);
    radial.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = radial;
    ctx.fillRect(0, 0, width, height);
  });

  const guideLines = [
    { y: baseY - 122, amp: 16, alpha: 0.05 },
    { y: baseY - 54, amp: 12, alpha: 0.055 },
    { y: baseY + 8, amp: 14, alpha: 0.065 },
    { y: baseY + 68, amp: 10, alpha: 0.05 },
  ];

  guideLines.forEach((line, index) => {
    ctx.beginPath();
    ctx.moveTo(-40, line.y);
    ctx.bezierCurveTo(
      width * 0.18,
      line.y - line.amp,
      width * 0.42,
      line.y + line.amp,
      width * 0.68,
      line.y - line.amp * 0.86,
    );
    ctx.bezierCurveTo(
      width * 0.82,
      line.y - line.amp * 1.22,
      width * 0.92,
      line.y + line.amp * 0.8,
      width + 40,
      line.y - 6,
    );
    ctx.lineWidth = index === 2 ? 1.8 : 1.2;
    ctx.strokeStyle = `rgba(17, 17, 17, ${line.alpha})`;
    ctx.stroke();
  });

  const ribbons = [
    { amp: 20, offset: 0, color: "rgba(255, 176, 126, 0.11)", width: 10 },
    { amp: 16, offset: 22, color: "rgba(148, 165, 255, 0.1)", width: 7 },
    { amp: 10, offset: 44, color: "rgba(17, 17, 17, 0.08)", width: 2.4 },
  ];

  ribbons.forEach((line, index) => {
    ctx.beginPath();
    ctx.moveTo(46, baseY + line.offset);
    ctx.bezierCurveTo(
      width * 0.24,
      baseY - line.amp + line.offset,
      width * 0.54,
      baseY + line.amp + line.offset,
      width - 52,
      baseY + line.offset - 12,
    );
    ctx.strokeStyle = line.color;
    ctx.lineWidth = line.width;
    ctx.lineCap = "round";
    ctx.stroke();

    const progress = (time * (0.14 + index * 0.03) + index * 0.24) % 1;
    const x = 46 + progress * (width - 98);
    const y = baseY + line.offset + Math.sin(progress * Math.PI * 2.8 + time) * line.amp * 0.22;
    drawGlowDot(ctx, x, y, index === 0 ? 5.6 : 4, index === 0 ? "rgba(255, 181, 134, 0.22)" : "rgba(154, 175, 255, 0.16)");
  });

  const anchors = [
    [width * 0.16, height * 0.22],
    [width * 0.34, height * 0.16],
    [width * 0.52, height * 0.3],
    [width * 0.72, height * 0.22],
    [width * 0.88, height * 0.42],
  ];

  anchors.forEach(([x, y], index) => {
    drawGlowDot(ctx, x, y, 3 + (index % 2), index % 2 === 0 ? "rgba(255, 180, 132, 0.14)" : "rgba(154, 175, 255, 0.12)");
    ctx.beginPath();
    ctx.moveTo(x, y + 84);
    ctx.lineTo(x + (index % 2 === 0 ? 14 : -12), baseY + 4);
    ctx.strokeStyle = "rgba(17, 17, 17, 0.045)";
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function initializeFlowJourneys() {
  if (window.gsap && window.ScrollTrigger && window.innerWidth > 920) {
    initializeGsapFlowJourneys();
    return;
  }

  const update = () => {
    flowJourneys.forEach((journey) => {
      const sticky = journey.querySelector(".journey-sticky");
      const track = journey.querySelector(".journey-track");
      const stageShell = journey.querySelector(".journey-stage-shell");
      const steps = [...journey.querySelectorAll(".journey-step")];
      const progressPill = journey.querySelector(".journey-progress-pill");
      const miniSteps = [...journey.querySelectorAll(".journey-mini-step")];
      const supportLayers = [...journey.querySelectorAll("[data-anchor-stage]")];
      const tracer = journey.querySelector(".journey-tracer");
      const tracerPath = journey.querySelector(".journey-line-warm");
      const focusHalo = journey.querySelector(".journey-focus-halo");
      if (!sticky || !track || !stageShell) {
        return;
      }

      const rect = journey.getBoundingClientRect();
      const span = Math.max(1, rect.height - window.innerHeight);
      const traveled = clampNumber(-rect.top / span, 0, 1);
      const maxShift = Math.max(0, track.scrollWidth - stageShell.clientWidth + 140);
      track.style.transform = `translate3d(${-maxShift * traveled}px, 0, 0)`;
      journey.style.setProperty("--journey-progress", traveled.toFixed(3));
      const activeIndex = updateJourneyStepState(steps, traveled, {
        progressPill,
        miniSteps,
        supportLayers,
      });
      updateJourneyTracer(tracerPath, tracer, traveled, steps[activeIndex]?.dataset.stageLabel);
      updateJourneyFocusHalo(stageShell, steps[activeIndex], focusHalo);
    });
  };

  update();
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
}

function initializeGsapFlowJourneys() {
  window.gsap.registerPlugin(window.ScrollTrigger);

  flowJourneys.forEach((journey) => {
    const sticky = journey.querySelector(".journey-sticky");
    const stageShell = journey.querySelector(".journey-stage-shell");
    const track = journey.querySelector(".journey-track");
    const steps = [...journey.querySelectorAll(".journey-step")];
    const scenics = [...journey.querySelectorAll(".journey-scenic")];
    const floaters = [...journey.querySelectorAll(".journey-floater")];
    const notes = [...journey.querySelectorAll(".journey-note-card")];
    const auras = [...journey.querySelectorAll(".journey-aura")];
    const progressPill = journey.querySelector(".journey-progress-pill");
    const miniSteps = [...journey.querySelectorAll(".journey-mini-step")];
    const supportLayers = [...journey.querySelectorAll("[data-anchor-stage]")];
    const tracer = journey.querySelector(".journey-tracer");
    const tracerPath = journey.querySelector(".journey-line-warm");
    const focusHalo = journey.querySelector(".journey-focus-halo");

    if (!sticky || !stageShell || !track || !steps.length) {
      return;
    }

    const getDistance = () => Math.max(0, track.scrollWidth - stageShell.clientWidth + 180);
    const getTravelEnd = () => Math.max(window.innerHeight * 1.6, getDistance() + window.innerHeight * 0.6);

    const trackTween = window.gsap.to(track, {
      x: () => -getDistance(),
      ease: "none",
      paused: true,
    });

    window.ScrollTrigger.create({
      trigger: journey,
      start: "top top+=88",
      end: () => `+=${getTravelEnd()}`,
      pin: sticky,
      scrub: 1.15,
      animation: trackTween,
      invalidateOnRefresh: true,
      anticipatePin: 1,
      snap: {
        snapTo: (value) => {
          const maxIndex = Math.max(1, steps.length - 1);
          return Math.round(value * maxIndex) / maxIndex;
        },
        duration: { min: 0.16, max: 0.34 },
        ease: "power1.inOut",
      },
      onUpdate: (self) => {
        journey.style.setProperty("--journey-progress", self.progress.toFixed(3));
        const activeIndex = updateJourneyStepState(steps, self.progress, {
          progressPill,
          miniSteps,
          supportLayers,
        });
        updateJourneyTracer(tracerPath, tracer, self.progress, steps[activeIndex]?.dataset.stageLabel);
        updateJourneyFocusHalo(stageShell, steps[activeIndex], focusHalo);
      },
    });

    steps.forEach((step, index) => {
      const preview = step.querySelector(".journey-preview");
      const intensity = index % 2 === 0 ? -26 : 22;

      window.gsap.fromTo(
        step,
        { rotate: index % 2 === 0 ? -2.2 : 2.2, y: intensity * 0.42, opacity: 0.92 },
        {
          rotate: 0,
          y: 0,
          opacity: 1,
          ease: "none",
          scrollTrigger: {
            trigger: step,
            containerAnimation: trackTween,
            start: "left 86%",
            end: "center center",
            scrub: true,
          },
        },
      );

      if (preview) {
        window.gsap.fromTo(
          preview,
          { y: 18, scale: 0.94, opacity: 0.72 },
          {
            y: 0,
            scale: 1,
            opacity: 1,
            ease: "none",
            scrollTrigger: {
              trigger: step,
              containerAnimation: trackTween,
              start: "left 82%",
              end: "center center",
              scrub: true,
            },
          },
        );
      }
    });

    notes.forEach((note, index) => {
      window.gsap.fromTo(
        note,
        { y: index % 2 === 0 ? -18 : 18, opacity: 0.78, scale: 0.95 },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          ease: "none",
          scrollTrigger: {
            trigger: note,
            containerAnimation: trackTween,
            start: "left 88%",
            end: "center center",
            scrub: true,
          },
        },
      );
    });

    scenics.forEach((scenic, index) => {
      window.gsap.fromTo(
        scenic,
        { y: index % 2 === 0 ? 30 : -24, scale: 0.95, opacity: 0.72 },
        {
          y: 0,
          scale: 1,
          opacity: 1,
          ease: "none",
          scrollTrigger: {
            trigger: scenic,
            containerAnimation: trackTween,
            start: "left 94%",
            end: "center center",
            scrub: true,
          },
        },
      );

      window.gsap.to(scenic, {
        x: index % 2 === 0 ? -42 - index * 6 : -18 - index * 4,
        rotate: index % 2 === 0 ? -3 : 3,
        ease: "none",
        scrollTrigger: {
          trigger: journey,
          start: "top top+=88",
          end: () => `+=${getTravelEnd()}`,
          scrub: 1,
          invalidateOnRefresh: true,
        },
      });
    });

    auras.forEach((aura, index) => {
      window.gsap.to(aura, {
        x: index % 2 === 0 ? 36 : -30,
        y: index === 1 ? -22 : 16,
        scale: 1.08,
        ease: "none",
        scrollTrigger: {
          trigger: aura,
          containerAnimation: trackTween,
          start: "left right",
          end: "right left",
          scrub: true,
        },
      });
    });

    floaters.forEach((floater, index) => {
      const xShift = index % 2 === 0 ? -160 - index * 18 : -110 - index * 22;
      const yShift = index % 2 === 0 ? 24 + index * 5 : -16 - index * 4;

      window.gsap.to(floater, {
        x: xShift,
        y: yShift,
        rotate: index % 2 === 0 ? -4 : 4,
        ease: "none",
        scrollTrigger: {
          trigger: journey,
          start: "top top+=88",
          end: () => `+=${getTravelEnd()}`,
          scrub: 1,
          invalidateOnRefresh: true,
        },
      });
    });
  });
}

function updateJourneyStepState(steps, progress, options = {}) {
  if (!steps.length) {
    return 0;
  }

  const {
    progressPill,
    miniSteps = [],
    supportLayers = [],
  } = options;

  const activeIndex = Math.min(
    steps.length - 1,
    Math.max(0, Math.round(progress * (steps.length - 1))),
  );

  steps.forEach((step, index) => {
    step.classList.toggle("is-active", index === activeIndex);
    step.classList.toggle("is-complete", index < activeIndex);
  });

  miniSteps.forEach((step, index) => {
    step.classList.toggle("is-active", index === activeIndex);
    step.classList.toggle("is-complete", index < activeIndex);
  });

  supportLayers.forEach((layer) => {
    const anchorStage = Number(layer.dataset.anchorStage ?? -1);
    const distance = Math.abs(anchorStage - activeIndex);
    layer.classList.toggle("is-relevant", distance <= 1);
    layer.classList.toggle("is-muted", distance > 1);
  });

  if (progressPill) {
    const activeStep = steps[activeIndex];
    const label = activeStep?.dataset.stageLabel || "流程中";
    progressPill.textContent = `${label} · ${String(activeIndex + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
  }

  return activeIndex;
}

function updateJourneyTracer(path, tracer, progress, label) {
  if (!path || !tracer || typeof path.getTotalLength !== "function") {
    return;
  }

  const totalLength = path.getTotalLength();
  const point = path.getPointAtLength(totalLength * clampNumber(progress, 0, 1));
  tracer.style.left = `${point.x}px`;
  tracer.style.top = `${point.y}px`;

  const tracerLabel = tracer.querySelector(".journey-tracer-label");
  if (tracerLabel && label) {
    tracerLabel.textContent = label;
  }
}

function updateJourneyFocusHalo(stageShell, activeStep, focusHalo) {
  if (!stageShell || !activeStep || !focusHalo) {
    return;
  }

  const shellRect = stageShell.getBoundingClientRect();
  const stepRect = activeStep.getBoundingClientRect();
  const haloX = stepRect.left - shellRect.left + stepRect.width * 0.5 - 150;
  const haloY = stepRect.top - shellRect.top + stepRect.height * 0.44 - 150;
  focusHalo.style.transform = `translate3d(${haloX}px, ${haloY}px, 0)`;
}

function drawSignalRibbon(ctx, width, height, time, config) {
  const { y, amplitude, color, speed } = config;
  ctx.beginPath();
  ctx.moveTo(18, y);
  ctx.bezierCurveTo(
    width * 0.26,
    y - amplitude,
    width * 0.7,
    y + amplitude,
    width - 18,
    y + Math.sin(time * speed * 3) * 3,
  );
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  const progress = (time * speed) % 1;
  const x = 18 + progress * (width - 36);
  const py = y + Math.sin(progress * Math.PI * 4 + time) * amplitude * 0.16;
  drawGlowDot(ctx, x, py, 6, color.replace("0.18", "0.26").replace("0.16", "0.24"));
}

function drawGlowDot(ctx, x, y, radius, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = radius * 4;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
