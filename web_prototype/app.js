const api = {
  bootstrap: "/api/prototype/bootstrap",
  upload: "/api/prototype/upload",
  analyze: "/api/prototype/analyze",
  register: "/api/prototype/register",
  feedback: "/api/prototype/feedback",
};

const modeLabels = {
  bridge: "桥接发现（Bridge）",
  novelty: "新颖探索（Novelty）",
  precision: "精准邻近（Precision）",
};

const lensLabels = {
  rhythm: "节奏（Rhythm）",
  timbre: "音色（Timbre）",
  emotion: "情绪（Emotion）",
};

const sceneNotes = {
  intake: "系统正在整理这段音乐的基本信息，准备进入分析流程。",
  embedding: "系统正在构建音乐表示与文化桥接线索。",
  recommend: "系统已经整理出推荐结果，并会把原因一起交给你。",
};

const fallbackTrack = {
  id: "sample-local",
  name: "夜市弦影（Night Market Strings）.wav",
  size_mb: 15.4,
  descriptor: "拨弦乐器、手鼓脉冲、装饰性旋律线",
  waveform: buildFallbackWaveform(),
  audio_url: "",
};

const fallbackRecommendations = [
  {
    id: "fallback-1",
    title: "安达卢西亚乌德行旅（Andalusian Oud Caravan）",
    origin: "西班牙南部 / 摩洛哥",
    bridge: 0.86,
    novelty: 0.62,
    similarity: 0.84,
    confidence: 0.91,
    score: 92,
    bpm: "96",
    axis: "拨弦音色 + 循环律动",
    summary: "拨弦纹理和循环节奏与源轨最自然地接上了。",
    reason: "它在拨弦攻击感和律动组织上都与你的源轨比较接近，所以会先作为桥接型推荐出现。",
    tags: ["乌德琴", "拨弦", "循环节奏", "桥接友好"],
    audio_url: "",
  },
  {
    id: "fallback-2",
    title: "维吾尔热瓦甫律动（Uyghur Rawap Motion）",
    origin: "中国新疆",
    bridge: 0.88,
    novelty: 0.55,
    similarity: 0.87,
    confidence: 0.92,
    score: 94,
    bpm: "102",
    axis: "拨弦共振 + 手鼓推动",
    summary: "音色和舞蹈能量最贴近，是一条高置信度推荐。",
    reason: "热瓦甫的拨弦共振和手鼓推动感都很容易与这段源轨建立联系，因此系统会优先呈现它。",
    tags: ["热瓦甫", "舞蹈脉冲", "共振", "高置信度"],
    audio_url: "",
  },
  {
    id: "fallback-3",
    title: "第比利斯复调回声（Tbilisi Polyphonic Echoes）",
    origin: "格鲁吉亚",
    bridge: 0.79,
    novelty: 0.74,
    similarity: 0.78,
    confidence: 0.84,
    score: 88,
    bpm: "82",
    axis: "持续张力 + 复调层叠",
    summary: "这条候选更偏发现型，适合继续往更远的文化空间探索。",
    reason: "虽然它不算最近邻，但在情绪张力和仪式感上与你的源轨有明显映射，所以依然值得被看见。",
    tags: ["复调", "仪式感", "高新颖度", "情绪映射"],
    audio_url: "",
  },
];

const refs = {
  audioInput: document.querySelector("#audioInput"),
  modeSelect: document.querySelector("#modeSelect"),
  lensSelect: document.querySelector("#lensSelect"),
  runDemoButton: document.querySelector("#runDemoButton"),
  useSampleButton: document.querySelector("#useSampleButton"),
  heroPlayButton: document.querySelector("#heroPlayButton"),
  showcaseTabs: [...document.querySelectorAll(".showcase-tabs button")],
  backendStatus: document.querySelector("#backendStatus"),
  uploadCount: document.querySelector("#uploadCount"),
  uploadedList: document.querySelector("#uploadedList"),
  waveformLabel: document.querySelector("#waveformLabel"),
  waveformBars: document.querySelector("#waveformBars"),
  waveformShell: document.querySelector(".waveform-shell"),
  sourceAudioPlayer: document.querySelector("#sourceAudioPlayer"),
  recommendationAudioPlayer: document.querySelector("#recommendationAudioPlayer"),
  progressFill: document.querySelector("#progressFill"),
  progressPercent: document.querySelector("#progressPercent"),
  stageList: [...document.querySelectorAll(".stage-item")],
  logList: document.querySelector("#logList"),
  logStatus: document.querySelector("#logStatus"),
  recommendationGrid: document.querySelector("#recommendationGrid"),
  recommendationTemplate: document.querySelector("#recommendationTemplate"),
  feedbackItemTemplate: document.querySelector("#feedbackItemTemplate"),
  detailTitle: document.querySelector("#detailTitle"),
  detailBadge: document.querySelector("#detailBadge"),
  detailReason: document.querySelector("#detailReason"),
  detailTags: document.querySelector("#detailTags"),
  detailSimilarity: document.querySelector("#detailSimilarity"),
  detailBridge: document.querySelector("#detailBridge"),
  detailNovelty: document.querySelector("#detailNovelty"),
  detailBpm: document.querySelector("#detailBpm"),
  summaryTrack: document.querySelector("#summaryTrack"),
  summaryTrackMeta: document.querySelector("#summaryTrackMeta"),
  summaryMode: document.querySelector("#summaryMode"),
  summaryLens: document.querySelector("#summaryLens"),
  summaryAxis: document.querySelector("#summaryAxis"),
  summaryAxisMeta: document.querySelector("#summaryAxisMeta"),
  factorSimilarity: document.querySelector("#factorSimilarity"),
  factorBridge: document.querySelector("#factorBridge"),
  factorNovelty: document.querySelector("#factorNovelty"),
  factorConfidence: document.querySelector("#factorConfidence"),
  factorSimilarityBar: document.querySelector("#factorSimilarityBar"),
  factorBridgeBar: document.querySelector("#factorBridgeBar"),
  factorNoveltyBar: document.querySelector("#factorNoveltyBar"),
  factorConfidenceBar: document.querySelector("#factorConfidenceBar"),
  evidenceNote: document.querySelector("#evidenceNote"),
  insightRhythm: document.querySelector("#insightRhythm"),
  insightTimbre: document.querySelector("#insightTimbre"),
  insightNovelty: document.querySelector("#insightNovelty"),
  insightConfidence: document.querySelector("#insightConfidence"),
  sourceNode: document.querySelector("#sourceNode"),
  mapNodes: [...document.querySelectorAll(".map-node-rec")],
  bridgePath: document.querySelector("#bridgePath"),
  heroTrackLabel: document.querySelector("#heroTrackLabel"),
  heroModeLabel: document.querySelector("#heroModeLabel"),
  heroSignalDescription: document.querySelector("#heroSignalDescription"),
  heroEmbeddingDim: document.querySelector("#heroEmbeddingDim"),
  heroBridgeScore: document.querySelector("#heroBridgeScore"),
  heroFeedbackCount: document.querySelector("#heroFeedbackCount"),
  registerForm: document.querySelector("#registerForm"),
  registerStatus: document.querySelector("#registerStatus"),
  feedbackForm: document.querySelector("#feedbackForm"),
  selectedTrackInput: document.querySelector("#selectedTrackInput"),
  ratingInput: document.querySelector("#ratingInput"),
  ratingValue: document.querySelector("#ratingValue"),
  feedbackStatus: document.querySelector("#feedbackStatus"),
  feedbackList: document.querySelector("#feedbackList"),
  feedbackAggregate: document.querySelector("#feedbackAggregate"),
};

const state = {
  usingApi: false,
  providerMode: "local-fallback",
  scene: "intake",
  mode: "bridge",
  lens: "rhythm",
  profile: loadLocal("echoPrototypeProfile", null),
  feedback: loadLocal("echoPrototypeFeedback", []),
  uploads: [],
  activeUpload: fallbackTrack,
  sampleTrack: null,
  analysis: null,
  recommendations: fallbackRecommendations,
  selectedRecommendation: fallbackRecommendations[0],
  stats: {
    uploads: 0,
    analyses: 0,
    feedback_count: 0,
    profile_count: 0,
    average_rating: null,
  },
  isRunning: false,
};

initialize();

async function initialize() {
  bindEvents();
  hydrateLocalProfile();
  attachWaveformAudioEffects();
  renderWaveform(state.activeUpload.waveform);
  renderUploads();
  renderFeedback();
  renderRecommendations(state.recommendations);
  setScene("intake");
  updateHero();
  updateSummaries();
  await bootstrapFromServer();
}

function bindEvents() {
  refs.audioInput?.addEventListener("change", async (event) => {
    const files = [...(event.target.files || [])];
    if (!files.length) {
      return;
    }
    await handleAudioSelection(files);
    event.target.value = "";
  });

  refs.modeSelect?.addEventListener("change", () => {
    state.mode = refs.modeSelect.value;
    updateHero();
    updateSummaries();
  });

  refs.lensSelect?.addEventListener("change", () => {
    state.lens = refs.lensSelect.value;
    updateHero();
    updateSummaries();
  });

  refs.runDemoButton?.addEventListener("click", () => {
    runDemoFlow();
  });

  refs.heroPlayButton?.addEventListener("click", () => {
    runDemoFlow();
  });

  refs.useSampleButton?.addEventListener("click", async () => {
    await useSampleTrack();
  });

  refs.showcaseTabs.forEach((button) => {
    button.addEventListener("click", () => {
      setScene(button.dataset.scene || "intake");
    });
  });

  refs.registerForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await handleRegister(new FormData(event.currentTarget));
  });

  refs.feedbackForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await handleFeedback(new FormData(event.currentTarget));
  });

  refs.ratingInput?.addEventListener("input", () => {
    refs.ratingValue.textContent = `${refs.ratingInput.value} / 5`;
  });
}

function hydrateLocalProfile() {
  if (!state.profile || !refs.registerForm) {
    updateRegisterStatus();
    return;
  }

  const nameInput = refs.registerForm.elements.namedItem("name");
  const emailInput = refs.registerForm.elements.namedItem("email");
  const roleInput = refs.registerForm.elements.namedItem("role");

  if (nameInput) {
    nameInput.value = state.profile.name || "";
  }
  if (emailInput) {
    emailInput.value = state.profile.email || "";
  }
  if (roleInput) {
    roleInput.value = state.profile.role || "listener";
  }

  updateRegisterStatus();
}

async function bootstrapFromServer() {
  try {
    const payload = await getJson(api.bootstrap);
    state.usingApi = true;
    state.providerMode = payload.provider || "local-fallback";
    state.stats = mergeStats(payload.stats);

    if (payload.profile) {
      state.profile = {
        id: payload.profile.id,
        name: payload.profile.name,
        email: payload.profile.email,
        role: payload.profile.role,
      };
      saveLocal("echoPrototypeProfile", state.profile);
      hydrateLocalProfile();
    }

    if (Array.isArray(payload.feedback) && payload.feedback.length) {
      state.feedback = payload.feedback;
      saveLocal("echoPrototypeFeedback", state.feedback);
      renderFeedback();
    }

    if (payload.sample_track) {
      state.sampleTrack = {
        id: payload.sample_track.id,
        name: payload.sample_track.title,
        title: payload.sample_track.title,
        origin: payload.sample_track.origin,
        size_mb: null,
        descriptor: payload.sample_track.descriptor,
        waveform: buildFallbackWaveform(payload.sample_track.id),
        audio_url: payload.sample_track.audio_url,
      };
    }

    refs.backendStatus.textContent =
      state.providerMode === "external-configured"
        ? "系统已连接真实后端，当前可调用外部嵌入服务并保存注册与反馈。"
        : "系统已连接后端，当前使用本地回退嵌入逻辑；你仍然可以完整上传、分析、试听和反馈。";
  } catch (error) {
    state.usingApi = false;
    refs.backendStatus.textContent = "当前未连接后端，页面会用演示数据完成同样的体验流程。";
  }

  updateHero();
  updateSummaries();
}

async function handleAudioSelection(files) {
  refs.backendStatus.textContent = state.usingApi
    ? "正在接收你选择的音频并准备分析。"
    : "正在载入本地音频，稍后会用演示逻辑完成推荐。";

  const uploads = [];

  for (const file of files) {
    try {
      const uploaded = state.usingApi ? await uploadFileToApi(file) : await createLocalTrackFromFile(file);
      uploads.push(uploaded);
    } catch (error) {
      refs.backendStatus.textContent = `有一段音频没有成功载入：${error.message}`;
    }
  }

  if (!uploads.length) {
    renderUploads();
    return;
  }

  state.uploads = dedupeUploads([...uploads, ...state.uploads]);
  state.activeUpload = uploads[0];
  state.analysis = null;
  state.recommendations = fallbackRecommendations;
  state.selectedRecommendation = fallbackRecommendations[0];

  renderWaveform(state.activeUpload.waveform);
  syncSourceAudio(state.activeUpload.audio_url);
  renderUploads();
  renderRecommendations(state.recommendations);
  updateHero();
  updateSummaries();
  setScene("intake");

  refs.backendStatus.textContent = `已载入 ${uploads.length} 段音频，可以开始生成推荐。`;
}

async function useSampleTrack() {
  if (!state.sampleTrack && !state.usingApi) {
    state.activeUpload = fallbackTrack;
    state.uploads = dedupeUploads([fallbackTrack, ...state.uploads]);
    renderWaveform(state.activeUpload.waveform);
    syncSourceAudio(state.activeUpload.audio_url);
    renderUploads();
    refs.backendStatus.textContent = "已切换到示例音轨，可以直接运行演示。";
    updateHero();
    updateSummaries();
    return;
  }

  if (state.sampleTrack && state.usingApi) {
    refs.backendStatus.textContent = "正在载入系统示例音轨。";
    try {
      const response = await fetch(state.sampleTrack.audio_url);
      if (!response.ok) {
        throw new Error("示例音轨读取失败");
      }
      const blob = await response.blob();
      const file = new File([blob], `${sanitizeName(state.sampleTrack.title)}.mp3`, {
        type: blob.type || "audio/mpeg",
      });
      const uploaded = await uploadFileToApi(file);
      uploaded.name = state.sampleTrack.title;
      uploaded.descriptor = state.sampleTrack.descriptor;
      state.activeUpload = uploaded;
      state.uploads = dedupeUploads([uploaded, ...state.uploads]);
      renderWaveform(uploaded.waveform);
      syncSourceAudio(uploaded.audio_url);
      renderUploads();
      refs.backendStatus.textContent = "示例音轨已经准备好，可以直接开始推荐。";
      updateHero();
      updateSummaries();
      return;
    } catch (error) {
      refs.backendStatus.textContent = `示例音轨暂时不可用：${error.message}`;
    }
  }

  state.activeUpload = fallbackTrack;
  state.uploads = dedupeUploads([fallbackTrack, ...state.uploads]);
  renderWaveform(state.activeUpload.waveform);
  syncSourceAudio(state.activeUpload.audio_url);
  renderUploads();
  refs.backendStatus.textContent = "已切换到内置示例音轨，可以直接运行演示。";
  updateHero();
  updateSummaries();
}

async function runDemoFlow() {
  if (state.isRunning) {
    return;
  }

  state.mode = refs.modeSelect?.value || state.mode;
  state.lens = refs.lensSelect?.value || state.lens;
  updateHero();
  updateSummaries();

  if (!state.activeUpload) {
    await useSampleTrack();
  }

  if (!state.activeUpload) {
    return;
  }

  state.isRunning = true;
  refs.runDemoButton.disabled = true;
  refs.heroPlayButton.disabled = true;
  refs.logStatus.textContent = "处理中";
  refs.logList.innerHTML = "";
  resetProgress();

  try {
    let payload;
    if (state.usingApi && state.activeUpload.id && String(state.activeUpload.id).startsWith("upload_")) {
      payload = await postJson(api.analyze, {
        upload_id: state.activeUpload.id,
        mode: state.mode,
        lens: state.lens,
        top_k: 3,
      });
      refs.backendStatus.textContent = payload.analysis.provider_warning
        ? `已完成分析，当前使用回退嵌入逻辑：${payload.analysis.provider_warning}`
        : "分析完成，推荐结果已经准备好。";
    } else {
      payload = buildLocalAnalysis(state.activeUpload, state.mode, state.lens);
      refs.backendStatus.textContent = "演示分析完成，已生成桥接推荐结果。";
    }

    state.analysis = payload.analysis;
    state.activeUpload = payload.upload;
    state.recommendations = payload.recommendations;
    state.stats = mergeStats(payload.stats);

    renderWaveform(state.activeUpload.waveform);
    syncSourceAudio(state.activeUpload.audio_url);
    await playStages(payload.stages);

    renderRecommendations(payload.recommendations);
    updateHero();
    updateSummaries();
    setScene("recommend");
  } catch (error) {
    refs.backendStatus.textContent = `这次分析没有成功完成：${error.message}`;
    refs.logStatus.textContent = "失败";
  } finally {
    state.isRunning = false;
    refs.runDemoButton.disabled = false;
    refs.heroPlayButton.disabled = false;
  }
}

async function handleRegister(formData) {
  const payload = {
    name: String(formData.get("name") || "").trim(),
    email: String(formData.get("email") || "").trim(),
    role: String(formData.get("role") || "listener"),
  };

  if (!payload.name || !payload.email) {
    refs.registerStatus.textContent = "请先填写姓名和邮箱。";
    return;
  }

  try {
    if (state.usingApi) {
      const result = await postJson(api.register, payload);
      state.profile = {
        id: result.profile.id,
        name: result.profile.name,
        email: result.profile.email,
        role: result.profile.role,
      };
      state.stats.profile_count = Math.max(state.stats.profile_count || 0, 1);
      refs.registerStatus.textContent = "听众档案已保存，后续推荐会带上你的反馈偏好。";
    } else {
      state.profile = {
        id: `local-profile-${Date.now()}`,
        ...payload,
      };
      refs.registerStatus.textContent = "档案已保存在当前浏览器里，下次打开时仍会保留。";
    }

    saveLocal("echoPrototypeProfile", state.profile);
    updateHero();
  } catch (error) {
    refs.registerStatus.textContent = `档案暂时没有保存成功：${error.message}`;
  }
}

async function handleFeedback(formData) {
  const comment = String(formData.get("comment") || "").trim();
  const rating = Number(formData.get("rating") || refs.ratingInput.value || 4);
  const currentRecommendation = state.selectedRecommendation;

  if (!currentRecommendation) {
    refs.feedbackStatus.textContent = "请先运行一次推荐流程，再提交反馈。";
    return;
  }

  const payload = {
    track: currentRecommendation.title,
    recommendation_id: currentRecommendation.id,
    rating,
    comment,
    profile_id: state.profile?.id || null,
  };

  try {
    if (state.usingApi) {
      const result = await postJson(api.feedback, payload);
      state.feedback = result.recent_feedback || [];
      state.stats = mergeStats(result.stats);
      refs.feedbackStatus.textContent = "反馈已保存。之后你看到的推荐会越来越懂你的口味。";
    } else {
      state.feedback = [
        {
          id: `local-feedback-${Date.now()}`,
          profile_name: state.profile?.name || "匿名听众",
          track: payload.track,
          recommendation_id: payload.recommendation_id,
          rating: payload.rating,
          comment: payload.comment,
          created_at: formatNow(),
        },
        ...state.feedback,
      ].slice(0, 8);
      state.stats.feedback_count = state.feedback.length;
      state.stats.average_rating = averageRating(state.feedback);
      refs.feedbackStatus.textContent = "反馈已保存在当前浏览器里。";
    }

    saveLocal("echoPrototypeFeedback", state.feedback);
    renderFeedback();
    updateHero();
  } catch (error) {
    refs.feedbackStatus.textContent = `反馈保存失败：${error.message}`;
  }
}

function renderWaveform(points) {
  const values = Array.isArray(points) && points.length ? points : buildFallbackWaveform();
  refs.waveformBars.innerHTML = "";

  values.slice(0, 64).forEach((value, index) => {
    const bar = document.createElement("span");
    const level = clampNumber(value, 0.1, 1);
    bar.style.height = `${Math.round(22 + level * 70)}px`;
    bar.style.opacity = `${0.55 + level * 0.45}`;
    bar.style.animationDelay = `${(index % 12) * 0.08}s`;
    bar.style.animationDuration = `${2.4 + (index % 7) * 0.15}s`;
    refs.waveformBars.appendChild(bar);
  });

  refs.waveformLabel.textContent = state.activeUpload?.name || "尚未载入";
}

function renderUploads() {
  const uploads = state.uploads.length ? state.uploads : (state.activeUpload ? [state.activeUpload] : []);
  refs.uploadCount.textContent = `${uploads.length} 个文件`;

  if (!uploads.length) {
    refs.uploadedList.innerHTML = '<p class="empty-state">还没有选择音乐。你可以上传本地文件，或者直接用示例音轨启动演示。</p>';
    return;
  }

  refs.uploadedList.innerHTML = "";

  uploads.forEach((track) => {
    const item = document.createElement("article");
    item.className = `uploaded-track${track.id === state.activeUpload?.id ? " is-current" : ""}`;
    item.tabIndex = 0;
    item.innerHTML = `
      <strong>${escapeHtml(track.name || track.title || "未命名音轨")}</strong>
      <span>${escapeHtml(track.descriptor || "适合进入分析流程的音乐输入")}</span>
      <span>${track.size_mb ? `${track.size_mb} MB` : "示例音轨"}${track.audio_url ? " · 可试听" : ""}</span>
    `;

    item.addEventListener("click", () => {
      state.activeUpload = track;
      renderUploads();
      renderWaveform(track.waveform);
      syncSourceAudio(track.audio_url);
      updateHero();
      updateSummaries();
    });

    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        item.click();
      }
    });

    refs.uploadedList.appendChild(item);
  });
}

function renderRecommendations(recommendations) {
  refs.recommendationGrid.innerHTML = "";
  const items = Array.isArray(recommendations) && recommendations.length ? recommendations : fallbackRecommendations;

  items.forEach((recommendation) => {
    const node = refs.recommendationTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.recommendationId = recommendation.id;
    node.querySelector(".rec-origin").textContent = recommendation.origin;
    node.querySelector(".rec-title").textContent = recommendation.title;
    node.querySelector(".rec-score").textContent = `${recommendation.score}`;
    node.querySelector(".rec-summary").textContent = recommendation.summary;
    node.querySelector(".rec-bridge").textContent = decimal(recommendation.bridge);
    node.querySelector(".rec-novelty").textContent = decimal(recommendation.novelty);
    node.querySelector(".rec-bpm").textContent = recommendation.bpm;

    const tagRow = node.querySelector(".rec-tags");
    recommendation.tags.forEach((tag) => {
      const tagEl = document.createElement("span");
      tagEl.textContent = tag;
      tagRow.appendChild(tagEl);
    });

    const select = () => selectRecommendation(recommendation.id);
    node.addEventListener("click", select);
    node.querySelector(".rec-select").addEventListener("click", (event) => {
      event.stopPropagation();
      select();
    });

    refs.recommendationGrid.appendChild(node);
  });

  const nextSelection =
    items.find((item) => item.id === state.selectedRecommendation?.id) ||
    items[0] ||
    null;

  state.recommendations = items;
  state.selectedRecommendation = nextSelection;
  if (nextSelection) {
    selectRecommendation(nextSelection.id, { silentScroll: true });
  }
}

function selectRecommendation(recommendationId, options = {}) {
  const recommendation = state.recommendations.find((item) => item.id === recommendationId);
  if (!recommendation) {
    return;
  }

  state.selectedRecommendation = recommendation;

  [...refs.recommendationGrid.children].forEach((card) => {
    card.classList.toggle("is-selected", card.dataset.recommendationId === recommendationId);
  });

  refs.detailTitle.textContent = recommendation.title;
  refs.detailBadge.textContent = recommendation.bridge >= 0.85 ? "最佳桥接" : recommendation.novelty >= 0.72 ? "探索向推荐" : "平衡推荐";
  refs.detailReason.textContent = recommendation.reason;
  refs.detailSimilarity.textContent = decimal(recommendation.similarity);
  refs.detailBridge.textContent = decimal(recommendation.bridge);
  refs.detailNovelty.textContent = `${Math.round(recommendation.novelty * 100)}%`;
  refs.detailBpm.textContent = `${state.analysis?.source_bpm || 92} vs ${recommendation.bpm}`;
  refs.selectedTrackInput.value = recommendation.title;

  refs.detailTags.innerHTML = "";
  recommendation.tags.forEach((tag) => {
    const tagEl = document.createElement("span");
    tagEl.textContent = tag;
    refs.detailTags.appendChild(tagEl);
  });

  refs.factorSimilarity.textContent = decimal(recommendation.similarity);
  refs.factorBridge.textContent = decimal(recommendation.bridge);
  refs.factorNovelty.textContent = decimal(recommendation.novelty);
  refs.factorConfidence.textContent = decimal(recommendation.confidence);

  refs.factorSimilarityBar.style.width = `${Math.round(recommendation.similarity * 100)}%`;
  refs.factorBridgeBar.style.width = `${Math.round(recommendation.bridge * 100)}%`;
  refs.factorNoveltyBar.style.width = `${Math.round(recommendation.novelty * 100)}%`;
  refs.factorConfidenceBar.style.width = `${Math.round(recommendation.confidence * 100)}%`;

  refs.evidenceNote.textContent = `${recommendation.summary} 当前主解释轴是“${recommendation.axis}”。`;
  refs.insightRhythm.textContent = decimal(lensValue(recommendation, "rhythm"));
  refs.insightTimbre.textContent = decimal(lensValue(recommendation, "timbre"));
  refs.insightNovelty.textContent = `+${Math.round(recommendation.novelty * 100 - 40)}%`;
  refs.insightConfidence.textContent = `${Math.round(recommendation.confidence * 100)}%`;

  syncRecommendationAudio(recommendation.audio_url);
  updateLatentMap();
  updateHero();
  updateSummaries();

  if (!options.silentScroll) {
    refs.detailTitle.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function renderFeedback() {
  const items = Array.isArray(state.feedback) ? state.feedback : [];
  if (!items.length) {
    refs.feedbackList.innerHTML = '<p class="empty-state">这里会显示你的评分和评论，它们会逐步帮助系统学会更适合你的跨文化连接方式。</p>';
    refs.feedbackAggregate.textContent = "平均评分：--";
    return;
  }

  refs.feedbackList.innerHTML = "";
  items.forEach((entry) => {
    const node = refs.feedbackItemTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".feedback-track").textContent = entry.track;
    node.querySelector(".feedback-rating").textContent = `${entry.rating} / 5`;
    node.querySelector(".feedback-comment").textContent = entry.comment || "这位听众没有留下文字评论。";
    node.querySelector(".feedback-meta").textContent = `${entry.profile_name || "匿名听众"} · ${entry.created_at || "刚刚"}`;
    refs.feedbackList.appendChild(node);
  });

  refs.feedbackAggregate.textContent = `平均评分：${state.stats.average_rating ?? averageRating(items)}`;
}

function updateHero() {
  refs.heroModeLabel.textContent = modeLabels[state.mode];
  refs.heroTrackLabel.textContent = state.activeUpload?.name || state.sampleTrack?.title || "等待上传本地音乐";
  refs.heroSignalDescription.textContent = `${sceneNotes[state.scene]} 当前偏好是 ${lensLabels[state.lens]} 视角。`;
  refs.heroEmbeddingDim.textContent = String(state.analysis?.embedding_dim || 1024);
  refs.heroBridgeScore.textContent = decimal(state.selectedRecommendation?.bridge || state.analysis?.bridge_score || 0.81);
  refs.heroFeedbackCount.textContent = String(state.stats.feedback_count || 0);
}

function updateSummaries() {
  refs.summaryTrack.textContent = state.activeUpload?.name
    ? `当前源轨：${state.activeUpload.name}`
    : "当前源轨：等待输入";
  refs.summaryTrackMeta.textContent = state.activeUpload?.descriptor || "上传音乐后，这里会显示系统整理出的音频描述。";
  refs.summaryMode.textContent = modeLabels[state.mode];
  refs.summaryLens.textContent = `视角：${lensLabels[state.lens]}`;
  refs.summaryAxis.textContent = state.selectedRecommendation?.axis || "等待生成主桥接轴";
  refs.summaryAxisMeta.textContent = state.selectedRecommendation
    ? `系统把你带向 ${state.selectedRecommendation.origin}，并保留了可解释的桥接线索。`
    : "系统会在这里解释这次推荐最关键的连接方式。";
}

function updateRegisterStatus() {
  refs.registerStatus.textContent = state.profile
    ? `当前档案：${state.profile.name} · ${state.profile.email}`
    : "创建一个听众档案后，推荐和反馈会更容易连续起来。";
}

function setScene(scene) {
  state.scene = scene;
  document.body.dataset.scene = scene;

  refs.showcaseTabs.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.scene === scene);
  });

  const activeIndexes = {
    intake: [0],
    embedding: [1, 2],
    recommend: [3, 4],
  }[scene] || [0];

  refs.stageList.forEach((item, index) => {
    item.classList.toggle("is-active", activeIndexes.includes(index));
  });

  updateHero();
}

async function playStages(stages) {
  const list = Array.isArray(stages) && stages.length ? stages : buildLocalStages(state.activeUpload?.name || "这段音乐", state.mode, state.lens);
  refs.logList.innerHTML = "";

  for (let index = 0; index < list.length; index += 1) {
    const stage = list[index];
    refs.progressFill.style.width = `${stage.progress}%`;
    refs.progressPercent.textContent = `${String(stage.progress).padStart(2, "0")}%`;
    refs.logStatus.textContent = stage.label;
    setScene(stage.scene);

    refs.stageList.forEach((item, stageIndex) => {
      item.classList.toggle("is-complete", stageIndex < index);
      item.classList.toggle("is-active", stageIndex === index);
    });

    const log = document.createElement("article");
    log.className = "log-entry";
    log.innerHTML = `
      <strong>${escapeHtml(stage.label)}</strong>
      <span>${escapeHtml(stage.detail)}</span>
    `;
    refs.logList.prepend(log);

    await sleep(index === list.length - 1 ? 220 : 460);
  }

  refs.logStatus.textContent = "已完成";
}

function updateLatentMap() {
  refs.sourceNode.textContent = cropLabel(state.activeUpload?.name || "源轨", 10);

  refs.mapNodes.forEach((node, index) => {
    const recommendation = state.recommendations[index];
    if (!recommendation) {
      node.textContent = `候选 ${index + 1}`;
      return;
    }

    const similarity = clampNumber(recommendation.similarity, 0.42, 0.97);
    const novelty = clampNumber(recommendation.novelty, 0.42, 0.97);
    const left = 230 + Math.round(similarity * 170) - index * 16;
    const top = 52 + Math.round((1 - novelty) * 180) + index * 34;

    node.textContent = cropLabel(recommendation.title, 8);
    node.style.left = `${left}px`;
    node.style.top = `${top}px`;
    node.style.transform = recommendation.id === state.selectedRecommendation?.id ? "scale(1.04)" : "scale(1)";
  });

  if (state.selectedRecommendation) {
    const index = state.recommendations.findIndex((item) => item.id === state.selectedRecommendation.id);
    const target = refs.mapNodes[index] || refs.mapNodes[0];
    const targetLeft = parseFloat(target.style.left || "356");
    const targetTop = parseFloat(target.style.top || "82");
    refs.bridgePath.setAttribute(
      "d",
      `M120 160 C220 ${Math.max(40, targetTop - 70)}, 300 ${Math.max(60, targetTop - 36)}, ${targetLeft - 12} ${targetTop + 20}`,
    );
  }
}

function syncSourceAudio(audioUrl) {
  if (!audioUrl) {
    refs.sourceAudioPlayer.removeAttribute("src");
    refs.sourceAudioPlayer.load();
    refs.sourceAudioPlayer.hidden = true;
    return;
  }

  refs.sourceAudioPlayer.hidden = false;
  if (refs.sourceAudioPlayer.src !== new URL(audioUrl, window.location.href).href) {
    refs.sourceAudioPlayer.src = audioUrl;
  }
}

function syncRecommendationAudio(audioUrl) {
  if (!audioUrl) {
    refs.recommendationAudioPlayer.removeAttribute("src");
    refs.recommendationAudioPlayer.load();
    refs.recommendationAudioPlayer.hidden = true;
    return;
  }

  refs.recommendationAudioPlayer.hidden = false;
  if (refs.recommendationAudioPlayer.src !== new URL(audioUrl, window.location.href).href) {
    refs.recommendationAudioPlayer.src = audioUrl;
  }
}

function attachWaveformAudioEffects() {
  const activate = () => refs.waveformShell?.classList.add("is-playing");
  const deactivate = () => refs.waveformShell?.classList.remove("is-playing");

  refs.sourceAudioPlayer?.addEventListener("play", activate);
  refs.sourceAudioPlayer?.addEventListener("pause", deactivate);
  refs.sourceAudioPlayer?.addEventListener("ended", deactivate);
}

async function uploadFileToApi(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(api.upload, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`上传失败（${response.status}）`);
  }

  const result = await response.json();
  return result.upload;
}

async function createLocalTrackFromFile(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  return {
    id: `local-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: file.name,
    size_mb: round(file.size / (1024 * 1024), 2),
    descriptor: guessDescriptor(file.name),
    waveform: buildWaveformFromBytes(bytes),
    audio_url: URL.createObjectURL(file),
  };
}

function buildLocalAnalysis(upload, mode, lens) {
  const sourceSeed = hashString(`${upload.name}|${mode}|${lens}|${upload.descriptor}`);
  const recommendations = fallbackRecommendations
    .map((item, index) => {
      const jitter = seededValue(sourceSeed + index * 19);
      const bridgeBias = mode === "bridge" ? 0.08 : mode === "precision" ? 0.03 : -0.02;
      const noveltyBias = mode === "novelty" ? 0.09 : mode === "precision" ? -0.05 : 0.01;
      const similarityBias = lens === "timbre" ? 0.05 : lens === "emotion" ? 0.02 : 0.04;
      const bridge = clampNumber(item.bridge + bridgeBias + jitter * 0.03, 0.42, 0.97);
      const novelty = clampNumber(item.novelty + noveltyBias + jitter * 0.04, 0.42, 0.97);
      const similarity = clampNumber(item.similarity + similarityBias + jitter * 0.03, 0.42, 0.97);
      const confidence = clampNumber(item.confidence + jitter * 0.02, 0.42, 0.97);
      return {
        ...item,
        bridge: round(bridge, 2),
        novelty: round(novelty, 2),
        similarity: round(similarity, 2),
        confidence: round(confidence, 2),
        score: Math.round((bridge * 0.4 + novelty * 0.2 + similarity * 0.4) * 100),
        axis: chooseAxis(lens, item),
        summary: buildSummary(upload, item, bridge, novelty),
        reason: buildReason(upload, item, bridge, similarity),
      };
    })
    .sort((left, right) => right.score - left.score);

  return {
    analysis: {
      id: `local-analysis-${Date.now()}`,
      mode,
      lens,
      embedding_dim: 1024,
      bridge_score: recommendations[0].bridge,
      source_bpm: 86 + Math.round(seededValue(sourceSeed) * 22),
      provider: "local-fallback",
    },
    upload: {
      ...upload,
      waveform: upload.waveform || buildFallbackWaveform(upload.name),
    },
    stages: buildLocalStages(upload.name, mode, lens),
    recommendations,
    stats: {
      ...state.stats,
      uploads: Math.max(state.stats.uploads || 0, state.uploads.length || 1),
      analyses: (state.stats.analyses || 0) + 1,
    },
  };
}

function buildLocalStages(sourceName, mode, lens) {
  return [
    {
      label: "接收本地音轨",
      detail: `已读取《${sourceName}》，正在整理输入描述和波形摘要。`,
      progress: 16,
      scene: "intake",
    },
    {
      label: "生成共享表示",
      detail: "系统正在把这段音乐投到共享嵌入空间中。",
      progress: 38,
      scene: "embedding",
    },
    {
      label: "寻找文化桥接线索",
      detail: `当前更强调 ${lensLabels[lens]}，正在提炼可讲述的连接轴。`,
      progress: 62,
      scene: "embedding",
    },
    {
      label: "执行轻量重排序",
      detail: `当前采用 ${modeLabels[mode]}，会平衡熟悉感和新发现。`,
      progress: 84,
      scene: "recommend",
    },
    {
      label: "整理推荐说明",
      detail: "推荐卡片、试听入口和反馈位都已经准备好了。",
      progress: 100,
      scene: "recommend",
    },
  ];
}

function resetProgress() {
  refs.progressFill.style.width = "0%";
  refs.progressPercent.textContent = "00%";
  refs.stageList.forEach((item, index) => {
    item.classList.remove("is-complete");
    item.classList.toggle("is-active", index === 0);
  });
}

async function getJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`请求失败（${response.status}）`);
  }
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`请求失败（${response.status}）`);
  }

  return response.json();
}

function loadLocal(key, fallbackValue) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallbackValue;
  } catch (error) {
    return fallbackValue;
  }
}

function saveLocal(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn("localStorage write failed", error);
  }
}

function mergeStats(nextStats) {
  return {
    uploads: nextStats?.uploads ?? state.stats.uploads ?? 0,
    analyses: nextStats?.analyses ?? state.stats.analyses ?? 0,
    feedback_count: nextStats?.feedback_count ?? state.stats.feedback_count ?? 0,
    profile_count: nextStats?.profile_count ?? state.stats.profile_count ?? 0,
    average_rating: nextStats?.average_rating ?? state.stats.average_rating ?? null,
  };
}

function buildWaveformFromBytes(bytes) {
  if (!bytes?.length) {
    return buildFallbackWaveform();
  }

  const bins = 64;
  const size = Math.max(1, Math.floor(bytes.length / bins));
  const values = [];
  for (let index = 0; index < bins; index += 1) {
    const start = index * size;
    const chunk = bytes.slice(start, start + size);
    let sum = 0;
    for (let inner = 0; inner < chunk.length; inner += 1) {
      sum += Math.abs(chunk[inner] - 128);
    }
    values.push(round(clampNumber(sum / Math.max(1, chunk.length) / 110, 0.08, 1), 3));
  }
  return values;
}

function buildFallbackWaveform(seedInput = "echo") {
  const seed = hashString(String(seedInput));
  return Array.from({ length: 64 }, (_, index) => {
    const primary = seededValue(seed + index * 17);
    const secondary = seededValue(seed * 3 + index * 29);
    return round(clampNumber(0.24 + primary * 0.48 + secondary * 0.2, 0.1, 0.96), 3);
  });
}

function chooseAxis(lens, recommendation) {
  if (lens === "rhythm") {
    return recommendation.id === "fallback-3" ? "声部层叠 + 推进张力" : "拨弦音色 + 循环律动";
  }
  if (lens === "timbre") {
    return recommendation.id === "fallback-3" ? "复调纹理 + 共鸣层次" : "乐器纹理 + 共振色彩";
  }
  return "情绪轮廓 + 仪式氛围";
}

function buildSummary(upload, recommendation, bridge, novelty) {
  if (bridge >= 0.85) {
    return `这条候选和《${cropLabel(upload.name, 8)}》之间的连接最顺畅，很适合先作为桥接入口。`;
  }
  if (novelty >= 0.72) {
    return "它会把你带到更远一点的文化空间，但仍然保留一条可以追踪的连接线。";
  }
  return "它在熟悉感和新发现之间比较平衡，适合作为稳妥的下一步。";
}

function buildReason(upload, recommendation, bridge, similarity) {
  return `系统选择《${recommendation.title}》，是因为它在“${recommendation.axis}”这条轴上与《${cropLabel(upload.name, 12)}》形成了 ${decimal(bridge)} 的桥接强度，并保留了 ${decimal(similarity)} 的邻近度。`;
}

function lensValue(recommendation, lens) {
  if (lens === "rhythm") {
    return clampNumber(recommendation.bridge - 0.04, 0.42, 0.97);
  }
  if (lens === "timbre") {
    return clampNumber(recommendation.similarity + 0.03, 0.42, 0.97);
  }
  return clampNumber(recommendation.novelty + 0.08, 0.42, 0.97);
}

function guessDescriptor(filename) {
  const lowered = String(filename || "").toLowerCase();
  if (/(drum|beat|perk|tabla|kick)/.test(lowered)) {
    return "鼓点突出、舞蹈驱动型能量";
  }
  if (/(voice|vocal|chant|opera)/.test(lowered)) {
    return "人声主导、旋律叙述感较强";
  }
  if (/(string|guitar|oud|lute|sitar)/.test(lowered)) {
    return "拨弦共振明显、装饰音比较丰富";
  }
  return "适合进入跨文化推荐流程的本地音乐输入";
}

function dedupeUploads(items) {
  const seen = new Set();
  return items.filter((item) => {
    const key = item.id || item.name;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function averageRating(items) {
  if (!items.length) {
    return null;
  }
  return round(items.reduce((sum, item) => sum + Number(item.rating || 0), 0) / items.length, 1);
}

function cropLabel(text, maxLength) {
  const value = String(text || "");
  return value.length > maxLength ? `${value.slice(0, maxLength)}…` : value;
}

function sanitizeName(value) {
  return String(value || "sample").replace(/[^\w\u4e00-\u9fa5-]+/g, "-");
}

function decimal(value) {
  return Number(value || 0).toFixed(2);
}

function round(value, precision) {
  const factor = 10 ** precision;
  return Math.round(value * factor) / factor;
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function hashString(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash);
}

function seededValue(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function formatNow() {
  const date = new Date();
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  const hh = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  const ss = String(date.getSeconds()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
