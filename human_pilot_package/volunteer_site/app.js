(function () {
  const tasks = Array.isArray(window.PILOT_TASKS) ? window.PILOT_TASKS : [];
  const storageKey = "dcas_human_pilot_v1";

  const setupPanel = document.getElementById("setupPanel");
  const taskPanel = document.getElementById("taskPanel");
  const finishPanel = document.getElementById("finishPanel");
  const participantInput = document.getElementById("participantInput");
  const startBtn = document.getElementById("startBtn");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const exportBtn = document.getElementById("exportBtn");
  const backBtn = document.getElementById("backBtn");
  const form = document.getElementById("answerForm");
  const confidenceInput = document.getElementById("confidenceInput");
  const confidenceValue = document.getElementById("confidenceValue");
  const commentInput = document.getElementById("commentInput");

  const els = {
    progressText: document.getElementById("progressText"),
    taskTitle: document.getElementById("taskTitle"),
    saveState: document.getElementById("saveState"),
    seedTitle: document.getElementById("seedTitle"),
    seedMeta: document.getElementById("seedMeta"),
    seedAudio: document.getElementById("seedAudio"),
    aTitle: document.getElementById("aTitle"),
    aMeta: document.getElementById("aMeta"),
    aAudio: document.getElementById("aAudio"),
    bTitle: document.getElementById("bTitle"),
    bMeta: document.getElementById("bMeta"),
    bAudio: document.getElementById("bAudio"),
    completionSummary: document.getElementById("completionSummary"),
  };

  let state = loadState();
  let currentIndex = 0;

  function loadState() {
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return { participant_id: "", answers: {} };
      const parsed = JSON.parse(raw);
      return {
        participant_id: parsed.participant_id || "",
        answers: parsed.answers || {},
      };
    } catch (error) {
      return { participant_id: "", answers: {} };
    }
  }

  function saveState() {
    localStorage.setItem(storageKey, JSON.stringify(state));
    els.saveState.textContent = "已自动保存";
  }

  function formatMeta(item) {
    const bits = [];
    if (item.culture) bits.push(`文化/地区：${item.culture}`);
    if (item.label) bits.push(`标签：${item.label}`);
    if (item.artist) bits.push(`艺术家：${item.artist}`);
    return bits.join(" · ");
  }

  function setAudio(audioEl, src) {
    audioEl.pause();
    audioEl.removeAttribute("src");
    audioEl.load();
    audioEl.src = src;
  }

  function renderTask() {
    const task = tasks[currentIndex];
    if (!task) return;

    els.progressText.textContent = `Task ${currentIndex + 1} / ${tasks.length}`;
    els.taskTitle.textContent = task.prompt || "请比较 A/B 两个推荐候选";

    els.seedTitle.textContent = task.seed.title || task.seed.track_id || "Seed";
    els.seedMeta.textContent = formatMeta(task.seed);
    setAudio(els.seedAudio, task.seed.audio);

    els.aTitle.textContent = task.candidate_a.title || "Candidate A";
    els.aMeta.textContent = formatMeta(task.candidate_a);
    setAudio(els.aAudio, task.candidate_a.audio);

    els.bTitle.textContent = task.candidate_b.title || "Candidate B";
    els.bMeta.textContent = formatMeta(task.candidate_b);
    setAudio(els.bAudio, task.candidate_b.audio);

    form.reset();
    confidenceInput.value = "3";
    confidenceValue.textContent = "3";
    commentInput.value = "";

    const answer = state.answers[task.task_id];
    if (answer) {
      setChoice("compatible_choice", answer.compatible_choice);
      setChoice("discovery_choice", answer.discovery_choice);
      setChoice("overall_choice", answer.overall_choice);
      confidenceInput.value = answer.confidence || "3";
      confidenceValue.textContent = confidenceInput.value;
      commentInput.value = answer.comment || "";
      els.saveState.textContent = "已加载已保存答案";
    } else {
      els.saveState.textContent = "未填写";
    }

    prevBtn.disabled = currentIndex === 0;
    nextBtn.textContent = currentIndex === tasks.length - 1 ? "保存并完成" : "保存并下一题";
  }

  function setChoice(name, value) {
    if (!value) return;
    const input = form.querySelector(`input[name="${name}"][value="${value}"]`);
    if (input) input.checked = true;
  }

  function getChoice(name) {
    const input = form.querySelector(`input[name="${name}"]:checked`);
    return input ? input.value : "";
  }

  function collectAnswer() {
    const task = tasks[currentIndex];
    return {
      participant_id: state.participant_id,
      task_id: task.task_id,
      seed_track_id: task.seed.track_id || "",
      candidate_a_id: task.candidate_a.track_id || "",
      candidate_b_id: task.candidate_b.track_id || "",
      compatible_choice: getChoice("compatible_choice"),
      discovery_choice: getChoice("discovery_choice"),
      overall_choice: getChoice("overall_choice"),
      confidence: confidenceInput.value,
      comment: commentInput.value.trim(),
      saved_at: new Date().toISOString(),
    };
  }

  function validateAnswer(answer) {
    if (!answer.compatible_choice || !answer.discovery_choice || !answer.overall_choice) {
      alert("请先完成三个选择题。如果实在不确定，可以选择“差不多 / 不确定”。");
      return false;
    }
    return true;
  }

  function showTaskPanel() {
    setupPanel.classList.add("hidden");
    finishPanel.classList.add("hidden");
    taskPanel.classList.remove("hidden");
    renderTask();
  }

  function showFinishPanel() {
    taskPanel.classList.add("hidden");
    setupPanel.classList.add("hidden");
    finishPanel.classList.remove("hidden");
    const answered = tasks.filter((task) => state.answers[task.task_id]).length;
    els.completionSummary.textContent = `匿名编号：${state.participant_id}\n已完成：${answered} / ${tasks.length}\n请导出 CSV 并发送给研究者。`;
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    if (/[",\n\r]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  }

  function exportCsv() {
    const headers = [
      "participant_id",
      "task_id",
      "seed_track_id",
      "candidate_a_id",
      "candidate_b_id",
      "compatible_choice",
      "discovery_choice",
      "overall_choice",
      "confidence",
      "comment",
      "saved_at",
    ];
    const rows = tasks.map((task) => {
      const answer = state.answers[task.task_id] || {};
      return headers.map((h) => csvEscape(answer[h] || "")).join(",");
    });
    const csv = "\ufeff" + headers.join(",") + "\n" + rows.join("\n") + "\n";
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `human_pilot_${state.participant_id || "participant"}_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  startBtn.addEventListener("click", () => {
    const pid = participantInput.value.trim();
    if (!pid) {
      alert("请先填写匿名编号，例如 P01。");
      return;
    }
    state.participant_id = pid;
    saveState();
    showTaskPanel();
  });

  nextBtn.addEventListener("click", () => {
    const answer = collectAnswer();
    if (!validateAnswer(answer)) return;
    state.answers[answer.task_id] = answer;
    saveState();
    if (currentIndex >= tasks.length - 1) {
      showFinishPanel();
    } else {
      currentIndex += 1;
      renderTask();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });

  prevBtn.addEventListener("click", () => {
    const answer = collectAnswer();
    if (answer.compatible_choice || answer.discovery_choice || answer.overall_choice || answer.comment) {
      state.answers[answer.task_id] = answer;
      saveState();
    }
    currentIndex = Math.max(0, currentIndex - 1);
    renderTask();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  backBtn.addEventListener("click", () => {
    currentIndex = 0;
    showTaskPanel();
  });

  exportBtn.addEventListener("click", exportCsv);

  confidenceInput.addEventListener("input", () => {
    confidenceValue.textContent = confidenceInput.value;
  });

  const defaultParticipantId = typeof window.PILOT_DEFAULT_PARTICIPANT_ID === "string"
    ? window.PILOT_DEFAULT_PARTICIPANT_ID
    : "";
  if (!state.participant_id && defaultParticipantId) {
    state.participant_id = defaultParticipantId;
    saveState();
  }
  participantInput.value = state.participant_id || defaultParticipantId || "";
  if (!tasks.length) {
    setupPanel.innerHTML = "<h2>任务未加载</h2><p>没有找到任务数据，请联系研究者。</p>";
  }
})();
