const statusPanel = document.getElementById("status");
const sessionPanel = document.getElementById("session");
const optionsContainer = document.getElementById("options");
const handIndexEl = document.getElementById("hand-index");
const heroCardsEl = document.getElementById("hero-cards");
const boardCardsEl = document.getElementById("board-cards");
const potEl = document.getElementById("pot");
const stackEl = document.getElementById("stack");
const handsPlayedEl = document.getElementById("hands-played");
const evLossEl = document.getElementById("ev-loss");
const profitEl = document.getElementById("profit");

const newSessionBtn = document.getElementById("new-session");
const handsInput = document.getElementById("hands");
const samplesInput = document.getElementById("samples");
const rivalSelect = document.getElementById("rival-style");

let currentSessionId = null;
let currentOptions = [];

async function startSession() {
  const hands = Number(handsInput.value) || 1;
  const mcSamples = Number(samplesInput.value) || 200;
  const rivalStyle = rivalSelect.value;

  try {
    const response = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hands,
        mc_samples: mcSamples,
        rival_style: rivalStyle,
      }),
    });
    if (!response.ok) {
      throw new Error(`Failed to start session (${response.status})`);
    }
    const state = await response.json();
    currentSessionId = state.session_id;
    renderState(state);
    showStatus("Session started", "success");
  } catch (error) {
    console.error(error);
    showStatus(error.message, "error");
  }
}

async function fetchState() {
  if (!currentSessionId) {
    return;
  }
  const response = await fetch(`/api/sessions/${currentSessionId}`);
  if (!response.ok) {
    showStatus("Session not found", "error");
    return;
  }
  const state = await response.json();
  renderState(state);
}

async function sendAction(action) {
  if (!currentSessionId) {
    showStatus("No active session", "error");
    return;
  }

  try {
    const response = await fetch(`/api/sessions/${currentSessionId}/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!response.ok) {
      throw new Error(`Action rejected (${response.status})`);
    }
    const state = await response.json();
    renderState(state);
  } catch (error) {
    console.error(error);
    showStatus(error.message, "error");
  }
}

function renderState(state) {
  if (!state || !state.node) {
    return;
  }

  handIndexEl.textContent = state.hand_index ?? 0;
  heroCardsEl.textContent = state.node.hero_cards?.join(" ") ?? "--";
  boardCardsEl.textContent = state.node.board?.join(" ") ?? "--";
  potEl.textContent = state.node.pot_bb?.toFixed(2) ?? "0.00";
  stackEl.textContent = state.node.effective_stack_bb?.toFixed(2) ?? "0.00";

  handsPlayedEl.textContent = state.summary?.hands_played ?? 0;
  evLossEl.textContent = (state.summary?.total_ev_loss_bb ?? 0).toFixed(2);
  profitEl.textContent = (state.summary?.total_profit_bb ?? 0).toFixed(2);

  currentOptions = state.node.action_options ?? [];
  optionsContainer.innerHTML = "";

  if (state.status === "completed") {
    const done = document.createElement("div");
    done.className = "completed";
    done.textContent = "Session complete. Start a new session to keep training.";
    optionsContainer.appendChild(done);
    return;
  }

  currentOptions.forEach((option, idx) => {
    const button = document.createElement("button");
    button.className = "option";
    button.textContent = `${idx + 1}. ${describeAction(option.action)} (${option.ev_delta_bb.toFixed(2)} bb)`;
    button.addEventListener("click", () => sendAction(option.action));

    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = option.description;

    const wrapper = document.createElement("div");
    wrapper.className = "option-row";
    wrapper.appendChild(button);
    wrapper.appendChild(hint);

    optionsContainer.appendChild(wrapper);
  });
}

function describeAction(action) {
  switch (action.kind) {
    case "fold":
      return "Fold";
    case "call":
      return action.size_bb ? `Call ${action.size_bb.toFixed(1)}bb` : "Call";
    case "check":
      return "Check";
    case "bet":
      return action.size_bb ? `Bet ${action.size_bb.toFixed(1)}bb` : "Bet";
    case "raise":
      return action.size_bb ? `Raise to ${action.size_bb.toFixed(1)}bb` : "Raise";
    default:
      return action.kind;
  }
}

function showStatus(message, tone = "info") {
  statusPanel.textContent = message;
  statusPanel.className = `status ${tone}`;
  if (message) {
    statusPanel.classList.remove("hidden");
  } else {
    statusPanel.classList.add("hidden");
  }
}

newSessionBtn.addEventListener("click", () => startSession());
document.addEventListener("keydown", (event) => {
  if (!currentOptions.length) {
    return;
  }
  if (event.key === "r") {
    fetchState();
    return;
  }
  if (/^[1-9]$/.test(event.key)) {
    const index = Number(event.key) - 1;
    const option = currentOptions[index];
    if (option) {
      sendAction(option.action);
    }
  }
});

startSession();
