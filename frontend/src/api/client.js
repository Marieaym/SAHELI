import axios from "axios";

// In production, set VITE_API_URL to your deployed FastAPI backend URL (e.g. Render).
// Falls back to localhost for local development.
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = axios.create({ baseURL: API_BASE, timeout: 60000 });

// Automatic retry for network errors (ECONNABORTED, ETIMEDOUT) — common on
// mobile when the Render free tier is waking up from sleep. Retries up to 2
// times with a 3-second pause, covering the typical 50-60s cold start window.
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error.config;
    if (!config) return Promise.reject(error);
    config._retryCount = config._retryCount || 0;
    const isNetworkError = !error.response && (error.code === "ECONNABORTED" || error.code === "ERR_NETWORK" || error.message === "Network Error");
    if (isNetworkError && config._retryCount < 2) {
      config._retryCount += 1;
      await new Promise((res) => setTimeout(res, 3000));
      return api(config);
    }
    return Promise.reject(error);
  }
);

const tok = localStorage.getItem("saheli_token");
if (tok) api.defaults.headers.common.Authorization = `Bearer ${tok}`;

api.interceptors.request.use((config) => {
  const t = localStorage.getItem("saheli_token");
  if (t) config.headers.Authorization = `Bearer ${t}`;
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && window.location.pathname !== "/login") {
      localStorage.removeItem("saheli_token");
      localStorage.removeItem("saheli_user");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

export const RISK_COLORS = {
  Low: "#5A6E4C",
  Medium: "#B87721",
  High: "#A8642B",
  Critical: "#A53A26",
};

export async function getDistricts() {
  const { data } = await api.get("/api/districts");
  return data;
}

export async function getDistrictDetail(name) {
  const { data } = await api.get(`/api/districts/${encodeURIComponent(name)}`);
  return data;
}

export async function getDistrictHistory(name, days = 365) {
  const { data } = await api.get(`/api/districts/${encodeURIComponent(name)}/history`, { params: { days } });
  return data;
}

export async function getZonesSummary() {
  const { data } = await api.get("/api/zones/summary");
  return data;
}

export async function getModelMetrics() {
  const { data } = await api.get("/api/model/metrics");
  return data;
}

export async function getModelValidation() {
  const { data } = await api.get("/api/model/validation");
  return data;
}

export async function getFederatedResults() {
  const { data } = await api.get("/api/model/federated");
  return data;
}

export async function getRLResults() {
  const { data } = await api.get("/api/model/rl");
  return data;
}

export async function getEdgeResults() {
  const { data } = await api.get("/api/model/edge");
  return data;
}

export async function getGroundTruthValidation() {
  const { data } = await api.get("/api/model/ground-truth");
  return data;
}

export async function getMonsoonSignal() {
  const { data } = await api.get("/api/model/monsoon-signal");
  return data;
}

export async function getAlert(district, lang) {
  const { data } = await api.get(`/api/alerts/${encodeURIComponent(district)}`, { params: { lang } });
  return data;
}

export async function getSmsStatus() {
  const { data } = await api.get("/api/alerts/sms-status");
  return data;
}

export async function sendAlert(district, phoneNumber, lang = "fr", channel = "sms") {
  const { data } = await api.post(`/api/alerts/${encodeURIComponent(district)}/send`, {
    phone_number: phoneNumber, lang, channel,
  });
  return data;
}

export async function simulateIntervention(budget, lang = "en") {
  const { data } = await api.get("/api/intervention/simulate", { params: { budget, lang } });
  return data;
}

export async function askAssistant(question, district, lang = "en") {
  const { data } = await api.post("/api/assistant/ask", { question, district, lang }, { timeout: 90000 });
  return data;
}

export async function getAssistantStatus() {
  const { data } = await api.get("/api/assistant/status");
  return data;
}

export async function getFeedEvents(limit = 50) {
  const { data } = await api.get("/api/feed/events", { params: { limit } });
  return data;
}

export function streamPipeline(district, lang, onEvent) {
  const token = localStorage.getItem("saheli_token");
  const controller = new AbortController();

  fetch(`${API_BASE}/api/pipeline/run/${encodeURIComponent(district)}?lang=${lang}`, {
    headers: { Authorization: `Bearer ${token}` },
    signal: controller.signal,
  }).then(async (response) => {
    if (!response.ok || !response.body) {
      onEvent({ step: "error", status: "error" });
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.trim();
        if (line.startsWith("data:")) {
          try {
            const json = JSON.parse(line.slice(5).trim());
            onEvent(json);
          } catch (e) {
            // ignore malformed frame
          }
        }
      }
    }
  }).catch(() => onEvent({ step: "error", status: "error" }));

  return () => controller.abort();
}

export async function simulateScenario(rainfallDeltaPct, lang = "en") {
  const { data } = await api.get("/api/scenario/simulate", { params: { rainfall_delta_pct: rainfallDeltaPct, lang } });
  return data;
}

export async function getCausalEffect() {
  const { data } = await api.get("/api/causal/effect");
  return data;
}

export async function getForecast(district) {
  const { data } = await api.get(`/api/forecast/${encodeURIComponent(district)}`);
  return data;
}

export async function getAnomaly(district) {
  const { data } = await api.get(`/api/anomaly/${encodeURIComponent(district)}`);
  return data;
}

export async function getFoodSecurityV2(district) {
  const { data } = await api.get(`/api/food-security-v2/${encodeURIComponent(district)}`);
  return data;
}

export async function getKeyMessages(lang = "en") {
  const { data } = await api.get("/api/key-messages", { params: { lang } });
  return data;
}

export async function getCommandCenter() {
  const { data } = await api.get("/api/command-center");
  return data;
}

export async function getCommandCenterBriefing(district, lang = "en") {
  const { data } = await api.get(`/api/command-center/${encodeURIComponent(district)}/briefing`, { params: { lang } });
  return data;
}

export async function predictCornLeaf(file, district = null) {
  const form = new FormData();
  form.append("file", file);
  if (district) form.append("district", district);
  const { data } = await api.post("/api/cv/corn-predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getCropScans(district) {
  const { data } = await api.get(`/api/cv/scans/${encodeURIComponent(district)}`);
  return data;
}

export async function getCountries() {
  try {
    const { data } = await api.get("/api/auth/countries");
    return data.countries?.length ? data.countries : ["Niger", "Mali", "Burkina Faso", "Chad", "Mauritania", "Senegal"];
  } catch {
    return ["Niger", "Mali", "Burkina Faso", "Chad", "Mauritania", "Senegal"];
  }
}

export async function registerUser(payload) {
  const { data } = await api.post("/api/auth/register", payload);
  return data;
}

export async function loginUser(email, password) {
  const { data } = await api.post("/api/auth/login", { email, password });
  return data;
}

export async function updateProfile(payload) {
  const { data } = await api.patch("/api/auth/me", payload);
  return data;
}

export async function getMyActivity() {
  const { data } = await api.get("/api/auth/activity");
  return data;
}

export async function downloadBrief(district, lang = "en") {
  const response = await api.get(`/api/brief/${encodeURIComponent(district)}`, { params: { lang }, responseType: "blob" });
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const a = document.createElement("a");
  a.href = url;
  const disposition = response.headers["content-disposition"] || "";
  const match = disposition.match(/filename=(.+)/);
  a.download = match ? match[1] : `SAHELI_Brief_${district}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}