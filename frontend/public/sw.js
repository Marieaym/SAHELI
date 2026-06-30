// SAHELI service worker — minimal and deliberately conservative.
//
// What this does NOT do, on purpose: cache any /api/* response. SAHELI is
// a live risk-classification dashboard; a cached "Critical" or "Low" risk
// reading shown to a decision-maker after the real data has changed would
// be actively misleading, not just stale. Every API call always goes to
// the real network, full stop, no exceptions, no fallback-to-cache.
//
// What this DOES do: cache the static app shell (the built JS/CSS bundle,
// icons, manifest) so the app reopens instantly from the home-screen icon
// and satisfies the browser's installability requirement for a real
// "Add to Home Screen" / "Install app" prompt. This is an app-shell cache
// for snappier reopening, not an offline-data product — SAHELI needs live
// data to mean anything, and doesn't pretend otherwise.

const CACHE_NAME = "saheli-shell-v1";
const SHELL_ASSETS = ["/", "/manifest.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never intercept API calls, auth, or anything cross-origin (the
  // backend lives on a different domain in production) — always real
  // network, always real data, no exceptions.
  if (event.request.method !== "GET" || url.pathname.startsWith("/api/") || url.origin !== self.location.origin) {
    return;
  }

  // Static app shell: try the network first (so a deployed update is
  // picked up immediately), fall back to the cached shell only if the
  // network is genuinely unreachable.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy)).catch(() => {});
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
