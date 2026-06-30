import { useEffect, useRef, useState } from "react";
import { ArrowUp, ArrowDown, Radio, Play, Pause, RotateCcw, Gauge, RefreshCw } from "lucide-react";
import { getFeedEvents } from "../api/client";
import { PageHeader, Card, LoadingState } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import { useLanguage } from "../context/LanguageContext";

const SPEEDS = [1, 3, 10, 25];
const TICK_MS = 140;
const MAX_VISIBLE_ROWS = 40;

export default function LiveFeed() {
  const { t } = useLanguage();
  const [rawEvents, setRawEvents] = useState(null);
  const [limit, setLimit] = useState(200);
  const [mode, setMode] = useState("replay"); // "replay" | "table"

  const [playCount, setPlayCount] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(1);
  const tickRef = useRef(null);

  const [lastChecked, setLastChecked] = useState(Date.now());
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    getFeedEvents(limit).then((d) => {
      setRawEvents(d.events);
      setPlayCount(0);
      setLastChecked(Date.now());
    });
  }, [limit]);

  // chronological: oldest first, so replay plays forward through real history
  const chronological = rawEvents ? [...rawEvents].reverse() : [];

  useEffect(() => {
    if (!isPlaying || !rawEvents) return;
    tickRef.current = setInterval(() => {
      setPlayCount((c) => {
        const next = c + SPEEDS[speedIdx];
        if (next >= chronological.length) {
          setIsPlaying(false);
          return chronological.length;
        }
        return next;
      });
    }, TICK_MS);
    return () => clearInterval(tickRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, speedIdx, rawEvents]);

  function handleManualRefresh() {
    setRefreshing(true);
    getFeedEvents(limit).then((d) => {
      setRawEvents(d.events);
      setLastChecked(Date.now());
      setTimeout(() => setRefreshing(false), 500);
    });
  }

  if (!rawEvents) return <LoadingState message={t("feed.loading")} />;

  const isTable = mode === "table";
  const playedSlice = isTable ? chronological : chronological.slice(0, playCount);
  const visibleEvents = [...playedSlice].reverse().slice(0, MAX_VISIBLE_ROWS);
  const nWorsening = playedSlice.filter((e) => e.direction === "worsening").length;
  const nImproving = playedSlice.filter((e) => e.direction === "improving").length;
  const progressPct = isTable ? 100 : chronological.length ? (playCount / chronological.length) * 100 : 0;
  const finished = !isTable && playCount >= chronological.length && chronological.length > 0;
  const secondsAgo = Math.max(0, Math.floor((Date.now() - lastChecked) / 1000));

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("feed.eyebrow")}
        title={t("feed.title")}
        subtitle={t("feed.subtitle")}
      />

      {/* Mode toggle */}
      <div className="flex items-center gap-2 mb-4">
        <button
          onClick={() => { setMode("replay"); setIsPlaying(false); setPlayCount(0); }}
          className={`px-3 py-1.5 rounded-full text-xs font-mono transition-colors ${isTable ? "text-muted border border-cardBorder" : "bg-primary text-white"}`}
        >
          {t("feed.replayMode")}
        </button>
        <button
          onClick={() => { setMode("table"); setIsPlaying(false); }}
          className={`px-3 py-1.5 rounded-full text-xs font-mono transition-colors ${isTable ? "bg-primary text-white" : "text-muted border border-cardBorder"}`}
        >
          {t("feed.tableMode")}
        </button>
      </div>

      {!isTable && (
        <Card className="mb-5">
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={() => {
                if (finished) { setPlayCount(0); setIsPlaying(true); }
                else setIsPlaying((p) => !p);
              }}
              className="w-10 h-10 rounded-full bg-primary text-white flex items-center justify-center flex-shrink-0 hover:scale-105 transition-transform"
            >
              {isPlaying ? <Pause size={16} /> : finished ? <RotateCcw size={16} /> : <Play size={16} />}
            </button>

            <div className="flex-1 min-w-[140px]">
              <div className="h-2 rounded-full bg-cardBorder/40 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-primary to-goldBright transition-all"
                  style={{ width: `${progressPct}%`, transitionDuration: isPlaying ? `${TICK_MS}ms` : "300ms" }}
                />
              </div>
              <div className="text-muted text-[10px] font-mono mt-1">
                {t("feed.replayProgress").replace("{n}", playCount).replace("{total}", chronological.length)}
                {chronological[Math.min(playCount, chronological.length - 1)]?.date &&
                  ` · ${chronological[Math.min(playCount, chronological.length - 1)].date.slice(0, 10)}`}
              </div>
            </div>

            <button
              onClick={() => setSpeedIdx((i) => (i + 1) % SPEEDS.length)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full border border-cardBorder text-muted text-xs font-mono hover:text-sand transition-colors flex-shrink-0"
            >
              <Gauge size={12} /> {SPEEDS[speedIdx]}x
            </button>

            {isPlaying && (
              <div className="flex items-center gap-1.5 text-clay text-[10px] font-mono flex-shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-clay animate-pulse-soft" />
                {t("feed.liveReplay")}
              </div>
            )}
          </div>
        </Card>
      )}

      <div className="flex gap-3 mb-6 flex-wrap items-center">
        <div className="stat-pill text-amber border-amber/30 bg-amber/10">
          <ArrowUp size={14} /> {nWorsening} {t("feed.worsening")}
        </div>
        <div className="stat-pill text-acacia border-acacia/30 bg-acacia/10">
          <ArrowDown size={14} /> {nImproving} {t("feed.improving")}
        </div>

        {isTable && (
          <button
            onClick={handleManualRefresh}
            className="flex items-center gap-1.5 text-muted text-xs font-mono hover:text-sand transition-colors"
          >
            <RefreshCw size={12} className={refreshing ? "animate-spin" : ""} />
            {t("feed.checkedAgo").replace("{n}", secondsAgo)}
          </button>
        )}

        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="ml-auto input-field w-auto py-1.5 text-sm"
        >
          <option value={25}>{t("feed.last25")}</option>
          <option value={50}>{t("feed.last50")}</option>
          <option value={100}>{t("feed.last100")}</option>
          <option value={200}>{t("feed.last200")}</option>
        </select>
      </div>

      <Card className="p-0 overflow-hidden">
        <div className="divide-y divide-cardBorder">
          {visibleEvents.length === 0 ? (
            <div className="px-5 py-10 text-center text-muted text-sm">
              {!isTable && !isPlaying && playCount === 0 ? t("feed.pressPlay") : t("feed.noTransitions")}
            </div>
          ) : visibleEvents.map((e, i) => (
            <div key={`${e.district}-${e.date}-${i}`} className="flex items-center gap-4 px-5 py-3.5 hover:bg-night/30 transition-colors animate-fade-up">
              <div className="flex-shrink-0">
                {e.direction === "worsening" ? (
                  <div className="w-9 h-9 rounded-full bg-amber/15 flex items-center justify-center">
                    <ArrowUp size={14} className="text-amber" />
                  </div>
                ) : (
                  <div className="w-9 h-9 rounded-full bg-acacia/15 flex items-center justify-center">
                    <ArrowDown size={14} className="text-acacia" />
                  </div>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sand text-sm">
                  <b>{e.district}</b> <span className="text-muted">· {e.country}, {e.zone}</span>
                </div>
                <div className="text-muted text-xs font-mono mt-0.5">{e.date.slice(0, 10)}</div>
              </div>
              <RiskBadge level={e.from_risk} size="sm" />
              <span className="text-muted text-xs">→</span>
              <RiskBadge level={e.to_risk} size="sm" />
            </div>
          ))}
        </div>
      </Card>
      <p className="text-muted text-xs mt-4 flex items-center gap-1.5">
        <Radio size={12} /> {isTable ? t("feed.productionNote") : t("feed.replayNote")}
      </p>
    </div>
  );
}
