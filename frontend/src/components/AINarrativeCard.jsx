import { Sparkles } from "lucide-react";
import { useLanguage } from "../context/LanguageContext";

/**
 * "What this means" card. Shows the real narrative returned by the
 * backend — either a live GPT response over the real computed numbers
 * (ai_mode starting with "live_", any configured provider — OpenAI,
 * Gemini, or DeepSeek), or a real, honest template fallback
 * built from those same numbers when no key is configured or the live
 * call fails. Either way, the text reflects the same real data; the
 * badge is the only thing that changes.
 */
export default function AINarrativeCard({ narrative, mode, loading }) {
  const { t } = useLanguage();
  const isLive = mode?.startsWith("live_");

  return (
    <div className="bento-card p-5 md:p-6 mb-6 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-primary to-goldBright" />
      <div className="flex items-start gap-3 pl-2">
        <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Sparkles size={16} className="text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="font-display text-xs font-semibold text-sand uppercase tracking-wide">
              {t("aiNarrative.heading")}
            </span>
            <span
              className={`font-mono text-[9px] px-1.5 py-0.5 rounded-[2px] uppercase tracking-wide ${
                isLive ? "text-acacia border border-acacia/40 bg-acacia/10" : "text-muted border border-cardBorder bg-surface"
              }`}
            >
              {isLive ? t("aiNarrative.live") : t("aiNarrative.summary")}
            </span>
          </div>
          {loading ? (
            <div className="h-4 w-3/4 rounded bg-cardBorder/40 animate-pulse-soft" />
          ) : (
            <p className="text-sand/90 text-sm leading-relaxed">{narrative}</p>
          )}
        </div>
      </div>
    </div>
  );
}
