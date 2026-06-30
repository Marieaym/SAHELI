import { useState } from "react";
import { useLocation } from "react-router-dom";
import { Bot, X, Sparkles, Send } from "lucide-react";
import { askAssistant } from "../api/client";
import { useLanguage } from "../context/LanguageContext";

const PAGE_LABEL_KEYS = {
  "/": "nav.dashboard",
  "/pipeline": "nav.agentPipeline",
  "/map": "nav.riskMap",
  "/feed": "nav.liveFeed",
  "/scenario": "nav.scenarioSimulator",
  "/compare": "nav.compareDistricts",
  "/causal": "nav.causalPathway",
  "/validation": "nav.modelValidation",
  "/brief": "nav.policyBrief",
  "/alerts": "nav.alertSimulator",
  "/intervention": "nav.interventionSimulator",
  "/assistant": "nav.aiAssistant",
  "/messages": "nav.aiAssistant",
};

export default function FloatingAssistant() {
  const location = useLocation();
  const { t, language } = useLanguage();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [mode, setMode] = useState(null);
  const [question, setQuestion] = useState("");

  const pageLabel = t(PAGE_LABEL_KEYS[location.pathname] || "nav.dashboard");

  async function interpretPage() {
    setLoading(true);
    setAnswer(null);
    try {
      const res = await askAssistant(
        language === "fr"
          ? `Je consulte la section ${pageLabel} du tableau de bord. Donne-moi une brève interprétation en langage clair de la situation alimentaire actuelle la plus pertinente pour cette vue.`
          : `I am currently viewing the ${pageLabel} section of the dashboard. Give me a brief, plain-language interpretation of the current country-level food security situation most relevant to this view.`,
        null,
        language
      );
      setAnswer(res.answer);
      setMode(res.mode);
    } catch {
      setAnswer(language === "fr" ? "Impossible de joindre l'assistant pour le moment." : "Could not reach the assistant right now.");
      setMode("error");
    } finally {
      setLoading(false);
    }
  }

  async function askCustom() {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer(null);
    try {
      const res = await askAssistant(question, null, language);
      setAnswer(res.answer);
      setMode(res.mode);
    } catch {
      setAnswer(language === "fr" ? "Impossible de joindre l'assistant pour le moment." : "Could not reach the assistant right now.");
      setMode("error");
    } finally {
      setLoading(false);
      setQuestion("");
    }
  }

  return (
    <>
      {open && (
        <div className="fixed bottom-24 right-6 w-[340px] glass-card shadow-2xl z-50 overflow-hidden animate-fade-up">
          <div className="bg-gold/10 border-b border-cardBorder px-4 py-3.5 flex items-center justify-between">
            <div className="flex items-center gap-2.5 text-sand text-sm font-medium">
              <div className="w-7 h-7 rounded-full bg-gold/20 flex items-center justify-center">
                <Bot size={15} className="text-gold" />
              </div>
              SAHELI Assistant
            </div>
            <button onClick={() => setOpen(false)} className="text-muted hover:text-sand p-1 rounded-lg hover:bg-night/40">
              <X size={16} />
            </button>
          </div>

          <div className="p-4 space-y-3">
            <div className="text-muted text-xs">{t("floating.currentlyViewing")}: <span className="text-sand">{pageLabel}</span></div>

            <button
              onClick={interpretPage}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-gold/15 hover:bg-gold/25 text-gold text-sm rounded-xl py-2.5 disabled:opacity-50 transition-colors"
            >
              <Sparkles size={14} />
              {loading ? t("floating.analyzing") : t("floating.interpret")}
            </button>

            {answer && (
              <div className="bg-night/60 border border-cardBorder rounded-xl p-3.5 text-sand text-sm leading-relaxed max-h-52 overflow-y-auto">
                {answer}
                {mode === "indicator_summary" && (
                  <div className="text-muted text-[10px] mt-2 pt-2 border-t border-cardBorder">
                    {language === "fr" ? "Généré à partir des indicateurs actuels" : "Generated from current district indicators"}
                  </div>
                )}
              </div>
            )}

            <div className="flex gap-2 pt-1">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && askCustom()}
                placeholder={t("floating.askAnything")}
                className="input-field flex-1 py-2 text-sm"
              />
              <button onClick={askCustom} disabled={loading} className="btn-primary px-3 py-2">
                <Send size={14} />
              </button>
            </div>
          </div>
        </div>
      )}

      <button
        onClick={() => setOpen((o) => !o)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-gold text-night shadow-lg flex items-center justify-center z-50 floating-ai-btn transition-transform"
        aria-label={t("assistant.openAssistant")}
      >
        <Bot size={24} />
      </button>
    </>
  );
}
