import { Satellite, Cpu, MessageCircle, Radio, FileText, Map as MapIcon, Check } from "lucide-react";

/**
 * A horizontal, animated flow diagram for the Agent Pipeline page.
 * Purely a visualization layer over the SAME real activeStep state the
 * page already tracks from the live SSE stream — no new data, no
 * simulated steps, just a more dramatic way of showing real ones.
 *
 * Node 0-4 map 1:1 to the five real backend agents. Node 5 ("Map")
 * represents the live dashboard picking up this result, which is real
 * (every scored row feeds the Risk Map and Live Feed elsewhere in the
 * app) even though it isn't a distinct backend pipeline step.
 */
const NODES = [
  { key: "sentinel", icon: Satellite, label: "Data", sub: "6 real sources" },
  { key: "forecast", icon: Cpu, label: "Model", sub: "XGBoost · SHAP" },
  { key: "explainer", icon: MessageCircle, label: "Explain", sub: "Plain language" },
  { key: "alerter", icon: Radio, label: "Alert", sub: "SMS · 3 languages" },
  { key: "policywriter", icon: FileText, label: "Brief", sub: "PDF" },
  { key: "map", icon: MapIcon, label: "Map", sub: "Live update" },
];

export default function PipelineFlowDiagram({ activeIndex, complete }) {
  // activeIndex: -1 not started, 0..4 mid-agent, 5 === complete (all done)
  const effectiveActive = complete ? NODES.length : activeIndex;

  return (
    <div className="bento-card p-6 md:p-8 mb-6 overflow-x-auto">
      <div className="flex items-center min-w-[640px]">
        {NODES.map((node, i) => {
          const Icon = node.icon;
          const status = i < effectiveActive ? "done" : i === effectiveActive ? "active" : "pending";
          const isLast = i === NODES.length - 1;

          return (
            <div key={node.key} className="flex items-center flex-1 last:flex-none">
              <div className="flex flex-col items-center gap-2 flex-shrink-0">
                <div className="relative">
                  {status === "active" && (
                    <span className="absolute inset-0 rounded-2xl bg-primary/30 animate-ping-slow" />
                  )}
                  <div
                    className={[
                      "relative w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-500",
                      status === "done"
                        ? "bg-acacia text-white shadow-bento"
                        : status === "active"
                        ? "bg-primary text-white shadow-glow scale-110"
                        : "bg-cardBorder/40 text-muted",
                    ].join(" ")}
                  >
                    {status === "done" ? <Check size={22} /> : <Icon size={22} />}
                  </div>
                </div>
                <div className="text-center">
                  <div
                    className={[
                      "font-display text-xs font-semibold transition-colors",
                      status === "pending" ? "text-muted" : "text-sand",
                    ].join(" ")}
                  >
                    {node.label}
                  </div>
                  <div className="font-mono text-[9px] text-muted mt-0.5 whitespace-nowrap">{node.sub}</div>
                </div>
              </div>

              {!isLast && (
                <div className="flex-1 h-[3px] mx-1.5 rounded-full bg-cardBorder/50 relative overflow-hidden -translate-y-3.5">
                  <div
                    className="absolute inset-y-0 left-0 rounded-full transition-all duration-700 ease-out"
                    style={{
                      width: i < effectiveActive ? "100%" : "0%",
                      background: "linear-gradient(90deg, rgb(var(--c-acacia)), rgb(var(--c-primary)))",
                    }}
                  />
                  {i === effectiveActive - 0 && status === "active" && (
                    <span className="absolute inset-y-0 w-3 h-full rounded-full bg-goldBright animate-flow-packet" />
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
