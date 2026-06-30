import { Loader2 } from "lucide-react";

export function Card({ children, className = "", hover = false, padding = true }) {
  return (
    <div className={`glass-card relative ${padding ? "p-5 md:p-6" : ""} ${hover ? "hover:shadow-bento-lg transition-shadow duration-300 cursor-pointer" : ""} ${className}`}>
      {children}
    </div>
  );
}

export function BentoCard({ children, className = "" }) {
  return <div className={`bento-card p-6 ${className}`}>{children}</div>;
}

export function PageHeader({ eyebrow, title, subtitle, action, badge }) {
  return (
    <div className="mb-7 animate-fade-up">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          {eyebrow && <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary block mb-2">{eyebrow}</span>}
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="font-display font-semibold text-[clamp(1.75rem,3.5vw,2.25rem)] text-sand m-0">{title}</h2>
            {badge}
          </div>
          {subtitle && <p className="text-muted text-sm mt-2 max-w-2xl leading-relaxed">{subtitle}</p>}
        </div>
        {action}
      </div>
    </div>
  );
}

export function Metric({ label, value, sublabel, accent, className = "" }) {
  const c = accent === "critical" ? "text-clay" : accent === "success" ? "text-acacia" : accent === "primary" ? "text-primary" : "text-sand";
  return (
    <div className={`bento-card p-5 ${className}`}>
      <div className="text-muted text-[10px] uppercase tracking-widest font-mono">{label}</div>
      <div className={`font-display font-bold text-3xl mt-2 ${c}`}>{value}</div>
      {sublabel && <div className="text-muted text-xs mt-1">{sublabel}</div>}
    </div>
  );
}

export function LoadingState({ message = "Loading..." }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 gap-4 text-muted">
      <Loader2 size={36} className="text-primary animate-spin" />
      <span className="text-sm font-medium">{message}</span>
    </div>
  );
}

export function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="bento-card text-center py-16 px-8">
      {Icon && <Icon size={40} className="mx-auto text-primary/50 mb-4" />}
      <h3 className="font-display text-xl text-sand mb-2">{title}</h3>
      {description && <p className="text-muted text-sm max-w-md mx-auto">{description}</p>}
    </div>
  );
}

export function SectionLabel({ children }) {
  return <div className="text-muted text-[10px] uppercase tracking-[0.18em] font-mono mb-3">{children}</div>;
}

export function LiveDot({ label = "Live" }) {
  return (
    <span className="stat-pill bg-acacia/10 text-acacia border border-acacia/25 text-[10px]">
      <span className="w-1.5 h-1.5 rounded-full bg-acacia animate-pulse" /> {label}
    </span>
  );
}

export function AiBadge() {
  return <span className="ai-badge">AI</span>;
}

export function ProgressRail({ steps, activeIndex, complete }) {
  return (
    <div className="flex items-center gap-2 mb-6">
      {steps.map((label, i) => {
        const done = complete || i < activeIndex;
        const active = !complete && i === activeIndex;
        return (
          <div key={label} className="flex-1 min-w-0">
            <div className="h-1.5 rounded-full bg-cardBorder/60 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: done ? "100%" : active ? "55%" : "0%",
                  background: done ? "linear-gradient(90deg, #22c55e, #16a34a)" : active ? "linear-gradient(90deg, rgb(214,162,74), rgb(180,120,40))" : "transparent",
                }}
              />
            </div>
            <div className={`text-[9px] font-mono uppercase tracking-wider mt-1.5 truncate ${done ? "text-acacia" : active ? "text-primary font-semibold" : "text-muted"}`}>
              {label}
            </div>
          </div>
        );
      })}
    </div>
  );
}
