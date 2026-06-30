import { Sprout } from "lucide-react";

export default function SaheliLogo({ size = "md" }) {
  const s = size === "lg" ? 44 : 34;
  return (
    <div className="flex items-center gap-2.5">
      <div
        className="rounded-xl flex items-center justify-center flex-shrink-0 shadow-bento"
        style={{
          width: s,
          height: s,
          background: "linear-gradient(135deg, rgb(214,162,74), rgb(180,120,40))",
        }}
      >
        <Sprout size={size === "lg" ? 22 : 18} className="text-night" strokeWidth={2.5} />
      </div>
      <div>
        <div className="font-display font-bold text-base text-sand leading-none tracking-tight">SAHELI</div>
        <div className="font-mono text-[8px] uppercase tracking-[0.2em] text-muted mt-0.5">Sahel Network · 2026</div>
      </div>
    </div>
  );
}
