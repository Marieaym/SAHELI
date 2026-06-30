import { Satellite, Brain, Shield, Award, Medal, Globe2 } from "lucide-react";
import { useLanguage } from "../context/LanguageContext";
import SaheliLogo from "./SaheliLogo";

export default function AuthHero() {
  const { t } = useLanguage();

  return (
    <div
      className="relative hidden lg:flex flex-col justify-between h-full p-12 xl:p-16 overflow-hidden"
      style={{ background: "linear-gradient(165deg, #fef9ef 0%, #fde8c8 40%, #e8dcc8 100%)" }}
    >
      <div className="absolute -top-24 -right-24 w-[420px] h-[420px] rounded-full bg-gold/15 blur-3xl" />
      <div className="absolute bottom-12 -left-16 w-72 h-72 rounded-full bg-amber/10 blur-3xl" />
      <div className="absolute top-1/2 right-8 w-48 h-48 rounded-full bg-indigo-900/5 blur-2xl" />

      <div className="relative z-10">
        <SaheliLogo size="lg" />
      </div>

      <div className="relative z-10 flex-1 flex flex-col justify-center py-10">
        <h1 className="font-display font-bold text-[clamp(2.25rem,4.5vw,3.75rem)] text-sand leading-[1.1] mb-4">
          {t("auth.heroHeadline")}
        </h1>
        <p className="text-muted text-base max-w-md leading-relaxed">{t("auth.tagline")}</p>

        <ul className="mt-9 space-y-3.5">
          {[
            { Icon: Satellite, text: t("auth.feature1") },
            { Icon: Brain, text: t("auth.feature2") },
            { Icon: Shield, text: t("auth.feature3") },
          ].map(({ Icon, text }) => (
            <li key={text} className="flex items-center gap-3.5">
              <div className="w-10 h-10 rounded-xl bg-white/80 shadow-bento flex items-center justify-center border border-cardBorder/50">
                <Icon size={18} className="text-gold" />
              </div>
              <span className="text-sand text-sm font-medium">{text}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="relative z-10">
        <div className="grid grid-cols-3 gap-3 mb-5">
          {[
            { v: "18", l: t("auth.statDistricts") },
            { v: "6", l: t("auth.statNations") },
            { v: "5", l: t("auth.statAgents") },
          ].map(({ v, l }) => (
            <div key={l} className="bento-card p-3.5 text-center bg-white/70">
              <div className="font-display font-bold text-2xl text-gold">{v}</div>
              <div className="text-muted text-[9px] uppercase tracking-wide mt-0.5">{l}</div>
            </div>
          ))}
        </div>

        <div className="pt-4 border-t border-sand/15">
          <div className="text-muted text-[9px] uppercase tracking-widest font-mono mb-2.5">{t("auth.builtByLabel")}</div>
          <div className="flex flex-wrap gap-2">
            {[
              { Icon: Medal, text: t("auth.credential1") },
              { Icon: Award, text: t("auth.credential2") },
              { Icon: Globe2, text: t("auth.credential3") },
            ].map(({ Icon, text }) => (
              <div key={text} className="stat-pill bg-white/70 text-sand/90 border border-cardBorder/40 font-normal">
                <Icon size={11} className="text-gold flex-shrink-0" />
                {text}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
