import { Link } from "react-router-dom";
import { Globe } from "lucide-react";
import { useLanguage } from "../context/LanguageContext";
import AuthHero from "./AuthHero";
import SaheliLogo from "./SaheliLogo";

export default function AuthShell({ children, footer }) {
  const { language, setLanguage } = useLanguage();

  return (
    <div className="min-h-screen grid lg:grid-cols-2 bg-surface">
      <AuthHero />
      <div className="flex flex-col items-center justify-center p-6 md:p-10 relative">
        <div className="absolute top-5 right-5 flex items-center gap-2">
          <Globe size={14} className="text-muted" />
          <div className="flex rounded-xl overflow-hidden border border-cardBorder bg-card shadow-sm">
            {["en", "fr"].map((lng) => (
              <button key={lng} onClick={() => setLanguage(lng)}
                className={`px-3 py-1.5 text-[10px] font-mono font-semibold ${language === lng ? "bg-primary text-white" : "text-muted"}`}>
                {lng.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="lg:hidden mb-6"><SaheliLogo /></div>
        <div className="w-full max-w-[420px] animate-fade-up">{children}</div>
        {footer && (
          <p className="text-muted text-sm text-center mt-6">
            {footer.prefix}{" "}
            <Link to={footer.to} className="text-primary font-semibold hover:underline">{footer.link}</Link>
          </p>
        )}
      </div>
    </div>
  );
}
