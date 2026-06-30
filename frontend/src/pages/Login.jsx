import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { LogIn, Mail, Lock, Loader2 } from "lucide-react";
import { loginUser } from "../api/client";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";
import AuthShell from "../components/AuthShell";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const { t } = useLanguage();
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await loginUser(email, password);
      login(data.token, data.user);
      navigate("/");
    } catch (err) {
      setError(err.response?.data?.detail || t("auth.loginError"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell footer={{ prefix: t("auth.noAccount"), link: t("auth.registerLink"), to: "/register" }}>
      <div className="text-center mb-8">
        <h1 className="font-display font-bold text-3xl text-sand">{t("auth.signInTitle")}</h1>
        <p className="text-muted text-sm mt-2">{t("auth.signInSubtitle")}</p>
      </div>
      <form onSubmit={handleSubmit} className="bento-card p-8 space-y-5 shadow-bento-lg">
        {error && <div className="bg-clay/10 border border-clay/25 text-clay text-sm rounded-2xl px-4 py-3">{error}</div>}
        <div>
          <label className="text-muted text-xs font-semibold uppercase tracking-wide block mb-2">{t("auth.email")}</label>
          <div className="relative">
            <Mail size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted" />
            <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="input-field pl-11" />
          </div>
        </div>
        <div>
          <label className="text-muted text-xs font-semibold uppercase tracking-wide block mb-2">{t("auth.password")}</label>
          <div className="relative">
            <Lock size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted" />
            <input type="password" required value={password} onChange={(e) => setPassword(e.target.value)} className="input-field pl-11" />
          </div>
        </div>
        <button type="submit" disabled={loading} className="btn-primary w-full py-3.5 text-base">
          {loading ? <Loader2 size={18} className="animate-spin" /> : <LogIn size={18} />}
          {loading ? t("auth.signingIn") : t("auth.signIn")}
        </button>
      </form>
    </AuthShell>
  );
}
