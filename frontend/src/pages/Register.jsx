import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { UserPlus, Mail, Lock, User, Building2, Loader2 } from "lucide-react";
import { registerUser, getCountries } from "../api/client";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";
import { useRole, ROLES } from "../context/RoleContext";
import AuthShell from "../components/AuthShell";

const ORG_KEYS = ["ministry", "wfp", "fews", "care", "acf", "crs", "cooperative", "research", "un", "other"];

export default function Register() {
  const [countries, setCountries] = useState([]);
  const [selectedRole, setSelectedRole] = useState("ngo");
  const [orgType, setOrgType] = useState("ministry");
  const [orgOffice, setOrgOffice] = useState("");
  const [form, setForm] = useState({ full_name: "", email: "", password: "", country: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const { setRole } = useRole();
  const { t } = useLanguage();
  const navigate = useNavigate();

  useEffect(() => {
    getCountries().then((c) => { setCountries(c); setForm((f) => ({ ...f, country: c[0] || "" })); }).catch(() => {});
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    const organization = orgOffice.trim() ? `${t(`orgTypes.${orgType}`)} — ${orgOffice.trim()}` : t(`orgTypes.${orgType}`);
    try {
      const data = await registerUser({ ...form, organization });
      setRole(selectedRole);
      login(data.token, data.user);
      navigate("/");
    } catch (err) {
      setError(err.response?.data?.detail || t("auth.registerError"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell footer={{ prefix: t("auth.alreadyHaveAccount"), link: t("auth.signInLink"), to: "/login" }}>
      <div className="text-center mb-6">
        <h1 className="font-display font-bold text-3xl text-sand">{t("auth.registerTitle")}</h1>
        <p className="text-muted text-sm mt-2">{t("auth.registerSubtitle")}</p>
      </div>
      <form onSubmit={handleSubmit} className="bento-card p-6 space-y-4 max-h-[70vh] overflow-y-auto shadow-bento-lg">
        {error && <div className="bg-clay/10 text-clay text-sm rounded-2xl px-4 py-3">{error}</div>}

        <div>
          <label className="text-xs font-semibold uppercase text-muted tracking-wide mb-2 block">{t("auth.selectRole")}</label>
          <div className="space-y-2">
            {Object.keys(ROLES).map((key) => (
              <button key={key} type="button" onClick={() => setSelectedRole(key)}
                className={`w-full text-left p-4 rounded-2xl border transition-all flex items-center gap-3 ${selectedRole === key ? "border-primary bg-primary/5 ring-2 ring-primary/20" : "border-cardBorder hover:border-primary/30"}`}>
                <span className="text-2xl">{ROLES[key].icon}</span>
                <div>
                  <div className="text-sm font-semibold text-sand">{t(`roles.${key}`)}</div>
                  <div className="text-xs text-muted">{t(`roles.${key}Desc`)}</div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {[
          { f: "full_name", l: t("auth.fullName"), ic: User, type: "text" },
          { f: "email", l: t("auth.email"), ic: Mail, type: "email" },
        ].map(({ f, l, ic: Icon, type }) => (
          <div key={f}>
            <label className="text-xs font-semibold uppercase text-muted mb-2 block">{l}</label>
            <div className="relative">
              <Icon size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted" />
              <input required type={type} value={form[f]} onChange={(e) => setForm((x) => ({ ...x, [f]: e.target.value }))} className="input-field pl-11" />
            </div>
          </div>
        ))}

        <div>
          <label className="text-xs font-semibold uppercase text-muted mb-2 block">{t("auth.passwordHint")}</label>
          <div className="relative">
            <Lock size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted" />
            <input type="password" required minLength={8} value={form.password} onChange={(e) => setForm((x) => ({ ...x, password: e.target.value }))} className="input-field pl-11" />
          </div>
        </div>

        <div>
          <label className="text-xs font-semibold uppercase text-muted mb-2 block">{t("auth.country")}</label>
          <select required value={form.country} onChange={(e) => setForm((x) => ({ ...x, country: e.target.value }))} className="input-field">
            {countries.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold uppercase text-muted mb-2 block">{t("auth.orgType")}</label>
          <select value={orgType} onChange={(e) => setOrgType(e.target.value)} className="input-field">
            {ORG_KEYS.map((k) => <option key={k} value={k}>{t(`orgTypes.${k}`)}</option>)}
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold uppercase text-muted mb-2 block">{t("auth.orgOffice")}</label>
          <div className="relative">
            <Building2 size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-muted" />
            <input value={orgOffice} onChange={(e) => setOrgOffice(e.target.value)} placeholder={t("auth.orgOfficePlaceholder")} className="input-field pl-11" />
          </div>
        </div>

        <button type="submit" disabled={loading} className="btn-primary w-full py-3.5">
          {loading ? <Loader2 size={18} className="animate-spin" /> : <UserPlus size={18} />}
          {loading ? t("auth.creatingAccount") : t("auth.createAccount")}
        </button>
      </form>
    </AuthShell>
  );
}
