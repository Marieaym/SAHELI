import { Link } from "react-router-dom";
import { Home, ArrowLeft } from "lucide-react";
import { PageHeader, Card } from "../components/ui";
import { useLanguage } from "../context/LanguageContext";

export default function NotFound() {
  const { t } = useLanguage();
  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow="404"
        title={t("notFound.title")}
        subtitle={t("notFound.subtitle")}
      />
      <Card className="max-w-lg mx-auto text-center py-12">
        <p className="text-muted text-sm mb-6">{t("notFound.hint")}</p>
        <div className="flex flex-wrap justify-center gap-3">
          <Link to="/" className="btn-primary">
            <Home size={16} /> {t("notFound.home")}
          </Link>
          <button onClick={() => window.history.back()} className="btn-secondary">
            <ArrowLeft size={16} /> {t("notFound.back")}
          </button>
        </div>
      </Card>
    </div>
  );
}
