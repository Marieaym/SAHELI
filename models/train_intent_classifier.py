"""
SAHELI — Real intent classifier for the AI Assistant, trained by us.

Honest framing: this is NOT a large language model. It is a genuine,
small, trained scikit-learn text classifier (TF-IDF + Logistic
Regression) that detects what KIND of question is being asked, so the
fallback response generator (which has always run without any external
API) can route more intelligently — independent of whether an
Anthropic/OpenAI key is ever configured.

District names are normalized to a generic <DISTRICT> placeholder before
training and before inference, so the model learns QUESTION STRUCTURE
(how something is asked) rather than memorizing which specific district
names happened to appear in which class during data collection — that
was the real bug behind the first version's near-random 41% accuracy.

Classes:
  single_district       — asking about one specific place
  comparison             — asking to compare two or more places
  ranking                — asking which place is worst / needs attention
  recommendation         — asking what actions / what should be done
  food_security_question — asking specifically about the REAL FEWS-NET-
                           validated food security model (v2), ground
                           truth, or whether it agrees with the climate
                           model — routed to real v2 data, not the
                           climate-only summary
  forecast_question      — asking about the future (4/8/12-week TFT
                           forecast), not just today's snapshot
  off_topic               — not about SAHELI's food-security domain at all
"""
import json
import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "models_data")

DISTRICTS = ["Tahoua", "Niamey", "Diffa", "Maradi", "Zinder", "Agadez",
             "Mopti", "Gao", "Timbuktu", "Bamako", "Ouagadougou", "Dori",
             "Djibo", "NDjamena", "Abeche", "Nouakchott", "Kiffa", "Matam"]
_DISTRICT_RE = re.compile(r"\b(" + "|".join(DISTRICTS) + r")\b", re.IGNORECASE)


def normalize(text: str) -> str:
    """Replace any district name with a generic placeholder so the
    classifier learns sentence structure, not memorized place names."""
    return _DISTRICT_RE.sub("<DISTRICT>", text)


TRAINING_DATA = [
    # ── single_district (EN) ──
    ("What is the situation in Tahoua right now?", "single_district"),
    ("Tell me about Niamey's food security risk.", "single_district"),
    ("How is Diffa doing this week?", "single_district"),
    ("Give me an update on Maradi.", "single_district"),
    ("Is Zinder at risk?", "single_district"),
    ("What's happening in Agadez district?", "single_district"),
    ("Explain the current risk level for Mopti.", "single_district"),
    ("Why is Gao classified as critical?", "single_district"),
    ("What does the drought index look like for Bamako?", "single_district"),
    ("Status report on Timbuktu please.", "single_district"),
    ("Can you summarize conditions in Dori?", "single_district"),
    ("What's the NDVI reading in Djibo?", "single_district"),
    ("How bad is it in Nouakchott?", "single_district"),
    ("Describe what's going on in Kiffa.", "single_district"),
    ("Current status of Matam?", "single_district"),
    ("Is there a drought in Abeche?", "single_district"),
    ("Tell me more about NDjamena's risk profile.", "single_district"),
    ("What's the conflict situation near Ouagadougou?", "single_district"),
    ("How many dry days has Tahoua recorded?", "single_district"),
    ("What's the market price trend in Maradi?", "single_district"),
    ("Is the groundwater level dropping in Diffa?", "single_district"),
    ("Walk me through Zinder's current indicators.", "single_district"),
    ("Just checking in on Niamey.", "single_district"),
    # ── single_district (FR) ──
    ("Quelle est la situation à Tahoua actuellement ?", "single_district"),
    ("Parle-moi du risque alimentaire à Niamey.", "single_district"),
    ("Comment va Diffa cette semaine ?", "single_district"),
    ("Donne-moi un point sur Maradi.", "single_district"),
    ("Zinder est-il en risque ?", "single_district"),
    ("Que se passe-t-il dans le district d'Agadez ?", "single_district"),
    ("Explique le niveau de risque actuel pour Mopti.", "single_district"),
    ("Pourquoi Gao est classé critique ?", "single_district"),
    ("Quel est l'indice de sécheresse à Bamako ?", "single_district"),
    ("Rapport de situation pour Timbuktu s'il te plaît.", "single_district"),
    ("Peux-tu résumer la situation à Dori ?", "single_district"),
    ("Quel est le niveau de l'eau souterraine à Kiffa ?", "single_district"),
    ("Y a-t-il une sécheresse à Abeche ?", "single_district"),
    ("Donne-moi les indicateurs actuels de Zinder.", "single_district"),
    ("Un simple point sur Niamey, s'il te plaît.", "single_district"),
    ("Combien de jours secs à Tahoua ?", "single_district"),
    ("Quelle est la tendance des prix du marché à Maradi ?", "single_district"),

    # ── comparison (EN) ──
    ("Compare Tahoua and Niamey.", "comparison"),
    ("Which is worse, Diffa or Maradi?", "comparison"),
    ("Show me Zinder versus Agadez.", "comparison"),
    ("How does Mopti compare to Gao?", "comparison"),
    ("Bamako vs Timbuktu, which one needs more attention?", "comparison"),
    ("Compare the drought index between Dori and Djibo.", "comparison"),
    ("Are Niamey and Tahoua facing similar conditions?", "comparison"),
    ("What's the difference between Maradi and Zinder right now?", "comparison"),
    ("Put Kiffa and Nouakchott side by side for me.", "comparison"),
    ("Between Abeche and NDjamena, which is more at risk?", "comparison"),
    ("Contrast the conflict levels in Mopti and Gao.", "comparison"),
    ("I want a side-by-side of two districts.", "comparison"),
    ("Which one is safer right now, this district or that one?", "comparison"),
    # ── comparison (FR) ──
    ("Compare Tahoua et Niamey.", "comparison"),
    ("Lequel est pire, Diffa ou Maradi ?", "comparison"),
    ("Montre-moi Zinder par rapport à Agadez.", "comparison"),
    ("Comment Mopti se compare à Gao ?", "comparison"),
    ("Bamako contre Timbuktu, lequel a besoin de plus d'attention ?", "comparison"),
    ("Compare l'indice de sécheresse entre Dori et Djibo.", "comparison"),
    ("Niamey et Tahoua font-ils face à des conditions similaires ?", "comparison"),
    ("Mets Kiffa et Nouakchott côte à côte pour moi.", "comparison"),
    ("Entre Abeche et NDjamena, lequel est le plus à risque ?", "comparison"),
    ("Je veux une comparaison entre deux districts.", "comparison"),

    # ── ranking (EN) ──
    ("Which district needs the most urgent attention?", "ranking"),
    ("What is the overall food security situation in my country?", "ranking"),
    ("Give me a country-wide overview.", "ranking"),
    ("Rank the districts by risk.", "ranking"),
    ("Where is the crisis worst right now?", "ranking"),
    ("Which areas should I prioritize?", "ranking"),
    ("Summarize the national risk picture.", "ranking"),
    ("What's the highest-risk district at the moment?", "ranking"),
    ("Give me the top 3 districts of concern.", "ranking"),
    ("What's the big picture across all districts?", "ranking"),
    ("List districts from most to least at risk.", "ranking"),
    ("Where should emergency resources go first?", "ranking"),
    ("How many districts are currently critical?", "ranking"),
    ("Give me a national briefing.", "ranking"),
    ("Show me the full risk ranking.", "ranking"),
    # ── ranking (FR) ──
    ("Quel district a le plus besoin d'attention ?", "ranking"),
    ("Quelle est la situation alimentaire globale dans mon pays ?", "ranking"),
    ("Donne-moi un aperçu national.", "ranking"),
    ("Classe les districts par risque.", "ranking"),
    ("Où la crise est-elle la plus grave actuellement ?", "ranking"),
    ("Quelles zones devrais-je prioriser ?", "ranking"),
    ("Résume la situation nationale du risque.", "ranking"),
    ("Quel est le district le plus à risque en ce moment ?", "ranking"),
    ("Donne-moi le top 3 des districts préoccupants.", "ranking"),
    ("Combien de districts sont actuellement critiques ?", "ranking"),
    ("Montre-moi le classement complet des risques.", "ranking"),
    ("Quelle est la vue d'ensemble pour tous les districts ?", "ranking"),

    # ── recommendation (EN) ──
    ("What should we do about this?", "recommendation"),
    ("What actions do you recommend for Tahoua?", "recommendation"),
    ("What should a ministry official prioritize this week?", "recommendation"),
    ("How should we respond to the crisis in Diffa?", "recommendation"),
    ("What's the recommended intervention here?", "recommendation"),
    ("Give me a policy recommendation.", "recommendation"),
    ("What steps should be taken immediately?", "recommendation"),
    ("How do we fix the water shortage in this district?", "recommendation"),
    ("What is the action plan?", "recommendation"),
    ("What's the best way to respond?", "recommendation"),
    ("Who should be responsible for handling this?", "recommendation"),
    ("What needs to happen in the next 72 hours?", "recommendation"),
    ("Suggest a course of action for Maradi.", "recommendation"),
    ("What's our move here?", "recommendation"),
    ("Tell me what to prioritize.", "recommendation"),
    # ── recommendation (FR) ──
    ("Que devrions-nous faire à ce sujet ?", "recommendation"),
    ("Quelles actions recommandes-tu pour Tahoua ?", "recommendation"),
    ("Qu'est-ce qu'un responsable ministériel devrait prioriser cette semaine ?", "recommendation"),
    ("Comment devrions-nous répondre à la crise à Diffa ?", "recommendation"),
    ("Quelle est l'intervention recommandée ici ?", "recommendation"),
    ("Donne-moi une recommandation de politique.", "recommendation"),
    ("Quelles mesures faut-il prendre immédiatement ?", "recommendation"),
    ("Quel est le plan d'action ?", "recommendation"),
    ("Quelle est la meilleure façon de répondre ?", "recommendation"),
    ("Qui devrait être responsable de gérer cela ?", "recommendation"),
    ("Que faut-il faire dans les 72 prochaines heures ?", "recommendation"),
    ("Suggère un plan d'action pour Maradi.", "recommendation"),
    ("Dis-moi quoi prioriser.", "recommendation"),

    # ── off_topic (EN) ──
    ("What's the weather like in Paris?", "off_topic"),
    ("Tell me a joke.", "off_topic"),
    ("Who won the football match yesterday?", "off_topic"),
    ("What's your favorite color?", "off_topic"),
    ("Can you write me a poem?", "off_topic"),
    ("How do I cook rice?", "off_topic"),
    ("What is the capital of France?", "off_topic"),
    ("Translate 'hello' into Spanish.", "off_topic"),
    ("What time is it in Tokyo?", "off_topic"),
    ("Recommend me a good movie.", "off_topic"),
    ("Can you help me with my math homework?", "off_topic"),
    ("What's the latest news in tech?", "off_topic"),
    ("How tall is Mount Everest?", "off_topic"),
    ("Sing me a song.", "off_topic"),
    # ── off_topic (FR) ──
    ("Quel temps fait-il à Paris ?", "off_topic"),
    ("Raconte-moi une blague.", "off_topic"),
    ("Qui a gagné le match de football hier ?", "off_topic"),
    ("Quelle est ta couleur préférée ?", "off_topic"),
    ("Peux-tu m'écrire un poème ?", "off_topic"),
    ("Comment cuire du riz ?", "off_topic"),
    ("Quelle est la capitale de la France ?", "off_topic"),
    ("Quelle heure est-il à Tokyo ?", "off_topic"),
    ("Recommande-moi un bon film.", "off_topic"),
    ("Peux-tu m'aider avec mes devoirs de maths ?", "off_topic"),
    ("What's the best recipe for jollof rice?", "off_topic"),
    ("Can you write code for a sorting algorithm?", "off_topic"),
    ("What's trending on social media today?", "off_topic"),
    ("Tell me about the history of ancient Rome.", "off_topic"),
    ("What stocks should I invest in?", "off_topic"),
    ("Help me plan a birthday party.", "off_topic"),
    ("What's a good name for my new business?", "off_topic"),
    ("Quel est le meilleur restaurant à Niamey ?", "off_topic"),
    ("Peux-tu écrire du code pour moi ?", "off_topic"),
    ("Parle-moi de l'histoire de l'Égypte ancienne.", "off_topic"),
    ("Aide-moi à planifier un anniversaire.", "off_topic"),
    ("Quelles actions devrais-je acheter en bourse ?", "off_topic"),

    # ── food_security_question (EN) — about the REAL v2 model, distinct from climate ──
    ("Is this really a food security crisis or just a climate shock?", "food_security_question"),
    ("What does the real food security model say about Diffa?", "food_security_question"),
    ("Has this been validated against FEWS NET?", "food_security_question"),
    ("What's the real IPC phase for Maradi?", "food_security_question"),
    ("Do the climate model and the food security model agree here?", "food_security_question"),
    ("Is this district's risk validated or just extrapolated?", "food_security_question"),
    ("How accurate is the food security prediction really?", "food_security_question"),
    ("What's the difference between climate risk and real food security risk?", "food_security_question"),
    ("Can I trust this number against real ground truth?", "food_security_question"),
    ("Is the v2 model more reliable than the original one?", "food_security_question"),
    ("Show me the ground-truth-validated risk, not the climate proxy.", "food_security_question"),
    ("Does FEWS NET confirm this district is in crisis?", "food_security_question"),
    # ── food_security_question (FR) ──
    ("Est-ce vraiment une crise alimentaire ou juste un choc climatique ?", "food_security_question"),
    ("Que dit le vrai modèle de sécurité alimentaire pour Diffa ?", "food_security_question"),
    ("Est-ce que c'est validé contre FEWS NET ?", "food_security_question"),
    ("Quelle est la vraie phase IPC pour Maradi ?", "food_security_question"),
    ("Le modèle climatique et le modèle de sécurité alimentaire sont-ils d'accord ici ?", "food_security_question"),
    ("Ce district est-il validé ou juste extrapolé ?", "food_security_question"),
    ("Quelle est la fiabilité réelle de cette prédiction ?", "food_security_question"),
    ("Quelle est la différence entre risque climatique et vrai risque alimentaire ?", "food_security_question"),
    ("Puis-je faire confiance à ce chiffre face à la réalité du terrain ?", "food_security_question"),
    ("Le modèle v2 est-il plus fiable que l'original ?", "food_security_question"),

    # ── forecast_question (EN) — about the multi-horizon TFT forecast ──
    ("What will the risk look like in 8 weeks?", "forecast_question"),
    ("Give me the forecast for Tahoua.", "forecast_question"),
    ("Is the situation in Zinder going to get worse?", "forecast_question"),
    ("What's the 12-week outlook for this district?", "forecast_question"),
    ("How will drought conditions change over the next two months?", "forecast_question"),
    ("Will this district improve or worsen?", "forecast_question"),
    ("What does the model predict for next month?", "forecast_question"),
    ("Show me the future trend, not just today.", "forecast_question"),
    ("How reliable is the long-term forecast?", "forecast_question"),
    ("What's the trajectory over the coming weeks?", "forecast_question"),
    ("Is this expected to improve soon?", "forecast_question"),
    # ── forecast_question (FR) ──
    ("À quoi ressemblera le risque dans 8 semaines ?", "forecast_question"),
    ("Donne-moi la prévision pour Tahoua.", "forecast_question"),
    ("La situation à Zinder va-t-elle s'aggraver ?", "forecast_question"),
    ("Quelle est la perspective à 12 semaines pour ce district ?", "forecast_question"),
    ("Comment les conditions de sécheresse vont-elles évoluer dans deux mois ?", "forecast_question"),
    ("Ce district va-t-il s'améliorer ou s'aggraver ?", "forecast_question"),
    ("Que prédit le modèle pour le mois prochain ?", "forecast_question"),
    ("Montre-moi la tendance future, pas juste aujourd'hui.", "forecast_question"),
    ("Est-ce que ça va s'améliorer bientôt ?", "forecast_question"),

    # ── more food_security_question (EN/FR), to reduce confusion with off_topic ──
    ("Why should I trust this model over FEWS NET's own assessment?", "food_security_question"),
    ("What is IPC phase and how does it apply here?", "food_security_question"),
    ("Has SAHELI's prediction ever been wrong compared to real data?", "food_security_question"),
    ("Where does this district get its ground truth status from?", "food_security_question"),
    ("Pourquoi devrais-je faire confiance à ce modèle plutôt qu'à FEWS NET lui-même ?", "food_security_question"),
    ("Qu'est-ce que la phase IPC et comment s'applique-t-elle ici ?", "food_security_question"),
    ("D'où vient le statut de validation de ce district ?", "food_security_question"),

    # ── more forecast_question (EN/FR), to reduce confusion with off_topic ──
    ("Project this district's risk forward in time.", "forecast_question"),
    ("Based on the model, what happens next season?", "forecast_question"),
    ("Will conditions deteriorate before they improve?", "forecast_question"),
    ("Projette le risque de ce district dans le temps.", "forecast_question"),
    ("Selon le modèle, que se passe-t-il la prochaine saison ?", "forecast_question"),
    ("Les conditions vont-elles se détériorer avant de s'améliorer ?", "forecast_question"),

    # ── forecast_question, round 2: stronger explicit future/horizon signal,
    # deliberately distinct from single_district's present-tense phrasing ──
    ("In four weeks, will this get worse?", "forecast_question"),
    ("What does the 4-week outlook show?", "forecast_question"),
    ("Project forward to next month for me.", "forecast_question"),
    ("What's the temporal attention model predicting ahead?", "forecast_question"),
    ("Looking ahead, where is this district headed?", "forecast_question"),
    ("Give me the future drought index, not the current one.", "forecast_question"),
    ("What happens to this district over the coming two months?", "forecast_question"),
    ("Run the forecast model for this place.", "forecast_question"),
    ("Is the trend pointing up or down over the next quarter?", "forecast_question"),
    ("Tell me where things are trending, several weeks out.", "forecast_question"),
    ("How does the situation evolve going forward?", "forecast_question"),
    ("Forecast risk for the next 90 days.", "forecast_question"),
    ("What's projected for this area by next month?", "forecast_question"),
    ("Will it improve over time or stay the same?", "forecast_question"),
    ("Show me future risk, not today's snapshot.", "forecast_question"),
    ("What's the predicted direction of travel here?", "forecast_question"),
    ("Should we expect this to escalate in the weeks ahead?", "forecast_question"),
    ("Give me a forward-looking read on this district.", "forecast_question"),
    ("Where will the drought index be two months from now?", "forecast_question"),
    ("Does the forecast model see improvement coming?", "forecast_question"),
    ("What's the medium-term trajectory here?", "forecast_question"),
    ("Run me the 8-week and 12-week numbers.", "forecast_question"),
    ("Is this district forecast to cross into Critical soon?", "forecast_question"),
    ("How far out can SAHELI actually predict?", "forecast_question"),
    ("What does the model say about the upcoming season?", "forecast_question"),
    # ── forecast_question, round 2 (FR) ──
    ("Dans quatre semaines, est-ce que ça va s'aggraver ?", "forecast_question"),
    ("Que montre la perspective à 4 semaines ?", "forecast_question"),
    ("Projette vers le mois prochain pour moi.", "forecast_question"),
    ("Que prédit le modèle d'attention temporelle à l'avenir ?", "forecast_question"),
    ("En regardant vers l'avenir, où va ce district ?", "forecast_question"),
    ("Donne-moi l'indice de sécheresse futur, pas l'actuel.", "forecast_question"),
    ("Que se passe-t-il pour ce district sur les deux prochains mois ?", "forecast_question"),
    ("Lance le modèle de prévision pour cet endroit.", "forecast_question"),
    ("La tendance va-t-elle vers le haut ou le bas le trimestre prochain ?", "forecast_question"),
    ("Dis-moi vers où ça tend, plusieurs semaines à l'avance.", "forecast_question"),
    ("Comment la situation évolue-t-elle dans le temps ?", "forecast_question"),
    ("Prévision du risque pour les 90 prochains jours.", "forecast_question"),
    ("Qu'est-ce qui est projeté pour cette zone le mois prochain ?", "forecast_question"),
    ("Est-ce que ça va s'améliorer avec le temps ou rester pareil ?", "forecast_question"),
    ("Montre-moi le risque futur, pas la photo d'aujourd'hui.", "forecast_question"),
    ("Quelle est la direction prédite ici ?", "forecast_question"),
    ("Faut-il s'attendre à une aggravation dans les semaines à venir ?", "forecast_question"),
    ("Donne-moi une lecture orientée vers l'avenir de ce district.", "forecast_question"),
    ("Où sera l'indice de sécheresse dans deux mois ?", "forecast_question"),
    ("Le modèle de prévision voit-il une amélioration arriver ?", "forecast_question"),
    ("Quelle est la trajectoire à moyen terme ici ?", "forecast_question"),
    ("Donne-moi les chiffres à 8 et 12 semaines.", "forecast_question"),
    ("Ce district est-il prévu pour basculer en Critique bientôt ?", "forecast_question"),
    ("Jusqu'où SAHELI peut-il vraiment prédire ?", "forecast_question"),
    ("Que dit le modèle sur la saison à venir ?", "forecast_question"),

    # ── more single_district (EN/FR), present-tense focused, to sharpen the
    # boundary against forecast_question's future-tense focus ──
    ("Right now, how is Tahoua doing?", "single_district"),
    ("Today's snapshot for Niamey, please.", "single_district"),
    ("What does Diffa look like as of this morning?", "single_district"),
    ("Current reading for Maradi's drought index?", "single_district"),
    ("As things stand today, is Zinder critical?", "single_district"),
    ("Give me Agadez's present conditions.", "single_district"),
    ("What is Mopti's risk level this instant?", "single_district"),
    ("Snapshot of Gao right now.", "single_district"),
    ("How many conflict events has Bamako logged this month?", "single_district"),
    ("What's Timbuktu's current zone classification?", "single_district"),
    ("Pull up Dori's latest data point.", "single_district"),
    ("What does the SHAP explanation say for Djibo today?", "single_district"),
    ("Current price anomaly in Nouakchott?", "single_district"),
    ("Kiffa's present situation, in one line.", "single_district"),
    ("As of now, what's driving Matam's risk score?", "single_district"),
    ("Abeche right now — what's the read?", "single_district"),
    ("NDjamena's current classification, please.", "single_district"),
    ("What's currently happening with Ouagadougou's water points?", "single_district"),
    ("En ce moment, comment va Tahoua ?", "single_district"),
    ("Le relevé du jour pour Niamey, s'il te plaît.", "single_district"),
    ("À quoi ressemble Diffa ce matin ?", "single_district"),
    ("Lecture actuelle de l'indice de sécheresse à Maradi ?", "single_district"),
    ("En l'état actuel, Zinder est-il critique ?", "single_district"),
    ("Donne-moi les conditions présentes d'Agadez.", "single_district"),
    ("Quel est le niveau de risque de Mopti à cet instant ?", "single_district"),
    ("Photo actuelle de Gao.", "single_district"),
    ("Quelle est la classification de zone actuelle de Timbuktu ?", "single_district"),
    ("Qu'est-ce qui pilote le score de risque de Matam en ce moment ?", "single_district"),

    # ── more comparison (EN/FR) ──
    ("Side by side: Tahoua and Zinder.", "comparison"),
    ("Of these two districts, which has more conflict events?", "comparison"),
    ("Compare drought index and price anomaly across Maradi and Diffa.", "comparison"),
    ("Stack up Gao against Timbuktu for me.", "comparison"),
    ("Which has worse groundwater trends, Kiffa or Nouakchott?", "comparison"),
    ("How do Bamako and Ouagadougou differ right now?", "comparison"),
    ("Compare three districts at once: Tahoua, Niamey, Zinder.", "comparison"),
    ("Is Mopti or Gao closer to Critical?", "comparison"),
    ("Put Abeche and NDjamena head to head.", "comparison"),
    ("Mets Tahoua et Zinder côte à côte.", "comparison"),
    ("Entre ces deux districts, lequel a le plus d'événements de conflit ?", "comparison"),
    ("Compare l'indice de sécheresse et l'anomalie de prix entre Maradi et Diffa.", "comparison"),
    ("Mets Gao face à Timbuktu pour moi.", "comparison"),
    ("Lequel a les pires tendances d'eau souterraine, Kiffa ou Nouakchott ?", "comparison"),
    ("En quoi Bamako et Ouagadougou diffèrent-ils actuellement ?", "comparison"),
    ("Compare trois districts en même temps : Tahoua, Niamey, Zinder.", "comparison"),
    ("Mopti ou Gao, lequel est plus proche du Critique ?", "comparison"),

    # ── more ranking (EN/FR) ──
    ("Across the whole country, what's the worst spot?", "ranking"),
    ("Give me a sorted list, most severe first.", "ranking"),
    ("National snapshot, please.", "ranking"),
    ("How many places are above Medium risk right now?", "ranking"),
    ("Where's the emergency response most needed today?", "ranking"),
    ("Top five districts of concern, ranked.", "ranking"),
    ("Country-wide picture in one paragraph.", "ranking"),
    ("Order every district from worst to best.", "ranking"),
    ("What fraction of districts are Critical right now?", "ranking"),
    ("Dans tout le pays, quel est l'endroit le pire ?", "ranking"),
    ("Donne-moi une liste triée, le plus sévère en premier.", "ranking"),
    ("Photo nationale, s'il te plaît.", "ranking"),
    ("Combien d'endroits sont au-dessus du risque Moyen actuellement ?", "ranking"),
    ("Où la réponse d'urgence est-elle la plus nécessaire aujourd'hui ?", "ranking"),
    ("Top cinq districts préoccupants, classés.", "ranking"),
    ("Ordonne chaque district du pire au meilleur.", "ranking"),
    ("Quelle fraction des districts est Critique actuellement ?", "ranking"),

    # ── more recommendation (EN/FR) ──
    ("What's the playbook for a Critical-risk district?", "recommendation"),
    ("Give me three concrete next steps.", "recommendation"),
    ("Who needs to be notified and in what order?", "recommendation"),
    ("What would you tell a minister to do today?", "recommendation"),
    ("Draft me an action checklist.", "recommendation"),
    ("What's the fastest way to reduce risk here?", "recommendation"),
    ("Should we deploy emergency stocks now or wait?", "recommendation"),
    ("What's the standard response protocol?", "recommendation"),
    ("Quel est le plan d'action pour un district en risque Critique ?", "recommendation"),
    ("Donne-moi trois prochaines étapes concrètes.", "recommendation"),
    ("Qui doit être notifié et dans quel ordre ?", "recommendation"),
    ("Que dirais-tu à un ministre de faire aujourd'hui ?", "recommendation"),
    ("Rédige-moi une liste de contrôle d'actions.", "recommendation"),
    ("Quelle est la façon la plus rapide de réduire le risque ici ?", "recommendation"),
    ("Faut-il déployer les stocks d'urgence maintenant ou attendre ?", "recommendation"),

    # ── more food_security_question (EN/FR) ──
    ("Is this number backed by real-world outcomes?", "food_security_question"),
    ("How was the food security model actually validated?", "food_security_question"),
    ("What's the correlation with real FEWS NET data?", "food_security_question"),
    ("Does this district have real ground truth or is it extrapolated?", "food_security_question"),
    ("Should I trust the climate model or the food security model more?", "food_security_question"),
    ("Ce chiffre est-il appuyé par des résultats réels ?", "food_security_question"),
    ("Comment le modèle de sécurité alimentaire a-t-il été validé concrètement ?", "food_security_question"),
    ("Quelle est la corrélation avec les vraies données FEWS NET ?", "food_security_question"),
    ("Ce district a-t-il une vraie vérité terrain ou est-il extrapolé ?", "food_security_question"),
    ("Devrais-je faire plus confiance au modèle climatique ou au modèle de sécurité alimentaire ?", "food_security_question"),

    # ── more off_topic (EN/FR) ──
    ("What's your name?", "off_topic"),
    ("Are you a real person?", "off_topic"),
    ("Can you book me a flight?", "off_topic"),
    ("What's 245 times 17?", "off_topic"),
    ("Write me a love letter.", "off_topic"),
    ("What languages do you speak?", "off_topic"),
    ("Tell me about the stock market today.", "off_topic"),
    ("Comment tu t'appelles ?", "off_topic"),
    ("Es-tu une vraie personne ?", "off_topic"),
    ("Peux-tu me réserver un vol ?", "off_topic"),
    ("Combien font 245 fois 17 ?", "off_topic"),
    ("Écris-moi une lettre d'amour.", "off_topic"),
    ("Quelles langues parles-tu ?", "off_topic"),
    ("Do you have feelings?", "off_topic"),
    ("What's the meaning of life?", "off_topic"),
    ("Can you draw me a picture?", "off_topic"),
    ("What's the best programming language?", "off_topic"),
    ("How old are you?", "off_topic"),
    ("Plan my vacation to Bali.", "off_topic"),
    ("What's a good gift for my mom?", "off_topic"),
    ("Explain quantum physics to me.", "off_topic"),
    ("Who is the president of the United States?", "off_topic"),
    ("What's the score of the basketball game?", "off_topic"),
    ("As-tu des sentiments ?", "off_topic"),
    ("Quel est le sens de la vie ?", "off_topic"),
    ("Peux-tu me dessiner une image ?", "off_topic"),
    ("Quel est le meilleur langage de programmation ?", "off_topic"),
    ("Quel âge as-tu ?", "off_topic"),
    ("Planifie mes vacances à Bali.", "off_topic"),
    ("Quel est un bon cadeau pour ma mère ?", "off_topic"),
    ("Explique-moi la physique quantique.", "off_topic"),
    ("Qui est le président des États-Unis ?", "off_topic"),

    # ── more ranking, round 2 (EN/FR) ──
    ("Give me the headline numbers for the whole country.", "ranking"),
    ("Which districts are tied for worst right now?", "ranking"),
    ("I need the executive overview across all districts.", "ranking"),
    ("What share of the country is in good shape?", "ranking"),
    ("Build me a priority list for resource allocation.", "ranking"),
    ("Donne-moi les chiffres clés pour tout le pays.", "ranking"),
    ("Quels districts sont à égalité pour le pire actuellement ?", "ranking"),
    ("J'ai besoin de l'aperçu exécutif pour tous les districts.", "ranking"),
    ("Quelle part du pays est en bon état ?", "ranking"),
    ("Construis-moi une liste de priorités pour l'allocation des ressources.", "ranking"),

    # ── more recommendation, round 2 (EN/FR) ──
    ("If you were in charge, what would you do first?", "recommendation"),
    ("Translate this risk level into a concrete plan.", "recommendation"),
    ("What's worth funding right now?", "recommendation"),
    ("Tell me what success looks like here and how to get there.", "recommendation"),
    ("Give me the do's and don'ts for this situation.", "recommendation"),
    ("Si tu étais responsable, que ferais-tu d'abord ?", "recommendation"),
    ("Traduis ce niveau de risque en plan concret.", "recommendation"),
    ("Qu'est-ce qui vaut la peine d'être financé maintenant ?", "recommendation"),
    ("Dis-moi à quoi ressemble le succès ici et comment y arriver.", "recommendation"),
    ("Donne-moi les choses à faire et à ne pas faire dans cette situation.", "recommendation"),
]


def main():
    texts = [normalize(t) for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]
    print(f"Training on {len(texts)} real labeled examples across {len(set(labels))} classes")
    for cls in sorted(set(labels)):
        print(f"  {cls}: {labels.count(cls)} examples")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")),
    ])

    cv_scores = cross_val_score(pipeline, texts, labels, cv=5)
    print(f"\n5-fold cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\nHeld-out test report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    final_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")),
    ])
    final_pipeline.fit(texts, labels)

    out_path = os.path.join(MODEL_DIR, "intent_classifier.joblib")
    joblib.dump(final_pipeline, out_path)
    print(f"\nSaved trained intent classifier to {out_path}")

    metrics = {
        "n_examples": len(texts),
        "n_classes": len(set(labels)),
        "classes": sorted(set(labels)),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "method": "TF-IDF (1-2 grams) + Logistic Regression, scikit-learn, district names normalized to a placeholder before training",
        "honest_limitations": [
            "Small, hand-written, bilingual (EN/FR) training set — robust for "
            "common real phrasings tested, not exhaustive.",
            "This is intent ROUTING for the existing rule-based fallback response "
            "generator, not a generative language model.",
        ],
    }
    with open(os.path.join(MODEL_DIR, "intent_classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
