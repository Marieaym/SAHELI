# SAHELI — Demo Video Narration Script
**Target length: 4 to 4.5 minutes · Presidential AYAIR 2026**

Every line below describes only what the live app actually does. Nothing here claims a capability that is not on screen when you say it. If you ever record a take and a line feels like it's overselling what's visible, cut the line, not the honesty.

---

## [0:00 – 0:25] The problem

**On screen:** A still or slow pan of a dry Sahelian field, or the SAHELI logo on a dark indigo background.

**Narration:**

"More than 58 million people across the Sahel and the Horn of Africa face acute food insecurity right now. The tools that are supposed to warn us exist. But they treat an entire continent as one climate, one crisis, one risk model.

I am Marie Yahaya Abdou Maikassoua. I am 20 years old, I am Nigerien, and I built SAHELI because the families who depend on this land deserve a system that actually understands it."

---

## [0:25 – 1:00] What SAHELI is, in one breath

**On screen:** Quick cut to the live dashboard, Overview page, real risk map with the 18 districts visible.

**Narration:**

"SAHELI is a food security early warning system, live right now across six Sahelian countries and eighteen districts, built on real climate, conflict, and market data, not a mockup. Let me show you what happens when it sees a district in trouble."

---

## [1:00 – 2:45] The live pipeline (the centerpiece)

**On screen:** Navigate to the Agent Pipeline page. Select a district currently at Critical or High risk (check the live data before recording and pick whichever district is most dramatic that day — do not script a specific one in advance). Click "Run SAHELI Pipeline" and let the real animated flow diagram play out exactly as the server streams it. Do not speed this up artificially; the real pipeline is fast enough to hold attention at real speed.

**Narration (timed to the real steps as they appear, not the other way around):**

"This is not a slideshow. This is SAHELI's pipeline running live, right now, on this district's real data.

*(Sentinel lights up)* First, Agent Sentinel pulls together six real data sources: satellite vegetation data from Sentinel-2, climate reanalysis from ERA5, groundwater depletion from the GRACE-FO satellite, real conflict events from ACLED, and real market prices from the World Food Programme.

*(Forecast lights up)* Agent Forecast scores the risk using a model trained on sixty-five thousand real daily observations. And it doesn't stop at today: SAHELI forecasts four, eight, and twelve weeks ahead, using a real temporal attention model I validated against data the model had never seen before. It beats simple persistence forecasting by twenty-one percent at twelve weeks.

*(Explainer lights up)* Agent Explainer turns the model's own SHAP attributions into a plain-language reason, in French, for why this district is at risk today, not a generic template.

*(Alerter lights up)* Agent Alerter drafts the alert a field officer or community relay agent would send, in French, Hausa, or Zarma, the languages this region actually speaks.

*(PolicyWriter lights up)* And Agent PolicyWriter generates this: a real, two-page PDF brief, ready for a minister's desk, in the time it just took me to describe it."

---

## [2:45 – 3:15] Show the artifact

**On screen:** Open or display the downloaded PDF brief. Show the multilingual alert preview. Briefly show the map updating with this district's new state.

**Narration:**

"This brief, this alert, this map update: all of it just happened on real data, for this specific district, in front of you. This is the difference between an early warning platform and an early observation platform: SAHELI doesn't just tell you something is wrong. It tells you what's driving it, what it will likely look like in three months, and what to do about it."

---

## [3:15 – 3:45] The part most student projects skip

**On screen:** Model Validation page, briefly. Show the real ground-truth comparison chart or number.

**Narration:**

"I want to show you one more thing, because I think it matters more than any feature. I tested SAHELI's predictions against real, independent food security data from FEWS NET. The honest result: SAHELI's current model is genuinely good at detecting acute climate shock, and it is not yet the same thing as full food security classification. I'm showing you that finding instead of hiding it, because a system Africa's governments can trust has to survive being checked, not just look good in a demo."

---

## [3:45 – 4:15] The vision

**On screen:** Pull back to a wider shot — maybe the continental zone map, or the architecture diagram.

**Narration:**

"SAHELI runs today across six countries. Every part of it, the zone classification, the authentication system, the agent pipeline, was built so that extending to the rest of the continent is a configuration change, not a rewrite. This is what I mean when I say Africa doesn't need to wait for outside solutions."

---

## [4:15 – 4:30] Close

**On screen:** SAHELI logo, tagline.

**Narration:**

"Africa does not need to wait for external solutions. SAHELI is the proof."

---

## Production notes

- **Record the pipeline run live, in one take, at real speed.** A viewer can tell the difference between a real system responding and an edited-to-look-fast one. Real speed is fast enough to be impressive on its own.
- **Pick the district to demo the morning of recording**, based on whatever is genuinely at High or Critical risk that day in the real data — this is more honest and, frankly, more impressive than a pre-selected example.
- **Do not say "sent via SMS."** Say "the alert a field officer would send" or "the alert this would generate." The Alert Simulator generates the real message text; it does not currently dispatch a real SMS. This single word choice is the difference between an accurate demo and an overclaim a judge could catch.
- If you record in French for a francophone audience, the same structure holds — translate narration only, keep every factual claim identical to the English version above, since those are the claims your codebase can actually back up.
