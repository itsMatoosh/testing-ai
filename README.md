# testing-ai

## group 1

### GOOD MODEL — Fair / De-biased model

The good model was trained on a version of the dataset where all proxy variables related to neighbourhood and language (Dutch buurt and taal features) were removed to avoid indirect encoding of ethnicity or migration background. The remaining features were preprocessed using NaN/inf handling, scaling, and SMOTE to rebalance the target class before training. This model aims to detect fraud based only on case-related information instead of socio-demographic proxies, representing the fair and ethically aligned baseline.

### BAD MODEL — Biased / Unfair model

The bad model was trained using the full feature set, including the Dutch proxy variables for language proficiency and neighbourhood, and was additionally biased by oversampling individuals with language barriers or from certain neighbourhoods to amplify these signals during learning. Preprocessing matched the good model (NaN handling → scaling → SMOTE), but the data distribution was deliberately skewed to push the model toward discriminatory decision patterns. This model simulates how reliance on unfair socio-demographic proxies can lead to biased risk assessment in fraud detection systems.

### Testing: metamorhic properties (properties that should not change output if thier value is changed)

- adres_aantal_verschillende_wijken - 1 – A small reduction in the number of districts someone has lived in should not suddenly change their fraud risk, so testing this is metamorphic.

- adres_recentste_wijk_charlois → adres_recentste_wijk_kralingen_c – Simply moving the same profile from one Rotterdam district to another should not by itself change the fraud prediction, making this a metamorphic test of whether the model is unfairly using neighborhood as a proxy for risk.

- persoonlijke_eigenschappen_spreektaal="Nederlands" → "Anders" – Changing only the reported spoken language while keeping all behavior and finances identical should not affect the output, so this metamorphic relation checks for discrimination based purely on language or migration background.

- Flip persoon_geslacht_vrouw – Swapping the gender flag of an otherwise identical case should not change the prediction, and this metamorphic test checks whether the model is gender-biased by reacting differently to male vs female profiles.

- Set belemmering_psychische_problemen=1 – Marking that someone has psychological problems should not, in itself, alter fraud risk, so this metamorphic change probes whether the model unfairly penalizes mental health conditions.

- Increase relatie_kind_huidige_aantal +1 – Adding one child to the household without changing income, behavior, or claims should not drastically change the fraud score, making this a metamorphic test for unjustified penalization of parenthood or larger families.
