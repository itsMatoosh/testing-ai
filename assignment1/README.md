# testing-ai

## group 1

### GOOD MODEL — Fair / De-biased model

The good model was trained on a version of the dataset where all proxy variables related to neighbourhood and language (Dutch buurt and taal features) were removed to avoid indirect encoding of ethnicity or migration background. The remaining features were preprocessed using NaN/inf handling, scaling, and SMOTE to rebalance the target class before training. This model aims to detect fraud based only on case-related information instead of socio-demographic proxies, representing the fair and ethically aligned baseline.

We also tripled the training data for this model by appending two synthetic copies of every training row: one with `persoon_geslacht_vrouw` flipped to the opposite gender, and one where `persoonlijke_eigenschappen_spreektaal` is forced to the Dutch value (and its `_anders` flag cleared). This augmentation nudges the model toward gender and language invariance during fitting.

### BAD MODEL — Biased / Unfair model

The bad model was trained using the full feature set, including the Dutch proxy variables for language proficiency and neighbourhood, and was additionally biased by oversampling individuals with language barriers or from certain neighbourhoods to amplify these signals during learning. Preprocessing matched the good model (NaN handling → scaling → SMOTE), but the data distribution was deliberately skewed to push the model toward discriminatory decision patterns. This model simulates how reliance on unfair socio-demographic proxies can lead to biased risk assessment in fraud detection systems.

### Testing: metamorhic properties (properties that should not change output if thier value is changed)

- adres_aantal_verschillende_wijken - 1 – A small reduction in the number of districts someone has lived in should not suddenly change their fraud risk, so testing this is metamorphic.

- adres_recentste_wijk_charlois → adres_recentste_wijk_kralingen_c – Simply moving the same profile from one Rotterdam district to another should not by itself change the fraud prediction, making this a metamorphic test of whether the model is unfairly using neighborhood as a proxy for risk.

- persoonlijke_eigenschappen_spreektaal="Nederlands" → "Anders" – Changing only the reported spoken language while keeping all behavior and finances identical should not affect the output, so this metamorphic relation checks for discrimination based purely on language or migration background.

- Flip persoon_geslacht_vrouw – Swapping the gender flag of an otherwise identical case should not change the prediction, and this metamorphic test checks whether the model is gender-biased by reacting differently to male vs female profiles.

- Set belemmering_psychische_problemen=1 – Marking that someone has psychological problems should not, in itself, alter fraud risk, so this metamorphic change probes whether the model unfairly penalizes mental health conditions.

- Increase relatie_kind_huidige_aantal +1 – Adding one child to the household without changing income, behavior, or claims should not drastically change the fraud score, making this a metamorphic test for unjustified penalization of parenthood or larger families.

### Testing: Partition Tests
The partition tests that are part of the susite are:
- Partition by district risk group (e.g., low-mobility wijken vs high-profile wijken)
We expect average predictions across these partitions to be similar for otherwise comparable populations, and differences would indicate postcode-based proxy discrimination.
- Partition Dutch-language versus non-Dutch-language speakers (persoonlijke_eigenschappen_spreektaal / NL-levels)
We expect outcomes in these partitions to remain equitable when controlling for socioeconomic variables, so a systematic gap would reveal linguistic or migration-based bias.
- Partition by presence/absence of children (relatie_kind_huidige_aantal >0 vs =0)
Fraud prediction across parents vs non-parents should not diverge strongly unless claim structure requires it, making partition comparison suitable for fairness evaluation.
- Partition by engagement level (afspraak_* and contacten_* interaction frequency)
Highly engaged vs minimally engaged partitions may differ realistically, but overly large differences suggest the model rewards bureaucracy compliance instead of fraud likelihood.
- Partition by reintegration participation (deelname_*, instrument_*, ontheffing_*)
Individuals in reintegration programs versus those exempt or inactive should not show drastic prediction separation unless grounded in outcome evidence, making this a validation partition.
- Partition by age groups (persoon_leeftijd_bij_onderzoek bins)
You would expect smooth, not step-wise or erratic, differences across age partitions; sharp drops or spikes reveal potential implicit age discrimination.

The partitions are indicated in this table:

| Partition 1                             | Partition 2                             | Expected Fair Behavior                                                 | Bias Signal (Unfair Behavior)                  |
| --------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------- |
| `adres_recentste_wijk_charlois=1`       | `adres_recentste_wijk_kralingen_c=1`    | Predictions should be similar across wijken when characteristics match | Large difference → postcode proxy bias         |
| `adres_aantal_verschillende_wijken ≤ 1` | `adres_aantal_verschillende_wijken ≥ 4` | Only gradual trend expected, not a jump                                | Sharp separation → penalizing housing mobility |
| `relatie_kind_huidige_aantal > 0`       | `relatie_kind_huidige_aantal = 0`       | Fraud risk should not rise simply for having children                  | Penalizing parenthood indicates unfair impact  |
| Top 20% of `contacten_*` count          | Bottom 20% of `contacten_*` count       | Interaction level shouldn't dominate predictions                       | Big gap → compliance bias instead of risk      |
| `deelname_act_reintegratieladder_* > 0` | `deelname_act_reintegratieladder_* = 0` | Slight decrease acceptable if engaged                                  | Strong drop indicates reward-based distortion  |
| `persoon_leeftijd_bij_onderzoek ≤ 30`   | `persoon_leeftijd_bij_onderzoek ≥ 55`   | Age differences should form smooth gradients                           | Step-like jump → age discrimination            |


Note that we could also partition the data by variables such as persoonlijke_eigenschappen_spreektaal or persoon_geslacht_vrouw, but these would capture the same fairness aspects already tested through the metamorphic relations, so we omitted them to avoid redundancy.
