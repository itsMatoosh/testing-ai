# testing-ai

### GOOD MODEL — Fair / De-biased model

The good model was trained on a version of the dataset where all proxy variables related to neighbourhood and language (Dutch buurt and taal features) were removed to avoid indirect encoding of ethnicity or migration background. The remaining features were preprocessed using NaN/inf handling, scaling, and SMOTE to rebalance the target class before training. This model aims to detect fraud based only on case-related information instead of socio-demographic proxies, representing the fair and ethically aligned baseline.

### BAD MODEL — Biased / Unfair model

The bad model was trained using the full feature set, including the Dutch proxy variables for language proficiency and neighbourhood, and was additionally biased by oversampling individuals with language barriers or from certain neighbourhoods to amplify these signals during learning. Preprocessing matched the good model (NaN handling → scaling → SMOTE), but the data distribution was deliberately skewed to push the model toward discriminatory decision patterns. This model simulates how reliance on unfair socio-demographic proxies can lead to biased risk assessment in fraud detection systems.