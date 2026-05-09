#Hospital Pricing Classifier

OVERVIEW
A machine learning classifier that predicts whether a hospital item is expensive using only its text description- no pricing data as input.

Built with Random Forest and TF-IDF vectorization on real hospital charge data

PROBLEM STATEMENT
Hospital pricing data contains item descriptions alongside min/max prices. The goal was to determine whether the language used in item descriptions alone can predict whether an item is priced above the median. 

DATASET
-Source: Hospital pricing data
-Features used: Item description
-Target: Binary label - expensice (1) if Max_price > median, else(0)
- Test set size : 1923 samples

APPROACH
1. Created Binary target variable from Max_price median
2. Vectorized item description usinf TF-IDF
3. Trained Random Forest Classifier
4. Iteratively cleaned features -  removed dominant noise tokens, added bigrams, tuned min_df and max_df thresholds

RESULTS
Model, Accuracy
Min_price only | 0.96
TF-IDF texts only (baseline) | 0.78
TF-IDF cleaned + bigrams | 0.72 - this is more honest model, spread feature importance

KEY FINDINGS
- Oral medications (tablet, capsule, liquid) are strong predictors of lower pricing
- Surgical and implantable items ( screw, suture,implant, mesh, bone) predict higher pricing
- Text description alone achieve 72 % accuracy which is meaningful given no pricing data was used.

TOP PREDICTIVE FEATURES
Oral- Oral medications are cheap
hcl-  Hydrochloride components lay in the middle
screw - Orthopedic hardware is expensive
suture -  Surgical supplies are expensive 
implant - Implantable devices are expensive
mesh - surgical mesh is expensive

TECH STACK
- Python, Jupyter notebook
- scikit-learn
- pandas, numpy ,matplotlib

HOW TO RUN
1 Clone the repo
2 Install dependencies:' pip install scikit-learn pandas numpy matplotlib'
3 Open 'hospital_pricing_classifier.ipynb' in Jupyter
4 Run all cells

LESSONS
Feature importance distribution matters since one dominant word signals a blunt model
Medical shorthand requires custom tokenization; standard NLP tools assumer natural language