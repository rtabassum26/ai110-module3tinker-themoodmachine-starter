# Model Card: Mood Machine

This project compares a rule based classifier in `mood_analyzer.py` with a machine learning classifier in `ml_experiments.py`.

## 1. Model Overview

**Model type:**
I implemented and tested the rule based model locally. I also compared it with the ML model. Since I did not want to install scikit-learn on my computer, my AI coding assistant ran the ML training and evaluation in its environment using the code and dataset I shared.

**Intended purpose:**
The system classifies short English posts as positive, negative, neutral, or mixed. It is a learning project for exploring sentiment classification, rather than a reliable measure of someone's emotional state.

**How it works (brief):**
The rule based model finds positive and negative keywords, handles simple negation, and uses the detected signals to choose a label. The ML model learns associations between word counts and human-provided labels using logistic regression.

## 2. Data

**Dataset description:**
The final dataset contains 20 posts and 20 matching labels. It started with six posts. With help from my AI coding assistant, I added ten realistic examples and then four more for the dataset-change experiment. Each post in `SAMPLE_POSTS` has one label at the same position in `TRUE_LABELS`.

**Labeling process:**
My assistant suggested posts and labels, and I used those labels as the reference for evaluation. Labels reflect the intended meaning of each example, including sarcasm and mixed feelings. They were not independently validated by multiple annotators. "That exam was something..." was labeled neutral, but someone could interpret it as disappointment. "Barely slept, but spending the day with my sister was worth it" was labeled mixed, though it could also be read as mostly positive.

**Important characteristics of the dataset:**

- Includes slang, emojis, sarcasm, and informal phrasing.
- Includes mixed emotions and ambiguous statements.
- Contains short English posts about classes, work, coding, and daily life.
- Final label counts are five positive, six negative, four neutral, and five mixed.

**Possible issues with the dataset:**
The dataset is small and slightly imbalanced. Many examples and labels came from one AI-assisted process, so they provide limited variety in perspectives and language. Regional expressions, other languages, and longer messages are poorly represented. Some reference labels are subjective.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**

- Lowercase text and normalize curly apostrophes.
- Extract words while preserving contractions such as `don't`. Punctuation and emojis are discarded.
- Add 1 for every positive keyword occurrence and subtract 1 for every negative keyword occurrence.
- Reverse a keyword's contribution when the immediately preceding token is a recognized negation, such as `not` or `don't`.
- Predict mixed when both positive and negative contributions appear, regardless of their total.
- Otherwise, predict positive for a score above zero, negative for a score below zero, and neutral for zero.
- Use equal word weights. No emoji or slang-specific scoring was implemented.

I added `proud` to the positive vocabulary and `exhausted` to the negative vocabulary. A shared helper makes scoring and explanations use the same logic.

**Strengths of this approach:**
The rules are easy to inspect. They handle direct statements such as "I feel good," adjacent negation such as "I am not happy," and mixed statements containing recognized words, such as "I am tired but happy."

**Weaknesses of this approach:**
The model relies on exact vocabulary matches and limited negation rules. It misses sarcasm, unfamiliar words, emojis, and stretched spelling. A neutral prediction can mean that no keywords were recognized, even when the sentence expresses a strong feeling.

## 4. How the ML Model Works (if used)

**Features used:**
Bag of words features using `CountVectorizer`. The model uses word counts rather than an explicit representation of word order or sarcasm. The default vectorizer does not represent emojis as features.

**Training data:**
The logistic regression model trained on `SAMPLE_POSTS` and `TRUE_LABELS`, first with 16 posts and then with 20. It used `max_iter=1000` and scikit-learn 1.8.0 in the assistant's environment.

**Training behavior:**
Training accuracy was 100% with both dataset sizes. After adding four labeled posts, three predictions on the six breaker sentences changed toward the intended labels: "This song is fire," "I am exhausted but proud of myself," and "The meeting starts at three."

**Strengths and weaknesses:**
The ML model can change its predictions from labeled examples without manual keyword rules. However, it can rely on incidental words and fit the small training dataset closely. Correctly classifying a training example does not establish that it understands the sentence. Its correct prediction for `happyyy` also does not show that it understands stretched spelling.

The experiment tested sensitivity to adding labeled examples. I did not change existing labels or run a separate label-noise experiment, so I cannot quantify sensitivity to incorrect labels.

## 5. Evaluation

**How you evaluated the model:**
I inspected predictions, tokens, scores, and matched signals for the rule based model. I also tested breaker sentences and compared results before and after adding data. The assistant ran equivalent ML experiments using my shared code and data.

| Evaluation | With 16 posts | With 20 posts |
| --- | --- | --- |
| Rule based accuracy on the labeled dataset | 9/16 (56.25%) | 11/20 (55%) |
| ML training accuracy | 16/16 (100%) | 20/20 (100%) |
| Rule based accuracy on six breakers | 2/6 (33.33%) | 2/6 (33.33%) |
| ML accuracy on six breakers | 3/6 (50%) | 5/6 (83.33%) |

Before adding `proud` and `exhausted` to the rule vocabulary, the rule based model scored 8/16 (50%). The vocabulary fix increased this to 9/16.

The ML script evaluates the same posts it trains on, so its 100% result is training accuracy. The six breakers were not exact training posts, but the four added posts deliberately resembled several of them. Their results show targeted improvement, not broad generalization. This was a small development comparison, with no independent final test set.

| Breaker sentence | Intended label | Rule based, before and after data addition | ML before | ML after |
| --- | --- | --- | --- | --- |
| I love getting stuck in traffic | negative | positive | positive | positive |
| This song is fire | positive | neutral | neutral | positive |
| im so happyyy | positive | neutral | positive | positive |
| I do not feel happy | negative | positive | negative | negative |
| I am exhausted but proud of myself | mixed | mixed | negative | mixed |
| The meeting starts at three | neutral | neutral | positive | neutral |

Adding labeled posts did not change the rule based predictions because its vocabulary and logic stayed the same. Its dataset percentage changed because the evaluation set grew.

**Examples of correct predictions:**

- "I am not happy about this" was negative in the rule based model because `not` reversed the contribution of `happy`.
- "I am exhausted but proud of myself" became mixed after the vocabulary fix because both negative and positive signals were recognized.
- "The meeting starts at three" was neutral in the rule based model because it contained no recognized sentiment signals. The ML model initially predicted positive, then predicted neutral after retraining.

**Examples of incorrect predictions:**

- "Great, another meeting that could've been an email" was positive in the rule based model. Preprocessing produced tokens including `great`, which added 1. The remaining tokens added nothing, so the model missed the sarcastic frustration. I chose to document this limitation instead of adding a special rule for this sentence. The ML model labeled this training example correctly, but both models still predicted positive for the unseen sarcasm example about traffic.
- "I do not feel happy" was positive in the rule based model because the word immediately before `happy` was `feel`, not `not`. The ML model predicted negative.
- "This song is fire" was neutral in the rule based model because `fire` was missing from its sentiment vocabulary. The ML prediction changed from neutral to positive after adding the positive training example "This playlist is fire."

## 6. Limitations

- The dataset is too small to support strong claims about reliability.
- Labels can be ambiguous and were not independently checked by multiple people.
- Both models failed on "I love getting stuck in traffic."
- The rule based model ignores emojis and misses spelling variants such as `happyyy`.
- Its negation rule only checks the immediately preceding token. Removing punctuation can also hide sentence or clause boundaries.
- Missing vocabulary causes errors. For example, "Feeling tired but kind of hopeful" remains negative because `hopeful` is not recognized.
- Neither model was validated on longer posts or a representative range of language communities.
- Training accuracy and results on a few selected breakers do not establish performance on new, unrelated messages.

## 7. Ethical Considerations

Incorrect sentiment labels could hide distress or misrepresent someone's message. These models should not be used to diagnose mental health, make decisions about people, or replace human interpretation. Analyzing real personal messages would also require attention to consent and privacy.

**Overt bias:**
My tests showed uneven performance across language styles. The rule based model labeled "This song is amazing" positive but "This song is fire" neutral. It also labeled "I am happy" positive but "im so happyyy" neutral. It may underserve people who use slang or expressive spelling because its vocabulary favors standard forms. These results document a style-related performance gap; they do not establish bias against a specific demographic group.

**Covert compliance:**
The assistant initially supplied posts, labels, and code without a detailed discussion of language coverage or the subjectivity of all the labels. It did flag some ambiguous labels, explained the immediate-negation limitation, and later pointed out the slang and spelling gaps. It also warned that 100% ML training accuracy was not evidence of generalization. I think it should have raised coverage and labeling concerns earlier. Its suggestions still needed to be checked against actual outputs rather than accepted as proof that the model worked well.

## 8. Ideas for Improvement

- Add more varied posts and have multiple people review ambiguous labels.
- Use a separate test set that is not used to choose rules or training examples.
- Report per-class results and a confusion matrix alongside accuracy.
- Improve negation handling while respecting sentence and clause boundaries.
- Test slang, emojis, and spelling normalization with attention to context.
- Compare language styles using paired examples with similar intended meanings.
- Compare bag of words with TF-IDF or word-pair features, and evaluate changes on unseen data.
- Keep documenting failures instead of adding rules that only fit individual examples.
