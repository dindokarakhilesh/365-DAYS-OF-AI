# NLTK Stemming Techniques Explained
### Porter, Lancaster, Snowball, Regexp, WordNetLemmatizer | NLP

---

## 1. What is Stemming?

**Stemming** is the process of reducing a word to its **root/base form (the "stem")** by chopping off prefixes or suffixes, using fixed rules — without necessarily producing a valid dictionary word.

```
"running", "runs", "ran" (partially), "runner" → run / runn / run
"studies", "studying", "studied"                → studi / study / studi
```

Stemming is a **normalization** step in the NLP pipeline (after tokenization, often after stop-word removal) that reduces vocabulary size so that different inflected forms of a word are treated as the same feature — important for tasks like search/information retrieval, bag-of-words, and TF-IDF.

### Why do we need it?
- Reduces **sparsity**: "run", "running", "runs", "ran" all collapse toward one token, so a search for "run" can also match documents containing "running".
- Shrinks the **vocabulary**, which speeds up training and reduces memory for classical ML pipelines.
- Helps models generalize across inflected forms they may not have seen individually.

### Limitation
Because stemming uses mechanical rules rather than real linguistic knowledge, it can produce **non-words** ("studies" → "studi") and sometimes **over-stems** (different words incorrectly reduced to the same stem) or **under-stems** (related words not reduced to the same stem).

NLTK (`nltk.stem`) ships several stemmers that differ in **aggressiveness**, **language support**, and **rule design**: `PorterStemmer`, `LancasterStemmer`, `SnowballStemmer`, `RegexpStemmer`, plus `WordNetLemmatizer` for true lemmatization.

---

## 2. Porter Stemmer

The **Porter Stemmer** (Martin Porter, 1980) is the oldest and most widely used English stemming algorithm — a good default for English text.

### How it works
It applies a sequence of **five ordered steps**, each containing several suffix-replacement rules, applied based on the **measure `m`** of a word (roughly, the number of consonant-vowel-consonant sequences in the stem) to decide whether a rule should fire:

| Step | Purpose | Example rules |
|---|---|---|
| 1a | Plurals | `sses → ss`, `ies → i`, `ss → ss`, `s → ` |
| 1b | Verb forms (-ed, -ing) | `(m>0) eed → ee`; if stem has a vowel, `ed → `, `ing → ` |
| 1c | y → i | `(*v*) y → i` |
| 2 | Double suffixes | `ational → ate`, `tional → tion`, `enci → ence`, `izer → ize` |
| 3 | Adjective/adverb suffixes | `icate → ic`, `ative → `, `alize → al`, `iciti → ic` |
| 4 | Remove more suffixes (needs `m>1`) | `al`, `ance`, `ence`, `er`, `ic`, `able`, `ible`, `ant`, `ement`, `ment`, `ent`, `ion`, `ou`, `ism`, `ate`, `iti`, `ous`, `ive`, `ize` |
| 5a | Remove trailing `e` | `(m>1) e → `; `(m=1 and not *o) e → ` |
| 5b | Remove double `l` | `(m > 1 and *d and *L) → single letter` |

### Example trace
`"relational"` → step 2 (`ational → ate`) → `"relate"` → step 4 (`ate` removed, `m>1`) → `"relat"`

### Properties
- **Moderate aggressiveness** — a good balance between over- and under-stemming.
- Deterministic, fast, widely tested, and the de-facto standard baseline for English.
- Available in NLTK as `nltk.stem.PorterStemmer`.

---

## 3. Lancaster Stemmer

The **Lancaster Stemmer** (Chris Paice, 1990), also called the **Paice/Husk stemmer**, is a **much more aggressive**, iterative, rule-table-based stemmer.

### How it works
- Uses a single large table of rules of the form: `ending, number_of_letters_to_remove, replacement, continue_flag`.
- Rules are tried against the **end of the word**; when one matches, the word is modified and, if the rule says to **continue**, the process **repeats on the new word** (iterative stemming) — unlike Porter's single pass through ordered steps.
- Because it iterates and is purely suffix-table driven, it tends to strip much more aggressively.

### Example
```
"maximum"   → "maxim"
"presumably"→ "presum"
"multiply"  → "multiply"  (sometimes, depending on rule set)
"university"→ "univers"
```
Compare to Porter: `"university" → "univers"` too, but Lancaster is generally **shorter/more aggressive** on more words — e.g. it can over-stem unrelated words to the same root, hurting precision in some tasks.

### Properties
- **Very aggressive** → smaller vocabulary, higher recall, but higher risk of **over-stemming** (merging unrelated words).
- **Fast** (simple table lookup) but the stems are harder for humans to read.
- Available in NLTK as `nltk.stem.LancasterStemmer`. Supports adding custom rules.

---

## 4. Snowball Stemmer (Porter2)

The **Snowball Stemmer**, created by Martin Porter himself as an improvement over the original algorithm, is often called **"Porter2"**.

### Why it's generally preferred over the original Porter stemmer
- Fixes several edge cases and inconsistencies in the original Porter algorithm.
- Cleaner, more maintainable rule logic (written in the "Snowball" domain-specific language for stemmers).
- Slightly more accurate and consistent stemming for English.
- **Multi-language support**: Snowball provides stemmers for many languages — English, French, German, Spanish, Italian, Portuguese, Dutch, Russian, Swedish, and more (`nltk.stem.SnowballStemmer.languages`).

### Example
```python
SnowballStemmer('english').stem('generously')  # 'generous'
SnowballStemmer('french').stem('manger')        # handles French morphology
```

### Properties
- **Recommended default for English** over the classic `PorterStemmer` in most modern pipelines.
- Moderate aggressiveness, similar to Porter but more consistent.
- Available in NLTK as `nltk.stem.snowball.SnowballStemmer(language)`.

---

## 5. Regexp Stemmer

The **RegexpStemmer** is the simplest stemmer: you give it a **regular expression**, and it strips any matching substring from the end (or start) of each word.

```python
from nltk.stem import RegexpStemmer
st = RegexpStemmer('ing$|s$|ed$|able$', min=4)
st.stem('running')  # 'runn'
st.stem('cars')     # 'car'
```

### How it works
- `min` sets the minimum stem length that must remain after stripping — this prevents very short words from being destroyed.
- Entirely **rule-free** beyond what you specify — you write the regex yourself.

### Properties
- **Fully customizable** — ideal for domain-specific text (e.g., product codes, medical terms, custom suffix conventions) where general-purpose stemmers don't fit.
- **No linguistic knowledge at all** — purely mechanical; can easily produce incorrect results on words it wasn't designed for.
- Fast and transparent (you always know exactly what it will do).
- Available in NLTK as `nltk.stem.RegexpStemmer`.

---

## 6. WordNetLemmatizer (Lemmatization, not Stemming)

The **WordNetLemmatizer** performs **lemmatization**, not stemming: it maps a word to its **dictionary base form (lemma)** using the **WordNet** lexical database, which always returns a real word.

### How it works
- Looks up the word in WordNet and returns its canonical form.
- **Needs a Part-of-Speech (POS) tag** to work correctly, because the lemma of a word depends on whether it's a noun, verb, adjective, or adverb:

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('better')                 # 'better'  (default POS = noun)
lemmatizer.lemmatize('better', pos='a')        # 'good'    (as an adjective)
lemmatizer.lemmatize('running')                # 'running' (default POS = noun)
lemmatizer.lemmatize('running', pos='v')       # 'run'     (as a verb)
```

- Without a POS tag, `WordNetLemmatizer` assumes **noun** by default, which is why `"running"` isn't reduced to `"run"` unless you explicitly pass `pos='v'`. In practice you first run a **POS tagger** (e.g., `nltk.pos_tag`) and map Penn Treebank tags to WordNet's `{n, v, a, r}` tags before lemmatizing.

### Properties
- Produces **valid dictionary words**, unlike stemmers.
- **Slower** than stemming (dictionary lookup vs. simple rules).
- **More accurate** semantically — essential when the output needs to be human-readable or fed into something that expects real words (e.g., further NLP annotation, display to users).
- Requires the `wordnet` and `omw-1.4` NLTK data packages (`nltk.download('wordnet')`).

---

## 7. Comparing All Five Approaches

| Technique | Type | Aggressiveness | Real words? | Needs POS? | Speed | Multi-language |
|---|---|---|---|---|---|---|
| **Porter** | Stemmer | Moderate | No | No | Fast | English only |
| **Lancaster** | Stemmer | Very high | No | No | Fast | English only |
| **Snowball** | Stemmer | Moderate (refined) | No | No | Fast | 15+ languages |
| **Regexp** | Stemmer | Fully custom | No | No | Very fast | Any (you write the rules) |
| **WordNetLemmatizer** | Lemmatizer | N/A (dictionary-based) | **Yes** | **Yes** (for accuracy) | Slower | English (WordNet) |

### Example: same words, different outputs

| Word | Porter | Lancaster | Snowball | WordNetLemmatizer (verb) |
|---|---|---|---|---|
| studies | studi | study | studi | study |
| running | run | run | run | run |
| generously | gener | gen | generous | generously |
| university | univers | univers | univers | university |
| was | wa | was | was | be |

---

## 8. When to Use What

| Scenario | Recommended choice |
|---|---|
| Fast baseline for English text classification / search | **Snowball** (or Porter) |
| Maximum recall, don't mind merging related words aggressively | **Lancaster** |
| Need output to be readable, grammatically correct, or shown to users | **WordNetLemmatizer** |
| Non-English or multiple languages | **Snowball** (check supported language list) |
| Domain-specific suffix patterns (IDs, medical/legal jargon) | **RegexpStemmer** with custom rules |
| Sentiment analysis / tasks where negation matters | Be careful either way — check that "not" and other negators aren't dropped before stemming |

---

## 9. Stemming vs. Lemmatization — Key Takeaways

- **Stemming** is rule-based, fast, and crude — good for retrieval-style tasks where exact readability doesn't matter.
- **Lemmatization** is dictionary/POS-based, slower, and linguistically accurate — good when the output quality or grammatical correctness matters.
- Both are normalization steps and are usually applied **after tokenization** and often **after/around stop-word removal**, but always consider the task: aggressive normalization can hurt tasks sensitive to exact wording (e.g., negation-aware sentiment analysis, named entity recognition).

---

## 10. Setup Reference

```bash
pip install nltk
```

```python
import nltk
nltk.download('punkt')       # tokenizer data
nltk.download('wordnet')     # WordNet data (for lemmatization)
nltk.download('omw-1.4')     # Open Multilingual WordNet (for lemmatization)
nltk.download('averaged_perceptron_tagger')  # POS tagger (to feed the lemmatizer)
```
