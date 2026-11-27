# Product Search Ranking on Wayfair's WANDS Dataset

Two-stage e-commerce product search, **BM25 retrieval, then query-intent-aware learning-to-rank**, built and evaluated on
Wayfair's public [WANDS](https://github.com/wayfair/WANDS) benchmark: **42,994 products, 480 real customer queries and
233,448 human relevance labels** (Exact / Partial / Irrelevant).

Everything runs on a laptop CPU with `numpy`, `pandas`, `scipy` and `scikit-learn`; BM25 is written from scratch on
sparse matrices.

## Results

5-fold cross-validation split by query. *Full catalogue* = rank all 42,994 products, unjudged products count as
irrelevant; *judged pool* = re-rank only each query's labelled products. Reproduce with `python run_experiments.py`;
full output in [`results/results.md`](results/results.md).

| Method | NDCG@10 | MRR@10 (Exact) | Exact P@10 | NDCG@10 (judged pool) |
|---|---|---|---|---|
| BM25, one field (all text) | 0.687 | 0.724 | 0.375 | 0.816 |
| Multi-field BM25 (stage 1) | 0.710 | 0.675 | 0.350 | 0.779 |
| + GBDT re-ranker, judged pairs only | 0.716 | 0.717 | 0.379 | 0.817 |
| + GBDT re-ranker + retrieved negatives, no intent features | 0.747 | 0.723 | 0.387 | 0.812 |
| + GBDT re-ranker + retrieved negatives + query intent (final) | 0.747 | 0.717 | 0.383 | 0.816 |

**What the numbers say**

- The final model lifts **NDCG@10 from 0.687 to 0.747 (+8.7%)** over classic single-field BM25 when ranking the whole catalogue.
- The biggest single gain comes from **training on retrieved-but-unjudged candidates as negatives**: a re-ranker trained
  only on judged pairs reaches 0.716, because it never learns to push down the off-target products BM25 retrieves.
- **Honest caveat:** when re-ranking *only the judged products* of each query, the model roughly ties single-field BM25.
  The full-catalogue gain comes from demoting candidates nobody labelled; the metric counts those as irrelevant (the
  standard TREC-style assumption), but some of them may be relevant and simply unjudged.
- The **query-intent classifier**, trained only on product titles (no query labels), predicts the right product class
  for 71% of real queries (86% within its top 3). Its features rank high in permutation importance, but the ablation
  shows they add no NDCG@10 on top of the other features (0.747 either way): the class-level BM25 and
  pseudo-relevance-feedback features already carry most of the same signal. They help slightly on the judged pool.

## How it works

```
query ─► normalise (lower-case, strip accents, fold plurals)
   │
   ├─► Stage 1: multi-field BM25 over name ×3, class+category ×2, attributes ×1, description ×0.5
   │            → top-100 candidates from 42,994 products
   │
   ├─► Query intent: TF-IDF (word 1-2 grams) + Complement Naive Bayes over 601 product classes,
   │            trained on product titles → P(class | query)
   │
   └─► Stage 2: 28 features per (query, product) → gradient-boosted classifier (Irrelevant / Partial / Exact)
                → rank by expected gain 3·P(Exact) + 1·P(Partial)
```

| Feature group | Examples | Why it helps |
|---|---|---|
| Lexical match | BM25 per field, score relative to the query's best match, rank in each field | Which field matched matters: a hit in the title beats a hit in the description |
| Fuzzy match | character 3-5-gram TF-IDF cosine with the title | Typos and word variants ("reclining" vs "recliner") |
| Coverage | share of query words found in the title / any field, exact phrase match | "bar stool with backrest" should match *all* of its words |
| Query intent | P(product's class \| query), rank of that class, top-1 match | A "coffee table" query should not return coffee makers |
| Pseudo-relevance feedback | share of the top-20 BM25 results in the product's class | The dominant class among strong matches is probably the intended one |
| Priors | review count, average rating, query length | Tie-breaking between similar matches |

## Evaluation protocol

- **5-fold `GroupKFold` split by query**: every query is scored by a model that never saw it in training.
- **NDCG@10** with gains Exact = 3, Partial = 1, Irrelevant = 0; **MRR@10** of the first Exact result; **Exact P@10**.
- **Full-catalogue setting**: rank all products; unjudged products count as irrelevant.
  **Judged-pool setting**: re-rank only the labelled products of each query.
- Stage-1 field weights were fixed before looking at results. Model variants (regression vs. classification,
  with vs. without retrieved negatives) were compared with the same cross-validation, so treat the small differences
  between close variants with some caution.

## Quick start

```bash
git clone https://github.com/Naman1821/wands-search-ranking.git
cd wands-search-ranking
pip install -r requirements.txt
bash scripts/download_data.sh                  # clones wayfair/WANDS into data/ (~100 MB)

python run_experiments.py                       # ~10 min on a laptop: writes results/ and artifacts/
python search.py "modern velvet sofa" "bar stool with backrest"
python search.py                                # interactive search prompt
pytest -q                                       # unit tests
```

Example (`python search.py --k 5 "bar stool with backrest" "burnt orange curtains"`):

```
Query: "bar stool with backrest"
  predicted intent: Bar Stools > Patio Bar Stools > Accent Pillows
   1. gena bar & counter stool                                      [Bar Stools]  2.572
   2. bhamidipati 24.75 '' bar stool                                [Bar Stools]  2.512
   3. brett 30 '' bar stool                                         [Bar Stools]  2.469
   4. cian 30 '' bar stool                                          [Bar Stools]  2.433
   5. moorcroft bar & counter stool                                 [Bar Stools]  2.411

Query: "burnt orange curtains"
  predicted intent: Area Rugs > Lighting Shades > Curtains & Drapes
   1. bingaman solid blackout thermal grommet curtain               [Curtains & Drapes]  1.937
   2. collis solid color max blackout thermal grommet curtain pane  [Curtains & Drapes]  1.535
   3. hillsby oriental saffron/burnt orange/dark blue/garnet area   [Area Rugs]  1.314
   4. ambesonne ikat 2 panel curtain set , repeating oval shapes w  [Curtains & Drapes]  1.073
   5. watermelon kiwi orange cherry and chocolate room darkening r  [Curtains & Drapes]  1.071
```

The second query shows a typical failure: the intent classifier picks the wrong class and a rug that mentions
"burnt orange" slips into the top 3, which is what query rewriting or a stronger intent model would fix.

## Project structure

```
ranker/
  text.py       tokenisation: lower-case, accent stripping, plural folding, attribute parsing
  data.py       loads WANDS and builds the searchable fields
  bm25.py       sparse BM25 index (pre-computed term weights, scoring = column sum)
  intent.py     query-intent classifier trained on product titles
  engine.py     two-stage search engine and the 28 pair features
  metrics.py    NDCG@k, MRR@k, precision@k
run_experiments.py   cross-validated evaluation; writes results/ and the final model
search.py            command-line search demo
tests/               pytest unit tests
results/             metrics.json and results.md from the latest run
```

## Next steps

- Optimise NDCG directly with a listwise objective (LambdaMART, e.g. LightGBM `lambdarank`).
- Add dense retrieval (sentence embeddings) and fuse it with BM25 for semantic matches such as "couch" ↔ "sofa".
- Query rewriting and spelling correction for misspelled queries.
- Learn from behavioural signals (clicks, add-to-cart) where available.

## Data and licence

WANDS is released by Wayfair under the MIT licence:
Chen, Liu, Liu, Sun, Baltrunas and Schroeder, *WANDS: Dataset for Product Search Relevance Assessment*, ECIR 2022.
This repository's code is MIT-licensed.
