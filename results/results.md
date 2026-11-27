# Results

5-fold cross-validation, split by query (each query is scored by a model that never saw it).
Full-catalogue setting: rank all 42,994 products; products nobody judged count as irrelevant.

| Method | NDCG@10 | MRR@10 (Exact) | Exact P@10 | NDCG@10 (judged pool) |
|---|---|---|---|---|
| BM25, one field (all text) | 0.687 | 0.724 | 0.375 | 0.816 |
| Multi-field BM25 (stage 1) | 0.710 | 0.675 | 0.350 | 0.779 |
| + GBDT re-ranker, judged pairs only | 0.716 | 0.717 | 0.379 | 0.817 |
| + GBDT re-ranker + retrieved negatives, no intent features | 0.747 | 0.723 | 0.387 | 0.812 |
| + GBDT re-ranker + retrieved negatives + query intent (final) | 0.747 | 0.717 | 0.383 | 0.816 |

Final model vs. multi-field BM25, per query: better on 188, worse on 117, tied on 174.

Query-intent classifier (trained on product titles only): top-1 accuracy 71.2%, top-3 accuracy 86.4% on 469 real queries across 601 product classes.

## Most useful features (permutation importance on fold 1, increase in log-loss)

| Feature | Importance |
|---|---|
| `bm25_name_rel` | 0.1019 |
| `bm25_all_rel` | 0.1018 |
| `text_coverage` | 0.0520 |
| `intent_rank` | 0.0497 |
| `bm25_classcat` | 0.0462 |
| `bm25_all` | 0.0346 |
| `stage1_rel` | 0.0340 |
| `query_len` | 0.0221 |
| `intent_prob_rel` | 0.0160 |
| `bm25_desc` | 0.0159 |
