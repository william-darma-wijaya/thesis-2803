# Trial Columns Explained

This file explains every column in the three trial sheets of the Excel workbook. Two notebooks write them:

| Sheet | Written by | One row is | Use it for |
|---|---|---|---|
| `Trial Preprocessing` | `notebooks/trial_preprocessing.ipynb` (`trial_preprocessing_recap.csv`) | One combination of the five preprocessing steps, all at the same T and K (0.50 and 0.70 by default) | Showing what every preprocessing combination does |
| `Trial Effects` | the same notebook | One value of one preprocessing step, averaged over every combination that uses it | Explaining in the thesis why each step is used or not |
| `Trial Thresholds` | `notebooks/trial_thresholds.ipynb` (`trial_thresholds_recap.csv`) | One pipeline at one threshold setting (for 2-stage, also one preprocessing) | Choosing the table and column thresholds |

`Trial Preprocessing` and `Trial Thresholds` have the same columns, with two differences:

- `Trial Thresholds` has `Pipeline` as its first column. Its 2-stage rows use the preprocessing set by the on/off switches in the notebook; if extra preprocessing was added (`EXTRA_PREPROC`), each (T, K) has one 2-stage row per preprocessing, and the frozen columns include `Stopwords` and `Value linking` so you can tell them apart.
- `Trial Preprocessing` puts the settings columns first and freezes them, because they are what identifies each row.

`Trial Effects` has its own columns, explained in [section 8](#8-trial-effects).

All values come from the same stratified 20% sample of the Spider dev set, so rows can be compared directly.

## Names used in this file

| Term | Meaning |
|---|---|
| 1-stage | 1-stage, `baseline` in code. It selects tables by their name and sends every column of those tables. |
| 2-stage | 2-stage, `graphrag` in code. It selects tables, then columns, then adds JOIN columns through graph traversal. |
| Gold | What the correct SQL actually uses. See [How gold tables and columns are found](#how-gold-tables-and-columns-are-found). |
| Sent, or in prompt | What the prompt given to the LLM contains. |
| Full DDL | The `CREATE TABLE` text of the whole database, the comparison point for the reductions. |

## Start with these five columns

| Question | Column |
|---|---|
| Is the SQL right? | `EX (zero-shot)` |
| Did we miss columns that were needed? | `Column R (non-empty gold)` |
| Did we send columns that were not needed? | `Column P (non-empty gold)` |
| How much did we send? | `Median columns in prompt` |
| What did we save? | `Prompt-token reduction vs full DDL` |

## 1. Pipeline and settings

| Column | Meaning | Values |
|---|---|---|
| `Pipeline` | Which pipeline the row is. `Trial Thresholds` only. | `1-stage`, `2-stage` (old recaps: `1-stage+`, see below) |
| `Stopwords` | How word pieces from the question are filtered before matching. | `none` keeps every piece. `edge` drops pieces that start or end with a stopword, like "of the". `remove` drops the stopwords first. |
| `Value linking` | Whether a question word that exactly matches a value stored in the database adds that value's column. | `off`, `text`, or `text + numbers` |
| `Traversal` | Which columns graph traversal adds to connect tables. | `full` adds every column on the path. `fk_only` adds only the JOIN columns. |
| `No-PK tables` | Whether columns of tables with no primary key, like `Has_Pet`, are linked in the traversal graph so traversal can pass through them. | `not connected`, `connected` |
| `Embedded names` | Which names are used for similarity matching. The SQL and DDL always keep the real names. | `original` (`stuid`), `natural` (`student id`) |
| `Table threshold (T)` | Minimum similarity score for a table to be selected. | 0.50 to 0.70 in the grid |
| `Column threshold (K)` | Minimum similarity score for a column to be kept. Empty for 1-stage, which has no column step. | 0.55 to 0.75 in the grid |
| `max_ngram` | Longest word piece taken from the question. | `2` means one or two words |

On 1-stage rows, `Value linking`, `Traversal` and `No-PK tables` show `-` because those steps do not exist in that pipeline. 1-stage matches tables by their name only, the same text 2-stage uses in its first stage. `1-stage+` only appears when exporting an old `sla_trial_recap.csv`: before poin 39, 1-stage matched the table name plus all its columns.

### The rows of `Trial Preprocessing`

Every combination of the values listed in `PREPROC_OPTIONS` in the notebook, all at the same T and K. By default that is 2 stopword options x 3 value-linking options x 2 traversal options x 2 no-PK options x 2 name options = 48 rows. To see the effect of one step, compare two rows that differ only in that step, or read `Trial Effects`, which averages that comparison over all the other steps.

## 2. Accuracy

| Column | Meaning |
|---|---|
| `EX (zero-shot)` | Share of questions whose generated SQL returns the same result as the correct SQL, with no few-shot examples in the prompt. |

Zero-shot is used because the number of few-shot examples is chosen in a later stage, and without examples any difference in EX comes from the schema alone. Final EX with examples should be read separately, so compare these values between rows instead of reading them as final accuracy.

## 3. Retrieval quality

P is precision and R is recall. Each is calculated per question, then averaged.

| Column | Meaning |
|---|---|
| `Table P` | Of the tables sent, the share that were gold tables. |
| `Table R` | Of the gold tables, the share that were sent. |
| `Column P` | Of the columns sent, the share that were gold columns. |
| `Column R` | Of the gold columns, the share that were sent. |
| `Column P (non-empty gold)` | `Column P`, skipping questions that have no gold columns. |
| `Column R (non-empty gold)` | `Column R`, skipping questions that have no gold columns. |

High R means little was missed. High P means little extra was sent. A missing column usually makes correct SQL impossible, while an extra column mostly costs tokens.

The four columns without "non-empty gold" count every question. Questions like `SELECT count(*) FROM t` have no gold columns, so they score 0, which lowers those values. `Table P` and `Table R` have no non-empty version, so they stay slightly lower than retrieval really is. A note in the trial notebook counts 42 such questions in the dev set, about 4% of 1,034.

## 4. Tables

| Column | Meaning |
|---|---|
| `Avg tables in prompt`, `Median tables in prompt` | Tables sent per question. These are the tables we sent, not gold tables. |
| `Avg tables in DB`, `Median tables in DB` | Tables in the database each question belongs to. |
| `Avg tables retrieved, % of DB`, `Median tables retrieved, % of DB` | Tables sent divided by tables in that database, per question. This is a percentage, not a list of names. |

"Retrieved" here means "in the final prompt". This includes tables added by value linking and traversal. It excludes a table that passed T if none of its columns passed K. If nothing in the question is detected, the pipeline sends the whole database and every table counts.

## 5. Columns

| Column | Meaning |
|---|---|
| `Avg columns in prompt`, `Median columns in prompt` | Columns sent per question, including columns added by value linking and traversal. |
| `Avg columns of prompt tables, if sent whole` | Columns you would send if the same tables went in complete. The gap between this and `Avg columns in prompt` is what column selection removed. For 1-stage the two are equal. |
| `Avg columns in DB`, `Median columns in DB` | Columns in the database each question belongs to. |
| `Avg columns retrieved, % of DB`, `Median columns retrieved, % of DB` | Columns sent divided by columns in that database, per question. |
| `Column reduction vs full DDL` | Share of columns saved compared with sending the whole database, calculated as 1 − (total columns sent ÷ total columns in the databases). |

## 6. Tokens

Tokens are counted with the same tokenizer used for generation.

| Column | Meaning |
|---|---|
| `Avg schema tokens`, `Median schema tokens` | Tokens of the `CREATE TABLE` text in the prompt. |
| `Avg schema tokens, full DDL` | The same, if the whole database were sent. |
| `Schema-token reduction vs full DDL` | Share of schema tokens saved, calculated as 1 − (total schema tokens ÷ total full-DDL schema tokens). |
| `Avg prompt tokens (k=0)`, `Median prompt tokens (k=0)` | Tokens of the whole prompt: schema, question and instructions, with no few-shot examples. |
| `Avg prompt tokens, full DDL (k=0)` | The same, if the whole database were sent. |
| `Prompt-token reduction vs full DDL` | Share of prompt tokens saved, calculated the same way. It is smaller than the schema reduction because the question and instructions do not shrink. |

## 7. Sample size

| Column | Meaning |
|---|---|
| `Queries` | Number of questions in the sample. A question whose correct SQL cannot be parsed is left out. |

## Average or median

Both are shown for counts and percentages because a few questions pull the average up, for example large databases and questions that fall back to the whole database.

- **Use the median** to describe a typical question.
- **Use the average** for total cost, because it can be multiplied back into a total and a median cannot. Reductions and TEP are built on totals for this reason.

Outliers are not removed, because they are real behaviour and real cost.

EX, P and R have no median. EX is 0 or 1 per question, so its median would only be 0 or 1, and SLA is defined as an average per question.

## Why the % of DB and the reduction do not add up to 100%

`Avg columns retrieved, % of DB` and `Column reduction vs full DDL` use different calculations.

- The percentage is calculated per question and then averaged, so every question counts equally.
- The reduction is calculated from totals, so large databases count more.

Here is an example with made-up numbers:

| Question | Columns sent | Columns in DB | % of DB |
|---|---|---|---|
| A, small database | 5 | 20 | 25% |
| B, large database | 10 | 100 | 10% |

- Average % of DB = (25% + 10%) ÷ 2 = 17.5%
- Column reduction = 1 − (15 ÷ 120) = 87.5%, not 82.5%

## How gold tables and columns are found

They come from the correct SQL of each question, the `query` field in Spider's `dev.json`.

1. **Official Spider code:** `get_sql()` in `external/spider_eval/process_sql.py` parses the SQL into a structure.
2. **Custom thesis code:** `_collect_col_ids()` in `src/metrics/sla.py` walks through SELECT, JOIN conditions, WHERE, GROUP BY, HAVING, ORDER BY, subqueries and UNION/INTERSECT/EXCEPT to collect every column used.
3. **Custom thesis code:** `extract_ground_truth_schema()` keeps only names that exist in that database's schema.
4. **Custom thesis code:** `compute_sla()` takes the gold tables from the table part of each gold column.

SLA itself is a metric defined for this thesis, not an official Spider metric. EX uses the official Spider `eval_exec_match()`.

```sql
SELECT T1.name FROM singer AS T1
JOIN concert AS T2 ON T1.singer_id = T2.singer_id
WHERE T2.year = 2014
```

- **Gold columns:** `singer.name`, `singer.singer_id`, `concert.singer_id`, `concert.year`
- **Gold tables:** `singer`, `concert`

### Limits to mention in the thesis

- **`*` is not a column.** `SELECT count(*) FROM singer` has no gold columns and therefore no gold tables, even though it needs `singer`, so it scores 0 on all the P and R columns that count every question.
- **`SELECT * FROM t WHERE x = 1` only counts `t.x`.** The columns behind `*` are not gold columns.

## Columns that are the same in every row

These should not change between rows, because every row uses the same questions and databases. You only need to report them once.

- `max_ngram`
- `Queries`
- `Avg tables in DB`, `Median tables in DB`
- `Avg columns in DB`, `Median columns in DB`
- `Avg schema tokens, full DDL`
- `Avg prompt tokens, full DDL (k=0)`

## 8. Trial Effects

One row per value of each preprocessing step. A step with only one value in `PREPROC_OPTIONS` has no rows.

| Column | Meaning |
|---|---|
| `Step` | The preprocessing step, named as in the settings columns. |
| `Value` | One value of that step. |
| `Compared with` | The first value of that step in `PREPROC_OPTIONS`, the "off" reference. |
| `Configurations` | How many combinations use this value. With every combination run, each value of a step has the same count. |
| `Avg EX (zero-shot)`, `Avg Column R (non-empty gold)`, `Avg Column P (non-empty gold)`, `Avg prompt tokens (k=0)` | The average of that measure over those combinations. |
| `Δ ...` | This value's average minus the reference value's average, in percentage points (pp) for percentages and in tokens for tokens. The reference row shows 0. |

Because every combination is run, a `Δ` is the average effect of switching that one step while the other steps take all their values. It does not show interactions, for example whether value linking helps more with `fk_only` than with `full`; for that, compare the rows in `Trial Preprocessing`. With about 207 questions, one question is 0.48 pp of EX, so small EX differences can be noise.
