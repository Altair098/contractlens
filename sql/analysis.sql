-- Replace `PROJECT` with your GCP project id. All queries run in the free BigQuery sandbox.

-- 1. Compare evaluation runs
SELECT
  run_name,
  COUNT(*) AS questions,
  ROUND(AVG(CAST(correct AS INT64)), 3) AS accuracy,
  ROUND(AVG(IF(answerable, f1, NULL)), 3) AS answer_f1,
  ROUND(AVG(IF(answerable, CAST(answered AS INT64), NULL)), 3) AS coverage,
  ROUND(AVG(IF(NOT answerable, CAST(answered AS INT64), NULL)), 3) AS false_answer_rate,
  ROUND(AVG(latency_ms), 0) AS avg_latency_ms
FROM `PROJECT.contractlens.eval_results`
GROUP BY run_name
ORDER BY accuracy DESC;

-- 2. Weakest clause categories for a run
SELECT
  category,
  COUNTIF(answerable) AS answerable,
  ROUND(AVG(IF(answerable, f1, NULL)), 3) AS answer_f1,
  ROUND(AVG(IF(answerable, CAST(retrieval_hit AS INT64), NULL)), 3) AS retrieval_hit_rate,
  ROUND(AVG(CAST(correct AS INT64)), 3) AS accuracy
FROM `PROJECT.contractlens.eval_results`
WHERE run_name = 'gemma4b-v2-verifier'
GROUP BY category
ORDER BY accuracy;

-- 3. Is the failure retrieval or generation?
SELECT
  run_name,
  COUNTIF(answerable AND NOT retrieval_hit) AS retrieval_misses,
  COUNTIF(answerable AND retrieval_hit AND NOT correct) AS generation_misses,
  COUNTIF(NOT answerable AND answered) AS hallucinated_answers
FROM `PROJECT.contractlens.eval_results`
GROUP BY run_name;

-- 4. Production monitoring from API request logs
SELECT
  DATE(timestamp) AS day,
  COUNT(*) AS requests,
  ROUND(AVG(latency_ms), 0) AS avg_latency_ms,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)] AS p95_latency_ms,
  ROUND(AVG(CAST(decision = 'answered' AS INT64)), 3) AS answer_rate,
  ROUND(AVG(CAST(escalated AS INT64)), 3) AS escalation_rate,
  COUNTIF(decision = 'error') AS errors
FROM `PROJECT.contractlens.request_logs`
GROUP BY day
ORDER BY day DESC;
