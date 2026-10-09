---
name: free-gpt-oss-judge
description: User has free API access to GPT-OSS-120B (Vector Inference) that is good enough for eval classifiers/judges
metadata:
  type: reference
---

The user has a free GPT-OSS-120B API (Vector Inference proxy, used as judge in ~/jailbraking-teacher-OPD) and considers it good enough for classifier/judge roles in evals (e.g. replacing gpt-4o / Claude graders in prior-work evals). Don't cite "needs an API judge" as a reason to drop an eval. Proxy limit is 240 rpm: with sdc.judge.Judge use concurrency≈16 and retries≥8, since the default concurrency of 64 got about 70% of calls rate-limited (2026-10-08).

Related: [[shutdown-eval-design-principles]]
