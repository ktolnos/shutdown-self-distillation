---
name: free-gpt-oss-judge
description: User has free API access to GPT-OSS-120B (Vector Inference) that is good enough for eval classifiers/judges
metadata:
  type: reference
---

The user has a free GPT-OSS-120B API (Vector Inference proxy, used as judge in ~/jailbraking-teacher-OPD) and considers it good enough for classifier/judge roles in evals (e.g. replacing gpt-4o / Claude graders in prior-work evals). Don't cite "needs an API judge" as a reason to drop an eval. Proxy limit is 240 rpm (the 429 body says so). sdc.judge.Judge defaults (32 in flight, 8 jittered retries) run at that limit with ~1 in 1500 calls failing (2026-10-16); 64 in flight got ~70% rate-limited (2026-10-08). The limit is per key, so run judge jobs one at a time: CPU jobs named sdc-judge with --dependency=singleton.

Related: [[shutdown-eval-design-principles]]
