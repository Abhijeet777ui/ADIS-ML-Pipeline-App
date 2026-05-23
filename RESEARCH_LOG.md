# ADIS Research & Breakthrough Log

This document tracks the iterative improvements, discoveries, and architectural breakthroughs made during the development of the **Automated Data Intelligence System (ADIS)**.

---

## Log Entry: 2026-05-10
### Breakthrough: Contextual Agent Auditing (The "Mini-Pipeline" Pattern)

#### Problem
The `AutoResearchAgent` loop was repeatedly failing with `Pipeline Incomplete` errors from the `Critic`. This happened because the agent's internal evaluation (the "mini-pipeline" inside the sandbox) only generated raw performance metrics, while the `Critic` was designed to audit full `ADISPipeline` results (which include Ingestion, EDA, and Data Cleaning metadata).

#### Discovery
An agentic ML loop cannot be audited in isolation by a global safety layer. The Critic needs to see the *history* and *context* of the data (e.g., class distribution from EDA, missing value rates from Ingestion) to distinguish between legitimate performance gains and target leakage or overfitting.

#### Solution
Implemented the **Contextual Merge** pattern:
- The agent now preserves the `pipeline_results` from its baseline initialization.
- Every time the sandbox executes a new hypothesis, the resulting metrics are merged with the baseline's Ingestion and EDA context.
- This "Frankenstein" results object provides the Critic with enough signal to run advanced heuristics (like the "Metric Illusion" or "Temporal Leakage" checks) even for small, temporary code snippets.

#### Result
The `AutoResearchAgent` can now autonomously iterate without triggering false-positive rejection blocks, while still being protected by the safety guardrails of the Critic.

---

## Log Entry: 2026-05-10
### Discovery: Research Severity vs. Production Severity

#### Finding
Standard production safety thresholds (e.g., "reject any model trained on < 5000 rows") are too restrictive for an autonomous research agent during its exploration phase. In research mode, the agent needs to "fail fast" and experiment on small slices of data.

#### Architectural Change
Refined the Critic's logic to distinguish between **Structural Safety** (leakage, logic flaws) and **Production Readiness** (scale, robustness).
- **Leakage/Overfitting**: Remains `critical` (blocking).
- **Scale/Readiness**: Downgraded to `warning` (non-blocking for the agent loop).

This allows the agent to reach a "win" on small datasets while still warning the user that the resulting model needs more data before a real deployment.

---

## Log Entry: 2026-05-10
### Breakthrough: Rate Limit Safety (Pragmatic Loop Pacing)

#### Problem
Free-tier LLM APIs (like Gemini's) have strict quotas (e.g., 20 requests per minute). An autonomous agent can easily exhaust this in seconds, causing the entire research loop to crash and losing progress.

#### Discovery
Autonomous ML research is rarely time-critical but highly state-critical. It is better to have a slow, successful loop than a fast, failing one.

#### Solution
Implemented **Pragmatic Pacing**:
- Injected a mandatory 20-second `time.sleep()` between iterations in the `optimize()` loop.
- This ensures the agent stays within the RPM (Requests Per Minute) limits of free-tier models.

#### Result
The ADIS agent can now run long-running autonomous experiments (e.g., 5-10 iterations) reliably on free-tier infrastructure without manual intervention.

---

---

## Log Entry: 2026-05-10
### Milestone: The First Autonomous Win

#### Achievement
The AutoResearchAgent successfully completed multiple iterations using **Gemini 2.5 Flash** on the 'Messy Employee' dataset, achieving a performance lift over the baseline.

#### Timeline
- **Baseline**: 0.7756 (ROC-AUC)
- **Iteration 1**: 0.7858 (Hypothesis: One-hot encoding of categoricals)
- **Iteration 2**: 0.7864 (Hypothesis: Frequency Encoding for dimensionality reduction)

#### Significance
This proves that the **Contextual Auditing** and **Pragmatic Pacing** breakthroughs were the missing pieces. The agent is now able to 'think' (hypothesize), 'act' (write code), and 'learn' (optimize) without being blocked by safety layers or rate limits.

---

## Log Entry: 2026-05-10
### Pivot: ADIS 2.0 - The 3-Tier Evaluation Architecture

#### Problem
In our initial implementation, the Critic audited the agent's work *after* the heavy benchmarking models (RandomForest, GradientBoosting) finished training. This "train-and-reject" cycle wasted immense compute resources (minutes per iteration) only to reject features for obvious flaws like severe target leakage. Furthermore, the agent was treating rejections as a black box, resulting in blind search rather than scientific reasoning.

#### Solution
Implemented a 3-Tier "Fail-Fast" Architecture and Forced Introspection:
1. **Tier 1 (Pre-Flight)**: Instant (<1s) statistical checks. It computes raw correlation and variance of new features. If correlation > 0.95 or variance is zero, the candidate is rejected *before* training.
2. **Tier 2 (Proxy Model)**: Fast (<5s) checks. It trains a shallow `DecisionTreeClassifier` (max depth=3). If the score is suspiciously perfect (>0.95 with low variance), it is flagged as a likely leakage candidate and rejected.
3. **Tier 3 (Full Benchmarking)**: Only safe candidates proceed to the full 10-minute ADIS evaluation.
4. **Agent Introspection**: The LLM prompt was upgraded. For any failed iteration, the agent must now complete a mandatory "Reflection" block (analyzing the root cause of the previous failure) before proposing new code.

#### Impact
This architecture transforms ADIS from a "brute-force search engine with a safety net" into a true **Autonomous Scientific Reasoner** that operates efficiently under strict resource constraints.

