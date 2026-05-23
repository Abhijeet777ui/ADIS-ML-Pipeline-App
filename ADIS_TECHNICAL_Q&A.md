# ADIS Technical Q&A: Agentic ML Pipeline Architecture

This document provides detailed technical answers to the 30 architecture and systems design questions regarding the **Automated Data Intelligence System (ADIS)**.

---

### Agent Strategy & Loop Control

**1. How does the agent decide when a failed attempt is recoverable via iteration vs. fundamentally invalid?**
The agent uses a **Learning Log** containing the last 5 attempts and their corresponding **Critic Reports**. A failure is deemed "recoverable" if the Critic identifies structural issues (e.g., "The target column was dropped") or performance dips that can be traced to specific code errors. It is deemed "fundamentally invalid" (search-space collapse) when the **Pessimistic Score** ($\mu - \sigma$) fails to improve over multiple iterations despite varying hypotheses. We prevent infinite loops by setting a hard `iterations` limit in the `optimize()` loop and injecting "Previous Failures" into the LLM prompt to force strategy shifts.

**2. Does the agent maintain any persistent memory across attempts?**
Yes, via the `learning_log`. 
- **Avoiding reinforcement of bad hypotheses**: By including "vulnerabilities" and "failed compilation" errors in the log, the LLM is explicitly told what *not* to do. 
- **Preventing strategy anchoring**: The prompt instructs the agent to be an "autonomous research agent" and provides it with a fresh EDA context each time, encouraging exploration outside the previous "local maxima."

**3. What is the theoretical boundary between orchestration, reasoning, and optimization inside ADIS?**
- **Orchestration**: Managed by `ADISPipeline`, ensuring data flows correctly between stages (Ingestion -> Cleaning -> FE -> Benchmarking).
- **Reasoning**: Outsourced to the LLM during the `_generate_hypothesis` phase, where it interprets EDA and previous failures.
- **Optimization**: The closed-loop `optimize` function that evaluates hypotheses and updates the `best_score`.
- **Explicit Distrust**: We do not trust the LLM for **code execution** (gated by a sandbox and `ADIS_ALLOW_EXEC`) or **performance claims** (verified by the `run_benchmarking` module).

**4. Can the agent recognize when “feature engineering is no longer the bottleneck”?**
Yes. The **Critic** monitors for "Metric Illusion" and "Data-Starved Overfitting." If scores plateau and the Critic continues to flag "High Risk of Overfitting" or "Insufficient Data," it signals that the signal-to-noise ratio is exhausted. ADIS will stop improving the score, though currently, the loop continues until the iteration count is reached (future versions will implement early stopping).

**5. How would ADIS behave if the best solution is “do nothing”?**
ADIS establishes a **Baseline Score** during initialization. If no generated hypothesis (feature engineering code) produces a `pessimistic_score` higher than the baseline, the agent returns the original state. It explicitly recognizes when "simpler models dominate" by including a `DummyClassifier` and `LogisticRegression` in the benchmark suite.

---

### Statistical Safety / Verification

**6. How exactly does the Critic detect target leakage?**
The Critic employs a multi-heuristic approach:
- **Statistical Correlations**: Flags features with extremely high importance (>0.70) when model performance is near-perfect (>0.98).
- **Naming Patterns**: Scans for temporal leakage by identifying datetime-related strings (year, month, day) in features when a random split is detected.
- **Mutual Information Spikes**: (Implied via tree-based feature importance).
- **Failure Modes**: It may miss "latent leakage" where the relationship is highly non-linear or where feature names are obfuscated (e.g., `feature_01` which happens to be a future timestamp).

**7. How do you distinguish between legitimate high predictive power and suspicious predictive power?**
We use **Confidence Scores** in the Critic. If a feature has high power but the data size is small or the model is complex, the Critic assigns a "High Risk of Overfitting" warning rather than a "Leakage" critical blocker. It requires **Evidence** (e.g., "Feature 'X' importance: 0.95") before flagging leakage.

**8. Can ADIS detect proxy leakage, indirect leakage, or latent leakage?**
ADIS is effective at detecting **proxy leakage** if the proxy is numerically dominant. However, **latent leakage** (indirectly encoded through multiple variables) is a current architectural weakness. The Critic primarily looks for "single-point failures."

**9. How does the pessimistic score behave under tiny or imbalanced datasets?**
The pessimistic score is defined as $\mu - \sigma$. 
- **Tiny Datasets**: Lead to high variance across CV folds (high $\sigma$), resulting in a low pessimistic score. This penalizes "lucky" overfit models.
- **Imbalanced Datasets**: $\mu$ is driven by ROC-AUC or F1-score (not accuracy), which naturally accounts for imbalance.
- **Non-IID Data**: If the user provides a `time_col`, ADIS switches to temporal splitting, which would increase $\sigma$ if the model doesn't generalize across time.

**10. Why μ - σ instead of confidence intervals or Bayesian estimates?**
- **Tradeoffs**: $\mu - \sigma$ is computationally cheap and provides a "safety margin" that the agent can optimize directly. Bayesian estimates or bootstrap uncertainty are more robust but would significantly slow down the agent's inner loop (which runs multiple benchmarks per iteration). It is a "pragmatic engineering" choice for fast iteration.

---

### Sandbox / Security

**11. What is your threat model?**
The primary focus is **accidental bad code** (infinite loops, memory leaks) and **restricted code execution**. 
- **Protection**: We use an in-memory `exec()` sandbox with restricted globals (`exec_globals`).
- **Exfiltration/Destruction**: Gated by the `ADIS_ALLOW_EXEC` environment variable. Without this explicit user consent, the agent cannot run any generated code.

**12. How do you prevent malicious code like `__import__("os").system(...)`?**
The sandbox is **namespace-restricted**. We provide a specific dict of allowed globals (`pd`, `np`, `scikit-learn` components). Standard dangerous built-ins are not injected. However, it is not a full containerized sandbox (like Docker) yet; it relies on **trust-based** execution with structural restrictions.

**13. Can the agent create covert side effects?**
In the current implementation, the agent has access to `pd` and `np`. While it could technically write to the filesystem via `df.to_csv()`, it is not provided with `os` or `shutil` by default. Multiprocessing is restricted by the execution environment.

**14. How deterministic is agent execution?**
- **Non-deterministic**: Because it uses LLMs (temperature > 0), two runs will likely follow different trajectories.
- **Statistical Similarity**: We set random seeds (`random_state=42`) in the underlying ML models to ensure that *for a given piece of code*, the evaluation is deterministic.

---

### Systems Design

**15. What is the formal contract between pipeline stages?**
Defined in `adis/schemas.py`. Every stage must return a dictionary that can be validated against a Pydantic model (e.g., `IngestionMetadata`, `ModelResult`, `Vulnerability`). For example, `run_cleaning()` guarantees a cleaned `df` and a `CleaningLog`.

**16. How do you handle schema drift?**
ADIS uses fixed `feature_names` during benchmarking. If an LLM generates code that mutates dtypes or causes cardinality explosions, the code will likely fail in the sandbox during the `df_engineered = build_features(df)` call, and the agent will receive the error in its learning log to fix it in the next iteration.

**17. Can ADIS reason about computational cost?**
The `ModelResult` schema includes `training_time_seconds`. While the agent currently optimizes for the metric score, the Critic can be extended to flag "Inefficient Feature Engineering" if the training time spikes significantly for marginal gains.

**18. Does the agent optimize for accuracy, robustness, or fairness?**
It optimizes for the **Primary Metric** (ROC-AUC for binary class, F1 for multi, R² for regression). Robustness is implicitly handled by the **Pessimistic Score**. Fairness and Latency are not yet formal optimization targets but are planned for the Critic's audit layer.

**19. How do you prevent optimization target drift?**
The **Critic** acts as the guardrail. If the agent finds a way to "game" the ROC-AUC (e.g., via leakage), the Critic rejects the iteration even if the score is high. This separates "Performance" (Agent) from "Validity" (Critic).

**20. Can the Critic audit the reasoning process?**
Currently, it audits **outputs** (metrics and features). The "reasoning" (the LLM's hypothesis) is visible in the log, but the Critic does not yet "critique the logic" before execution—it critiques the **result** of the logic.

---

### Long-Term Architecture

**21. What happens when the LLM becomes stronger?**
ADIS becomes more powerful at discovering complex features but also more "dangerous" in its ability to find subtle leaks. The safety architecture (the Critic) must evolve to be **model-based** (recursive auditing) to keep pace.

**22. Could ADIS support non-tabular data?**
The core `ADISPipeline` and `AutoResearchAgent` are currently **tabular-centric**. Supporting Graph ML or Time-Series would require specialized `Ingestion` and `EDA` modules, but the "Agent-Critic" loop is data-agnostic.

**23. Is the Critic rule-based or model-based?**
It is currently a **hybrid**: rule-based heuristics (for speed/certainty) and statistical analysis. We are moving toward a **model-based Critic** where a second LLM audits the code and results for subtle logic flaws.

**24. What is the weakest component in the system today?**
The **Sandbox Isolation**. Running `exec()` in the same process as the pipeline is a security and stability risk. A production version would require a gRPC-based containerized execution worker.

**25. If ADIS failed catastrophically in production, what would the postmortem say?**
*"The agent discovered a high-fidelity proxy for the target variable that was semantically valid (e.g., 'customer_outcome_score') but was actually calculated using post-event data, and the Critic's heuristics lacked the domain context to flag it as leakage."*

---

### Research-Level Questions

**26. Can autonomous feature engineering outperform human experts?**
**Yes, in speed and breadth of search.** An agent can test 100 interaction terms in 10 minutes; a human cannot. However, humans still hold a ceiling on **Domain Synthesis**—connecting external context (e.g., "The factory was closed that day") that isn't in the dataset.

**27. Is ADIS fundamentally search, reasoning, or verification?**
It is a **Verification System** dressed as an Optimization System. The "Intelligence" lies in the **Critic's ability to say 'No'**. The search/reasoning is just the engine; the verification is the steering wheel.

**28. Could the Critic itself become the bottleneck?**
Absolutely. If the Agent improves faster than the Critic's ability to detect invalidity, the system will optimize for "valid-looking failures." This is why recursive auditing is critical.

**29. Can ADIS explain why a feature improved generalization?**
Only at the "Hypothesis" level. It can say "I created a rolling mean to capture trend," and then observe the score increase. It does not yet perform **Causal Attribution** to prove the improvement wasn't noise.

**30. What would it take for ADIS to move to "Autonomous Scientific Discovery"?**
Moving from **"Predict Y given X"** to **"Discover the Relationship between X and Y."** This requires the agent to generate and test *theories* (causal models) rather than just *features*. ADIS is currently a "Feature Scientist"; the next step is "Model Scientist."
