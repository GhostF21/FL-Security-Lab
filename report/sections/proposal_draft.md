# AI-2: Federated Learning Security Lab
## Project Proposal
### Topic: Malicious Clients in Federated Learning
*Yusif Yagubzade — NeptuinID: SW92BP*

*Week 2 — Submitted: February 28, 2026*

---

## 1. Problem Statement

Federated Learning (FL) is a distributed machine learning paradigm in which a central server coordinates model training across a large population of clients without ever collecting their raw data. Instead, each client trains on its local dataset and transmits only model updates — gradients or weight deltas — back to the server, which aggregates them into an updated global model. This architecture is privacy-preserving by design: the server observes aggregated statistics rather than individual records, which makes FL attractive for applications in healthcare, finance, and mobile computing.

However, this decentralized architecture introduces a critical security vulnerability: the server cannot inspect, audit, or verify how any individual client produced its update. A malicious participant — whether a compromised device, a dishonest data owner, or an adversarial actor — can submit deliberately crafted updates designed to corrupt the global model. Two principal attack strategies have been identified in the literature. First, **data poisoning attacks** manipulate a client's local training data — for example, by flipping class labels or embedding hidden trigger patterns — causing the model to learn incorrect associations that degrade accuracy or introduce backdoor behaviors. Second, **model manipulation (model replacement) attacks** bypass data-level corruption entirely: the adversary directly crafts a poisoned model update and scales it up so that, after server aggregation, the global model is effectively replaced by the attacker's version, achieving near-perfect backdoor success in a single round.

The problem is compounded by the non-IID (non-independently and identically distributed) nature of real federated data. Because each client holds a private, heterogeneous local dataset, honest client updates are already highly variable. This variability provides natural camouflage for malicious updates, making anomaly-based detection substantially harder than in homogeneous settings.

This project investigates both attack families and evaluates two classes of countermeasures: **Byzantine-robust aggregation defenses** (Krum, Trimmed Mean) that replace naive averaging with outlier-resistant rules, and **gradient-anomaly detection** methods (FoolsGold, statistical fingerprinting) that identify suspicious contributors before aggregation. A further dimension of the problem is the interaction between these defenses and Secure Aggregation (SecAgg+): the cryptographic protocol used in production FL systems to guarantee that the server sees only the sum of client updates, not individual contributions. This privacy guarantee directly disables anomaly-based defenses, creating a fundamental tension between privacy and security that this project will analyze and document.

---

## 2. Background and Context

The foundations of Federated Learning were established by McMahan et al. [1], who introduced the FedAvg algorithm. FedAvg operates in rounds: the server broadcasts the current global model to a subset of clients; each client performs *E* epochs of local stochastic gradient descent on its private data; and the server aggregates the resulting model weights via a weighted average proportional to local dataset sizes:

$$w_{t+1} = \sum_k \frac{n_k}{n} \cdot w_k^{t+1}$$

McMahan et al. demonstrated that this approach reduces communication rounds by 10–100× compared to minibatch SGD while maintaining competitive accuracy across diverse models and datasets. FedAvg forms the baseline protocol in this project and is the direct target of all attack implementations.

The Byzantine robustness problem — the question of how to aggregate updates reliably when an unknown fraction of participants are adversarial — was studied theoretically by Blanchard et al. [2], who proved that no linear aggregation rule (including weighted averaging) can tolerate even a single Byzantine worker, and proposed Krum as the first provably Byzantine-resilient alternative. Krum selects the single gradient vector whose cumulative squared distance to its *n − f − 2* nearest neighbors is minimized, where *f* is the known number of Byzantine workers. The method requires `2f + 2 < n`, meaning for our 10-client federation, up to three malicious clients can be tolerated. Yin et al. [3] complemented this with coordinate-wise Trimmed Mean, an aggregation rule that removes the top and bottom *β* fraction of values for each gradient dimension independently before averaging, achieving order-optimal statistical convergence rates under Byzantine failures for strongly convex, non-convex, and smooth objectives.

On the attack side, Bagdasaryan et al. [4] demonstrated that model replacement is substantially more powerful than data poisoning. The attacker trains a backdoored local model *M_mal* that achieves high accuracy on both the primary task and a secret backdoor task (e.g., classifying images with a trigger pattern as a chosen target class). The update is then scaled by a factor *γ = n / η* — large enough to dominate the weighted average — such that the global model converges to *M_mal* after a single aggregation round. The constrain-and-scale variant additionally bounds weight norms to evade distance-based outlier detection. This paper establishes the theoretical basis for the Week 5 model manipulation experiments in this project.

Fung et al. [5] proposed FoolsGold, a defense specifically designed for the sybil threat model in which multiple colluding malicious clients share a common poisoning objective. The key insight is that sybil clients, optimizing toward the same backdoor target, submit gradient updates with high pairwise cosine similarity. Honest clients, drawing from unique non-IID data distributions, contribute directionally diverse updates. FoolsGold exploits this divergence: it tracks the cumulative gradient history of each client, computes pairwise similarities, and multiplicatively reduces the learning rate contribution of clients whose historical gradients are suspiciously similar to others. Critically, FoolsGold imposes no assumption on the fraction of malicious clients, making it applicable even when adversaries constitute a majority of participants. This property is directly relevant to our 30% attack fraction experiments.

Finally, Bonawitz et al. [6] developed SecAgg+, a practical Secure Aggregation protocol for FL presented at ACM CCS 2017. SecAgg+ operates in four rounds:

1. **AdvertiseKeys** — clients broadcast Diffie-Hellman public keys to the server, which distributes the full key directory to all participants.
2. **ShareKeys** — clients use *t*-out-of-*n* Shamir Secret Sharing to distribute their private keys among peers, enabling dropout recovery in later rounds.
3. **MaskedInputCollection** — each client masks its gradient vector with canceling pairwise pseudorandom vectors derived from shared DH seeds, then submits the masked update `y_u = x_u + Σ_v p_{u,v} + p_u`.
4. **Unmasking** — surviving clients reveal Shamir shares; the server reconstructs and cancels all masks, recovering the exact aggregate `Σ_u x_u`.

While SecAgg+ is essential for user privacy in production FL, it directly prevents the server from inspecting individual gradients — disabling the defenses of Krum, Trimmed Mean, and FoolsGold. Analyzing this privacy–security tension is a core contribution of this project.

---

## 3. Plan of Work

The project is organized into five sequential phases spanning 15 weeks, with bi-weekly reports, a mid-term report, an end-term report, and a final presentation as formal deliverables. All experiments will use Python 3.11 with PyTorch and the Flower (`flwr ≥ 1.8`) simulation framework. The primary dataset is MNIST (60,000 train / 10,000 test, 10 classes), with CIFAR-10 used for cross-dataset validation in Phase 3. Data is partitioned across 10 clients using a Dirichlet distribution with concentration parameter *α = 0.5*, which produces moderate non-IID heterogeneity representative of real federated settings. All code is version-controlled in a private GitHub repository.

### Phase 1 — Setup & Proposal (Weeks 1–2, Feb 16 – Mar 1)

Configure Ubuntu 24.04 environment with Miniconda, PyTorch (CPU/GPU), Flower, and all scientific Python dependencies. Establish GitHub project structure with modular source packages (`fl_core`, `attacks`, `defenses`, `detection`, `utils`). Download and partition MNIST and CIFAR-10 datasets. Run and analyze the official Flower SecAgg+ simulation example. Read all six foundational papers and complete literature notes. Submit this proposal.

**Deliverable:** Project Proposal (due Feb 28).

### Phase 2 — Attacks (Weeks 3–4, Mar 2 – Mar 13)

Implement the FedAvg baseline (SimpleCNN, FLClient, FedAvg strategy) and establish clean accuracy on MNIST non-IID (~97%). Implement two data poisoning attacks: (a) label-flipping (source class 1 → target class 7) and (b) pixel-trigger backdoor injection (3×3 white-pixel pattern, 50% poison rate). Run 6 experiments across attack types × malicious client fractions {10%, 20%, 30%}. Record main-task accuracy and backdoor attack success rate (ASR) for each configuration.

**Deliverable:** Bi-Weekly Report 1 (due Mar 13).

### Phase 3 — Defenses (Weeks 5–8, Mar 14 – Apr 13)

Implement model manipulation (gradient scaling with factor *γ = n/η*) in Week 5, then implement Krum aggregation (Weeks 6–7) and coordinate-wise Trimmed Mean (*β = 0.05*) as Byzantine-robust defenses. Evaluate each defense against each attack variant across all malicious fractions. Validate all results on CIFAR-10 in Week 8 to confirm dataset generalization. Produce accuracy recovery curves and defense comparison tables.

**Deliverable:** Bi-Weekly Report 2 (due Mar 27).

### Phase 4 — Detection (Weeks 9–11, Apr 14 – May 8)

Implement three anomaly detection methods: (a) FoolsGold cosine-similarity-based contribution reweighting, (b) gradient norm monitoring with configurable alert thresholds, and (c) principal-component-based anomaly scoring. Evaluate detection precision, recall, and F1 score for each method under all attack configurations. Run full ablation studies varying malicious fraction and attack intensity. Analyze the interaction between each detection method and SecAgg+, documenting which defenses remain viable under cryptographic aggregation constraints.

**Deliverables:** Bi-Weekly Report 3 (due Apr 17), Mid-Term Report (due Apr 27), Bi-Weekly Report 4 (due May 4).

### Phase 5 — Report & Presentation (Weeks 12–15, May 9 – Jun 16)

Write the full end-term report (~15 pages) covering all five phases: introduction, related work, methodology, results, and conclusion. Produce all final figures and tables. Build a 12-slide presentation deck and rehearse a 10-minute talk with a 5-minute live demo. Finalize the GitHub repository with a complete README, demo Jupyter notebook (`notebooks/demo/fl_security_demo.ipynb`), `requirements.txt`, and `environment.yml`.

**Deliverables:** Bi-Weekly Report 5 (due May 22), End-Term Report (due Jun 5), Final Presentation + Source Code (due Jun 16).

---

## 4. Expected Outcomes

This project will produce the following concrete research artifacts and empirical findings.

### 4.1 Code Library

A modular, well-documented Python library implementing: the FedAvg baseline with configurable federation parameters; label-flip and backdoor data poisoning attacks; gradient-scaling model replacement attacks; Krum and Trimmed Mean Byzantine-robust aggregation defenses; and FoolsGold, gradient-norm, and PCA-based anomaly detectors. All modules will be fully tested and reproducible via a single configuration file. The library will be published on GitHub with a comprehensive README and a demo notebook.

### 4.2 Empirical Results

Quantitative results across the full attack–defense–detection evaluation matrix. Specifically, we expect to demonstrate that:

- The FedAvg baseline achieves ~97% accuracy on MNIST non-IID before any attack.
- Label-flipping with 30% malicious clients reduces main-task accuracy to ~65–75%.
- Backdoor attacks with 30% malicious clients achieve an ASR of ~85–95% while preserving near-baseline main-task accuracy.
- Model replacement attacks achieve 100% ASR in a single round.
- Krum fully recovers main-task accuracy when the malicious fraction is below the `2f+2 < n` threshold (i.e., ≤30% of 10 clients), but fails above it.
- Trimmed Mean with *β = 0.05* provides partial recovery at higher attack fractions.
- FoolsGold achieves F1 > 0.85 for sybil detection under standard attack conditions, with degradation at very low malicious fractions where gradient diversity is insufficient.

### 4.3 Formal Deliverables

| Deliverable | Due Date |
|---|---|
| Bi-Weekly Report 1 | Mar 13, 2026 |
| Bi-Weekly Report 2 | Mar 27, 2026 |
| Bi-Weekly Report 3 | Apr 17, 2026 |
| Mid-Term Report | Apr 27, 2026 |
| Bi-Weekly Report 4 | May 4, 2026 |
| Bi-Weekly Report 5 | May 22, 2026 |
| End-Term Report (~15 pages) | Jun 5, 2026 |
| Final Presentation + Source Code | Jun 16, 2026 |

### 4.4 SecAgg+ Privacy–Security Analysis

A dedicated analysis section in the end-term report will examine the fundamental tension between SecAgg+ and Byzantine-robust defenses. Because SecAgg+ guarantees the server observes only the aggregate sum of client updates, it structurally prevents the inspection of individual gradients required by Krum, Trimmed Mean, and FoolsGold. We will quantify this trade-off by comparing attack success rates under standard FL versus SecAgg+-protected FL, and will evaluate proposed hybrid approaches (e.g., verifiable aggregation with zero-knowledge proofs, or trusted execution environments as partial mitigations). This analysis will serve as the project's primary conceptual contribution beyond the experimental evaluation, addressing an open research question with direct implications for deploying FL in adversarial settings without sacrificing user privacy.

---

## References

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in *Proc. AISTATS*, 2017, pp. 1273–1282.

[2] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent," in *Proc. NeurIPS*, 2017, pp. 119–129.

[3] D. Yin, Y. Chen, K. Ramchandran, and P. Bartlett, "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates," in *Proc. ICML*, 2018, pp. 5650–5659.

[4] E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, and V. Shmatikov, "How to Backdoor Federated Learning," in *Proc. AISTATS*, 2020, pp. 2938–2948.

[5] C. Fung, C. J. M. Yoon, and I. Beschastnikh, "Mitigating Sybils in Federated Learning Poisoning," arXiv preprint arXiv:1808.04866, 2018.

[6] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Talwar, and K. Rush, "Practical Secure Aggregation for Privacy-Preserving Machine Learning," in *Proc. ACM CCS*, 2017, pp. 1175–1191.
