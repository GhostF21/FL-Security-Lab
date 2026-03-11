# Literature Review Notes
## [1] McMahan et al. 2017 — FedAvg

**Key contribution:** Introduced Federated Learning as a practical paradigm for training deep neural networks on decentralized, privacy-sensitive data. Proposed the FedAvg algorithm, which dramatically reduces communication rounds compared to naive distributed SGD by having each client perform multiple local SGD steps before syncing. Demonstrated effectiveness across five model architectures and four datasets (AISTATS 2017).
**Algorithm 1 formula: Global model update via weighted average:** w_{t+1} = Σ_k (n_k / n) · w_k^{t+1}
where w_k^{t+1} is client k's locally-trained model after E epochs of local SGD, n_k is client k's dataset size, and n = Σ n_k is the total number of data points. The server broadcasts w_t, clients train locally, then the server aggregates.
**Why relevant to this project:** FedAvg is the baseline protocol we implement and evaluate in Week 3. All attack and defense experiments run on top of it. Understanding its weighted-average aggregation step is essential because it is precisely this step that malicious clients can exploit — the server has no visibility into how client models were trained, creating the attack surface for both data poisoning (Week 4) and model replacement attacks (Week 5).
* * *

## [2] Blanchard et al. 2017 — Krum

**Key contribution:** Proved that no linear aggregation rule (including simple averaging) can tolerate even a single Byzantine worker. Proposed Krum, the first provably Byzantine-resilient aggregation rule for distributed SGD. Krum selects the single gradient vector whose summed squared distances to its n - f - 2 nearest neighbors is minimal, effectively choosing the update most "surrounded" by other updates and farthest from outliers (NeurIPS 2017).
**Byzantine requirement:** Requires 2f + 2 < n, meaning the number of Byzantine workers f must be strictly less than (n - 2) / 2. For our 10-client setup this means the system can tolerate at most 3–4 Byzantine clients. Time complexity is O(n² · d), linear in gradient dimension d.
**Why relevant:** Krum is the primary Byzantine-robust defense we implement in Week 6 (Bi-Weekly Report 2). It gives us a theoretically-grounded baseline defense against both label-flip and model-replacement attacks. Its strict 2f + 2 < n requirement is also a limitation we will test — above the threshold, Krum's guarantees break down, which we will demonstrate experimentally.
* * *

## [3] Yin et al. 2018 — Trimmed Mean

**Key contribution:** Developed two distributed gradient descent algorithms — one based on coordinate-wise median and one on coordinate-wise trimmed mean — that are provably robust against Byzantine failures and achieve order-optimal statistical error rates for strongly convex, non-strongly convex, and smooth non-convex loss functions. Proved matching lower bounds showing these rates are statistically optimal (ICML 2018).
**Trimming mechanism:** For each coordinate j of the gradient vector, sort the m worker values for that coordinate, remove the top β fraction and the bottom β fraction (where β is the known Byzantine fraction), then average the remaining (1 - 2β) · m values. Applied independently per coordinate. Requires β < 1/2. In practice β = 0.05 (5%) is typical for moderate attack scenarios.
**Why relevant:** Trimmed Mean serves as a complementary defense to Krum in Week 6. Unlike Krum (which selects a single client's full gradient), Trimmed Mean aggregates information from all clients but clips extremes per-coordinate. This makes it more statistically efficient under moderate attack rates and is directly comparable to Krum in our Week 6 defense evaluation. The coordinate-wise operation also makes it computationally cheap and easy to implement.
* * *

## [4] Bagdasaryan et al. 2020 — Backdoor in FL

**Key contribution:** Demonstrated that model poisoning via model replacement is a far more powerful attack than data poisoning alone. Any single selected participant can inject a persistent backdoor into the global model — achieving 100% backdoor task accuracy in a single round — by scaling up their malicious model update before submission. Also introduced the constrain-and-scale technique to evade anomaly-detection defenses by incorporating evasion into the attacker's training objective (AISTATS 2020).
**Model replacement** technique: The attacker trains a backdoored local model M_mal that performs well on both the main task and the backdoor task. Before submission, the attacker scales the update by factor γ = n / η (where n is the number of clients and η is the server's learning rate / aggregation weight) to ensure the global model G_{t+1} is effectively replaced by M_mal after weighted averaging. A scaling factor of ~100 is sufficient to dominate aggregation. The constrain-and-scale variant further bounds weight norms to evade detection.
**Why relevant:** This paper is the theoretical foundation for our Week 4–5 backdoor attack implementation. Our backdoor.py implements a simplified version of this — injecting a pixel-pattern trigger and relabeling samples. In Week 5 we add the scaling factor (γ) to implement the full model replacement attack, directly replicating Bagdasaryan's core finding. SecAgg+ (Paper [6]) is explicitly mentioned in this paper as a barrier to anomaly detection, creating a direct link to our Week 1 SecAgg+ analysis.
* * *

## [5] Fung et al. 2018 — FoolsGold

**Key contribution:** Identified sybil-based poisoning (multiple colluding malicious clients sharing a common objective) as a distinct and underaddressed threat to federated learning. Proposed FoolsGold, a defense that identifies sybils by tracking the diversity of gradient update history per client and adaptively reducing the learning rate contribution of clients whose accumulated gradients are highly similar to other clients' (arXiv 2018, journal version 2020).
**Detection signal:** FoolsGold uses cosine similarity between clients' cumulative historical gradient vectors as its detection signal. Honest clients, training on unique non-IID data, contribute gradients pointing in diverse directions. Sybils, sharing a poisoning objective, contribute gradients that are directionally similar (high cosine similarity). FoolsGold computes pairwise cosine similarities between all clients' historical gradient sums, then reduces the per-round learning rate α_i of clients with high maximum similarity to any other client, down to near zero. A pardoning mechanism prevents penalizing honest clients that incidentally look similar due to SGD variance.
**Why relevant:** FoolsGold is our detection mechanism for Week 9 and a defense we compare against Krum in the multi-client attack scenario. It is particularly relevant because, unlike Krum, FoolsGold makes no assumption about the minority proportion of attackers — it can theoretically detect sybils even when they dominate the federation. This is critical to our Week 4 experiments with 30% malicious fraction (3 out of 10 clients), which exceeds Krum's 2f+2 < n threshold.
* * *

## [6] Bonawitz et al. 2017 — SecAgg+

**Key contribution:** Designed the first practical, communication-efficient, failure-robust Secure Aggregation protocol for federated learning (SecAgg / SecAgg+), presented at ACM CCS 2017. The protocol allows a server to compute only the sum of client model updates without learning any individual client's contribution, even if some clients drop out mid-protocol. Proved security in both honest-but-curious and active adversary settings.
Protocol steps (4 rounds):

**AdvertiseKeys:** Each client generates and broadcasts a Diffie-Hellman public key and a secret-sharing public key to the server, which distributes the full key directory to all clients.
**ShareKeys:** Each client pair (u, v) computes a shared pairwise seed via Diffie-Hellman key agreement. Each client secret-shares its private key among all other clients using t-out-of-n Shamir Secret Sharing, enabling dropout recovery.
**MaskedInputCollection:** Each client masks its gradient vector x_u with canceling pairwise pseudorandom masks p_{u,v} (generated from shared seeds via PRG) and a private self-mask p_u, then uploads the masked vector y_u = x_u + Σ_v p_{u,v} + p_u.
**Unmasking:** Surviving clients send Shamir shares of dropped-out clients' private keys. The server reconstructs masks for dropouts and cancels all masks from the sum, recovering Σ_u x_u.
**Privacy-security tension:** SecAgg+ guarantees that the server learns only the aggregate sum of updates not individual gradients — which is essential for user privacy. However, this same property directly conflicts with Byzantine robustness and backdoor defense: the server cannot inspect individual updates to detect anomalies, outliers, or poisoned models. Defenses like Krum, Trimmed Mean, and FoolsGold all require access to individual client gradients, which SecAgg+ is specifically designed to hide. This fundamental tension is a core theme of our project we analyze it in the SecAgg+ notes (Week 1) and revisit it in our final report when evaluating whether defenses remain viable under cryptographic aggregation constraints.
