# Secure Aggregation (SecAgg+) Notes

## What is SecAgg+?
Flower's SecAgg+ implements the SecAgg+ protocol from:
> Keith Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
> ACM CCS 2017. https://dl.acm.org/doi/10.1145/3133956.3133982

## Protocol Steps (Bonawitz et al. 2017)
1. **Setup**: Server generates pairwise shared secrets between clients
2. **Share Keys**: Clients exchange Diffie-Hellman public keys
3. **Masked Gradients**: Each client adds random masks summing to zero
4. **Server Aggregation**: Server sums masked gradients — masks cancel out
5. **Reconstruction**: If clients drop out, Shamir secret sharing reconstructs

## Key Parameters in Flower
- `num_shares`: How many shares each secret is split into
- `reconstruction_threshold`: Minimum shares needed to reconstruct

## Privacy-Security Trade-off
| Property | FedAvg (no SecAgg) | With SecAgg+ |
|----------|-------------------|--------------|
| Individual gradient visible | YES | NO |
| Norm-threshold detection | Works | DISABLED |
| Cosine clustering detection | Works | DISABLED |
| Privacy from server | Weak | Strong |
| Byzantine resilience (Krum) | Works | Works |

## Research Tension
SecAgg+ provides privacy but removes our ability to perform anomaly detection.
This is a fundamental trade-off in privacy-preserving FL security.
Must be discussed in Expected Outcomes and Limitations sections.
