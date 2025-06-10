# Configuration Guide

All configurations target the 40-mode GMM unless otherwise stated.

### CTDS Configs
| Notebook | Purpose | 
| -------- | ------- |
| `learned_esh.yaml`   | CTDS w/ learned-linear continuum, ESH proposal. |
| `learned_tempered.yaml` | CTDS w/ learned linear continuum, tempered proposal.   |
| `interpolant_tempered.yaml` | CTDS w/ interpolant continuum, tempered proposal. |

### NETS Configs

| Notebook | Purpose | 
| -------- | ------- |
| `learned_baseline.yaml`   | NETS w/ learned-linear path, baseline proposal. |
| `learned_overdamped.yaml`   | NETS w/ learned-linear path, overdamped Langevin proposal. |
| `learned_inertial.yaml`   | NETS w/ learned-linear path, inertial Langevin proposal. |
| `learned_esh.yaml`   | NETS w/ learned-linear path, ESH Langevin proposal. |
| `interpolant_overdamped.yaml`   | NETS w/ interpolant path, overdamped Langevin proposal. |
| `interpolant_inertial.yaml`   | NETS w/ interpolant path, inertial Langevin proposal. |
| `interpolant_esh.yaml`   | NETS w/ interpolant path, ESH Langevin proposal. |

