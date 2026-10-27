# Monte Carlo experiments

These scripts run the thesis controller (the code of `../code/rsvc_steering.py`)
many times in the Robotarium Python simulator without graphics.

## Setup

```bash
pip install numpy scipy matplotlib pandas cvxopt
pip install git+https://github.com/robotarium/robotarium_python_simulator.git
```

## Run (from this folder)

```bash
python run_trial.py                          # quick test: one run of each method, N = 10
python montecarlo.py --trials 30 --out results/mc.csv   # main study (~25 min on 2 cores)
python ablation.py                           # theta_max and unwinding ablations
python symmetric.py                          # perfectly symmetric cases (no noise)
python analyze.py results/mc.csv             # table + figures for Chapter 5
```

`analyze.py` writes `results/mc_table.tex` (included by Chapter 5) and the
figures `mc_*.pdf` into `../figures/`, so after re-running it just recompile
the thesis.

## Files

| File | Content |
|---|---|
| `rsvc_agent.py` | The controller, same logic as the Robotarium script, plus switches for the baselines |
| `run_trial.py` | One trial: scenario, simulation, metrics |
| `montecarlo.py` | All trials -> CSV |
| `ablation.py` | Ablation of `theta_max` and of the unwinding |
| `analyze.py` | Summary table and figures |
| `results/` | CSV files of the runs reported in the thesis |

Methods: `proposed`, `rsvc` (no deadlock handling), `noise` (random
perturbation of u0 while deadlocked), `sbc` (Robotarium barrier certificates
on the nominal controller).
