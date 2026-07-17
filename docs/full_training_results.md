# Full Training Results

Full training was run on the restored `Dataset 50_0.5_0.02.ini` dataset with:

```bash
python3 -m djss_rl.cli train --episodes 1000 --output-dir outputs/full-training-20260717
```

The run completed all 1000 episodes.

## Training Summary

- Best episode: 722 / 1000
- Best episode reward: 1.50
- Best episode tardiness rate: 0.58
- Best episode mean machine utilization: 79.50%
- Final episode reward: -34.50
- Final episode tardiness rate: 0.61
- Generated checkpoint: `outputs/full-training-20260717/Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth`
- Training log: `outputs/full-training-20260717/training.log`

The training CLI currently does not force a random seed, so these are the observed results from this run rather than a deterministic benchmark.

## Evaluation After Full Training

The best checkpoint from the full run was evaluated with:

```bash
python3 -m djss_rl.cli evaluate --checkpoint 'outputs/full-training-20260717/Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth'
```

| Method | Tardiness rate | Makespan | Mean machine utilization |
|---|---:|---:|---:|
| Full-trained checkpoint | 0.6034 | 47h 18m | 76.13% |
| SPT_DR_O | 0.6207 | 45h 54m | 75.09% |
| MRT_DR_O | 0.6355 | 42h 14m | 95.31% |
| Original checkpoint | 0.6404 | 50h 55m | 79.44% |
| ATC_DR_O | 0.7167 | 47h 18m | 74.44% |
| LRT_DR_O | 0.7414 | 53h 59m | 75.93% |
| SLK_DR_O | 0.8177 | 55h 23m | 72.71% |
| LSPO_DR_O | 0.8276 | 55h 20m | 72.73% |
| EDD_DR_O | 0.8374 | 51h 38m | 78.43% |
| MCR_DR_O | 0.8424 | 53h 41m | 75.82% |
| CR_DR_O | 0.8547 | 52h 58m | 74.95% |

Lower tardiness rate is better. On this run, the full-trained checkpoint produced the best tardiness rate among the evaluated methods.
