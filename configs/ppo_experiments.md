# PPO LunarLander-v3 — experimentos

Todos os treinos usaram `train.py --config <nome> --override exp_name=<run> seed=<seed>`.
As configurações YAML neste diretório são suficientes para repetir as comparações principais.
`ppo_lunarlander_value` é a configuração final. As seeds 1 e 2 dessa configuração,
assim como as duas do A2C, usaram 2 milhões de passos planejados.

| Runs | Configuração | Overrides adicionais | Passos planejados |
|---|---|---|---:|
| `ppo-base-s1/s2` | `ppo_lunarlander` | `capture_video=false` | 1 milhão |
| `ppo-epochs10-s1` | `ppo_lunarlander` | `capture_video=false ppo.update_epochs=10` | 1 milhão |
| `ppo-long-s1` | `ppo_lunarlander` | `capture_video=false num_envs=16 ppo.num_steps=1024 ppo.num_minibatches=256 ppo.gamma=0.999 ppo.gae_lambda=0.98` | 1 milhão |
| `ppo-nonanneal-s1` | `ppo_lunarlander` | `capture_video=false ppo.anneal_lr=false` | 1 milhão |
| `ppo-tuned-s1/s2` | `ppo_lunarlander_tuned` | nenhum | 2 milhões |
| `ppo-lowent-s1` | `ppo_lunarlander_lowent` | nenhum | 2 milhões |
| `ppo-novclip-s1` | `ppo_lunarlander_tuned` | `total_timesteps=1000000 ppo.clip_vloss=false` | 1 milhão |
| `ppo-value-s1/s2` | `ppo_lunarlander_value` | nenhum | 2 milhões |
| `a2c-lunar-s1/s2` | `a2c_lunarlander` | nenhum | 2 milhões |

O clipping da política (`ppo.clip_coef=0.2`) permaneceu ativo. A ablação
`ppo-novclip-s1` remove somente o clipping da perda de valor do crítico.
Os resultados, inclusive tentativas que não alcançaram a meta, estão no
[relatório W&B](https://wandb.ai/leonardo-leonardo2-federal-university-of-goi-s/ppo-assignment).
