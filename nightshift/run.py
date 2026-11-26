"""CLI: python -m nightshift.run --engine {stub,tiny,hf} [options]"""

from __future__ import annotations

import argparse
import dataclasses

from .experiment import Config, Experiment


def main(argv=None):
    ap = argparse.ArgumentParser(description="nightshift: a model that learns on the job and consolidates at night")
    for f in dataclasses.fields(Config):
        default = f.default
        flag = f"--{f.name.replace('_', '-')}"
        if f.name == "gate":
            ap.add_argument(flag, choices=("log", "enforce", "off"), default=default,
                            help="log: score updates but never roll back (default, fair comparison); "
                                 "enforce: roll back regressions; off: no gate scoring")
            ap.add_argument("--no-gate", dest="gate", action="store_const", const="off", help="same as --gate off")
        elif isinstance(default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, tuple):
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=lambda s, t=type(default[0]): tuple(t(x) for x in s.split(",")),
                            default=default)
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(default), default=default)
    args = ap.parse_args(argv)
    cfg = Config(**vars(args))

    exp = None
    if cfg.engine == "stub":
        from .policy import StubPolicy
        exp = Experiment(cfg, None)
        exp.policy = StubPolicy(exp.lookup)
        exp._wrap(exp.policy)
    elif cfg.engine == "tiny":
        from .hf_policy import HFPolicy
        exp = Experiment(cfg, HFPolicy(tiny=True, lr=cfg.lr, lora_r=cfg.lora_r, gen_batch=cfg.gen_batch, seed=cfg.seed))
    elif cfg.engine == "hf":
        from .hf_policy import HFPolicy
        exp = Experiment(cfg, HFPolicy(cfg.model, lr=cfg.lr, lora_r=cfg.lora_r, gen_batch=cfg.gen_batch, seed=cfg.seed))
    else:
        raise SystemExit(f"unknown engine {cfg.engine}")
    exp.run()
    print(f"\nwrote {cfg.out_dir}/run_artifacts.json and traces.jsonl")


if __name__ == "__main__":
    main()
