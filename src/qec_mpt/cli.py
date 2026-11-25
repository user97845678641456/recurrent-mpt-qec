"""Evaluate released BB72 and BB144 models with absolute-LLR CS8."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .data import CODES, POINTS, matrices, sample_dem, test_data


def init_decoder(code, p):
    global h, logical, rows, packed, weights
    from .osd import pack

    data = matrices(code, p)
    h, logical, rows = data["h"], data["l"], data["rows"]
    q = data["q"]
    packed = pack(h)
    weights = np.log((1 - q) / q)


def decode_batch(batch):
    from .osd import one

    detectors, labels, logits = batch
    failures = []
    for syndrome, label, z in zip(detectors[:, rows], labels, logits):
        z = z.astype(np.float64)
        syndrome = syndrome.astype(np.uint8)
        fault, _ = one(packed, syndrome, z, .5 * (weights - z), True, 8, False)
        if not np.array_equal(h @ fault % 2, syndrome):
            raise RuntimeError("CS8 recovery violated detector constraints")
        failures.append(bool(np.any(logical @ fault % 2 != label)))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--code", choices=CODES, required=True)
    evaluate.add_argument("--noise", type=float, choices=POINTS, default=.005)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--dataset", choices=("report", "fresh"), default="report")
    evaluate.add_argument("--start", type=int, default=5000)
    evaluate.add_argument("--shots", type=int, default=3000)
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--batch", type=int, default=64)
    evaluate.add_argument("--seed", type=int, default=20261126)
    training = sub.add_parser("train")
    training.add_argument("--code", choices=CODES, required=True)
    training.add_argument("--device", default="cuda")
    training.add_argument("--samples", type=int, required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--initialize")
    training.add_argument("--seed", type=int, default=21501115)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--schedule", choices=("cosine", "step", "two_stage"), default="step")
    training.add_argument("--batch", type=int, default=64)
    training.add_argument("--validation-shots", type=int, default=1024)
    training.add_argument("--eval-every", type=int, default=100000)
    args = parser.parse_args()
    if args.command == "train":
        import torch
        from .model import FINAL_DEPTH
        from .training import train

        torch.set_num_threads(4)
        args.width = 96
        args.depth = FINAL_DEPTH[args.code]
        args.fault_neighbors = True
        train(args)
        return
    if args.workers < 1 or args.batch < 1 or args.shots < 1 or args.start < 0:
        parser.error("Require positive workers, batch and shots, and nonnegative start")
    if args.dataset == "report" and args.noise != .005:
        parser.error("The archived benchmark uses p=.005. Use --dataset fresh for other noise strengths.")

    import torch
    from .model import infer

    torch.set_num_threads(4)
    if args.dataset == "report":
        inputs = test_data(args.code, args.noise)
        selection = slice(args.start, args.start + args.shots)
    else:
        inputs = sample_dem(args.code, args.noise, args.shots, args.seed)
        selection = slice(None)
    d, y = inputs["detectors"][selection], inputs["labels"][selection]
    if len(d) != args.shots:
        parser.error("Requested frame range exceeds archived test set")
    z = infer(args.code, args.noise, d, args.device, batch=args.batch)
    batches = [(d[i:i+200], y[i:i+200], z[i:i+200]) for i in range(0, len(d), 200)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_decoder,
                             initargs=(args.code, args.noise)) as pool:
        failures = sum(sum(part) for part in pool.map(decode_batch, batches))
    print(json.dumps(dict(code=args.code, p=args.noise, shots=len(d),
                          failures=failures, block_failure_rate=failures / len(d))))


if __name__ == "__main__":
    main()
