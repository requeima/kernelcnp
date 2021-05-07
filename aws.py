import argparse
import time

import aws
import aws.experiment as experiment
import numpy as np
import wbml.out as out

parser = argparse.ArgumentParser()
parser.add_argument("--sync-stopped", action="store_true")
parser.add_argument("--spawn", type=int)
parser.add_argument("--kill", action="store_true")
parser.add_argument("--start", action="store_true")
args = parser.parse_args()

experiment.config["ssh_user"] = "ubuntu"
experiment.config["ssh_pem"] = "~/.ssh/kernelcnp.pem"
experiment.config["ssh_setup_commands"] = [
    ["cd", "/home/ubuntu/kernelcnp"],
]

if args.sync_stopped:
    with out.Section("Syncing all stopped instances in five batches"):
        for batch in np.array_split(aws.get_state("stopped"), 5):
            # Start the instances.
            aws.start(*batch)

            try:
                # Wait for the instances to have booted.
                out.out("Waiting a minute for the instances to have booted...")
                time.sleep(60)

                # Refresh the instances to get the IPs.
                instance_ids = [instance["InstanceId"] for instance in batch]
                batch = aws.get_instances(*instance_ids)

                # Sync.
                experiment.sync(
                    sources=[
                        "/home/ubuntu/kernelcnp/_experiments",
                        "/home/ubuntu/kernelcnp/logs",
                    ],
                    target="sync",
                    ips=[instance["PublicIpAddress"] for instance in batch],
                )
            finally:
                # Stop the instances again.
                aws.stop(*batch)

    out.out("Syncing completed: not continuing execution of script.")
    exit()


if args.spawn:
    with out.Section("Starting all stopped instances"):
        aws.start_stopped()

    with out.Section("Spawning instances"):
        experiment.spawn(
            image_id="",
            total_count=args.spawn,
            instance_type="p3.2xlarge",
            key_name="",
            security_group="",
        )

    while not aws.check_all_running():
        out.out("Waiting for all instances to be running...")
        time.sleep(5)

    out.out("Waiting a minute for all instances to have booted...")
    time.sleep(60)

if args.kill:
    with out.Section("Killing all experiments"):
        experiment.kill_all()

if args.start:
    configs = []
    datas = (
        "eq",
        "matern52",
        "noisy-mixture",
        "weakly-periodic",
        "sawtooth",
        "mixture",
    )
    models = (
        "GNP",
        "AGNP",
        "ANP",
        "ConvGNP",
        "ConvCNP",
        "ConvNP",
        "FullConvGNP",
    )
    for data in datas:
        for model in models:
            configs.append({"model": model, "data": data})
    num_instances = len(aws.get_running_ips())
    pieces = np.array_split(configs, num_instances)

    def log_path(config):
        return "{model}_{data}_{loss}.log".format(**config)

    def determine_commands(config):
        return [
            [
                "./train.sh",
                "--model",
                config["model"],
                "--data",
                config["data"],
                "--loss",
                config["loss"],
                "2>&1",
                "|",
                "tee",
                f"logs/train_{log_path(config)}",
            ],
        ]

    with out.Section("Starting experiments"):
        out.kv("Number of configs", len(configs))
        out.kv("Number of instances", num_instances)
        out.kv("Runs per instance", max([len(piece) for piece in pieces]))
        experiment.ssh_map(
            *[
                [
                    ["git", "pull"],
                    ["rm", "-rf", "logs", "_experiments"],
                    ["mkdir", "logs"],
                    *sum([determine_commands(config) for config in configs], []),
                    ["logout"],
                ]
                for configs in pieces
            ],
            start_tmux=True,
            in_tmux=True,
        )

while True:
    out.kv("Instances still running", len(aws.get_running_ips()))
    experiment.sync(
        sources=[
            "/home/ubuntu/kernelcnp/_experiments",
            "/home/ubuntu/kernelcnp/logs",
        ],
        target="sync",
        shutdown=True,
    )
    out.out("Sleeping for two minutes...")
    time.sleep(2 * 60)
