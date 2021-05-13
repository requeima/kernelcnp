import aws
from itertools import product

KEY = "elukbook"
REPO = "kernelcnp"
SECURITY_GROUP = "sg-00e6c4ed6ef493a3a"
IMAGE_ID = "ami-043f324346849c8f8"

aws.config["ssh_user"] = "ubuntu"  # "ec2-user"
aws.config["ssh_key"] = f"/home/stratis/.ssh/{KEY}.pem"
aws.config["setup_commands"] = [
    f"cd /home/ec2-user/{REPO}",
    "ssh-keygen -F github.com || ssh-keyscan github.com >> ~/.ssh/known_hosts",
    "git pull",
]

# Model and data generator configurations
data_generators = [
    "random",
    "sawtooth",
    "eq",
    "matern",
    "noisy-mixture",
    "weakly-periodic",
]

# Conditional models -- without FullConvGNP
cond_models = ["GNP", "AGNP", "convGNP"]

# Latent models
latent_models = ["ANP", "convNP"]

# Covariances for conditional models
covs = ["innerprod-homo", "kvv-homo", "meanfield"]

# Seeds to try
seeds = [str(i) for i in range(1)]

# Configs for conditional models
cond_configs = list(product(seeds, data_generators, cond_models, covs))

# Configs for FullConvGNP
fcgnp_configs = list(product(seeds, data_generators, ["FullConvGNP"], ["meanfield"]))

# Configs for latent models
latent_configs = list(product(seeds, data_generators, latent_models, ["meanfield"]))

# Configs for all experiments
configs = cond_configs + fcgnp_configs + latent_configs

commands = [
    [
        f"mkdir -p logs",
        f"python -u train.py {gen} {model} {cov} --x_dim 1 --seed {seed} --gpu 0"
        f" 2>&1 | tee \"logs/gen_{gen}_model_{model}_cov_{cov}_seed_{seed}.txt\"",
    ]
    for seed, gen, model, cov in configs
]

aws.manage_cluster(
    commands,
    instance_type="p3.2xlarge",
    key_name=KEY,
    security_group_id=SECURITY_GROUP,
    image_id=IMAGE_ID,
    sync_sources=[
        f"/home/ubuntu/{REPO}/toy-results",
        f"/home/ubuntu/{REPO}/logs",
    ],
    sync_target=aws.LocalPath("sync"),
    monitor_call=aws.shutdown_when_all_gpus_idle_call(duration=600),
    monitor_delay=120,
    monitor_aws_repo="/home/ubuntu/aws",
)
