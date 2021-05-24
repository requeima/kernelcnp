import aws
from itertools import product

KEY = "elukbook"
REPO = "kernelcnp"
SECURITY_GROUP = "sg-00e6c4ed6ef493a3a"
IMAGE_ID = "ami-0b9e8bb933b9914b3"

aws.config["ssh_user"] = "ubuntu"  # "ec2-user"
aws.config["ssh_key"] = f"/home/stratis/.ssh/{KEY}.pem"
aws.config["setup_commands"] = [
    f"cd /home/ubuntu/{REPO}",
    "ssh-keygen -F github.com || ssh-keyscan github.com >> ~/.ssh/known_hosts",
    "git fetch",
    "git reset --hard origin/main",
    ". venv/bin/activate"
]

# Model and data generator configurations
data_generators = [
    "noisy-mixture-slow",
    "weakly-periodic-slow"
]

# Conditional models -- without FullConvGNP
cond_models = ["GNP", "AGNP", "convGNP"]

latent_models = ["ANP", "ConvNPHalfUnet"]

# Covariances for conditional models
covs = ["innerprod-homo", "kvv-homo", "meanfield"]

# Seeds to try
seeds = ["0"]

# Configs for conditional models
cond_configs = list(product(seeds, data_generators, cond_models, covs))
latent_configs = list(product(seeds, data_generators, latent_models, ["meanfield"]))
fcgnp_configs = list(product(seeds, data_generators, ["FullConvGNP"], ["meanfield"]))

configs = cond_configs + latent_configs + fcgnp_configs

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
    monitor_call=aws.shutdown_when_all_gpus_idle_call(duration=1200),
    monitor_delay=120,
    monitor_aws_repo="/home/ubuntu/aws",
)
