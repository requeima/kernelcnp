import aws
from itertools import product

KEY = "elukbook"
REPO = "kernelcnp"
SECURITY_GROUP = "sg-00e6c4ed6ef493a3a"
IMAGE_ID = "ami-068bc7620e20aa12c"

aws.config["ssh_user"] = "ubuntu"  # "ec2-user"
aws.config["ssh_key"] = f"/home/stratis/.ssh/{KEY}.pem"
aws.config["setup_commands"] = [
    f"cd /home/ubuntu/{REPO}",
    "ssh-keygen -F github.com || ssh-keyscan github.com >> ~/.ssh/known_hosts",
    "git fetch",
    "git reset --hard origin/main",
    ". venv/bin/activate"
]

# Experiments which failed form the core set
failed_core = [
    ["0", "eq", "GNP", "kvv-homo"],   
    ["0", "noisy-mixture", "AGNP", "innerprod-homo"],   
    ["0", "sawtooth", "AGNP", "kvv-homo"],   
]

# Experiments which failed due to kvv memory
data_generators = [
    "noisy-mixture-slow",
    "weakly-periodic-slow"
]

models = [
    "GNP",
    "AGNP",
    "convGNP"
]

failed_kvv_memory = list(product(["0"],
                                 data_generators,
                                 models,
                                 ["kvv-homo"]))

# Experiments which failede due to invalid dict access
failed_dict_access = list(product(["0"],
                                  data_generators,
                                  ["convNPHalfUNet"],
                                  ["meanfield"]))

# Put all configs together and create commands
configs = failed_core + failed_kvv_memory + failed_dict_access

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
