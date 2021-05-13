import aws
 
KEY = "elukbook"
REPO = "kernelcnp"
SECURITY_GROUP = "sg-00e6c4ed6ef493a3a"
IMAGE_ID = "ami-08958d372207fc80b"
 
aws.config["ssh_user"] = "ubuntu" # "ec2-user"
aws.config["ssh_key"] = f"~/.ssh/{KEY}.pem"
aws.config["setup_commands"] = [
 f"cd /home/ec2-user/{REPO}",
 "ssh-keygen -F github.com || ssh-keyscan github.com >> ~/.ssh/known_hosts",
 "git pull"
]
 
commands = [
 ["mkdir -p results", "touch results/one.txt", "echo "],
 ["mkdir -p results", "touch results/two.txt"],
 ["mkdir -p results", "touch results/three.txt"],
]
 
aws.manage_cluster(
 commands,
 instance_type="t2.small",
 key_name=KEY,
 security_group_id=SECURITY_GROUP,
 image_id=IMAGE_ID,
 sync_sources=[f"/home/ubuntu/{REPO}/results"],
 sync_target=aws.LocalPath("sync"),
 monitor_call=aws.shutdown_timed_call(duration=60),
 monitor_delay=60,
 monitor_aws_repo="/home/ubuntu/aws",
)
