cd ~/qwen-bio
apt update && apt install -y python3.10-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e ".[vllm]"
cd ..