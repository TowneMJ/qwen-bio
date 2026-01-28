cd ~/qwen-bio
apt update && apt install -y python3.10-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Core dependencies
pip install torch==2.9.0
pip install axolotl datasets accelerate

# Downgrade vllm (0.14.0 has compatibility issues with axolotl)
pip install vllm==0.12.0

# Fix axolotl telemetry bug
touch venv/lib/python3.10/site-packages/axolotl/telemetry/whitelist.yaml

# Install dependencies for question generation
pip install requests

# Evaluation harness
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e ".[vllm]"
cd ..

echo "Setup complete. Activate with: source venv/bin/activate"