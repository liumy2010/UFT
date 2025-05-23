pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3
pip3 install ray==2.42.1

pip install -e .

pip3 install flash-attn --no-build-isolation
pip install wandb IPython matplotlib