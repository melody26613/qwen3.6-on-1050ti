# Qwen 3.6 35B A3B on 1050ti

### [Please refer to these slides first](./Qwen3_on_1050ti.pptx)

## Test Qwen3.6 35B A3B MTP

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp.20260525
cd llama.cpp.20260525

git log
	commit 314e729347defd9851e857f78084160c5786a7d8 (HEAD -> master, origin/master, origin/HEAD)
	Author: Tim Neumann <mail@timnn.me>
	Date:   Mon May 25 09:29:28 2026 +0200

		llama : document that only one on-device state can be saved per sequence (#23520)


# check cuda architecture
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
docker run --gpus all -it pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash
python3 -c "import torch; print(torch.cuda.get_device_capability())"
	(6, 1)
exit


# modify the cuda version to match the host's version
# set cuda version to 12.3.2 & ubuntu 22.04
# check docker image at
# https://hub.docker.com/r/nvidia/cuda
# and setup to use gcc12, so we need to patch for ubuntu 22.04
git diff 
	diff --git a/.devops/cuda.Dockerfile b/.devops/cuda.Dockerfile
	index 621fe8b6a..c9b43e70f 100644
	--- a/.devops/cuda.Dockerfile
	+++ b/.devops/cuda.Dockerfile
	@@ -1,6 +1,6 @@
	-ARG UBUNTU_VERSION=24.04
	+ARG UBUNTU_VERSION=22.04
	 # This needs to generally match the container host's environment.
	-ARG CUDA_VERSION=12.8.1
	+ARG CUDA_VERSION=12.3.2
	 # Target the CUDA build image
	 ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
	 
	@@ -15,10 +15,17 @@ FROM ${BASE_CUDA_DEV_CONTAINER} AS build
	 # CUDA architecture to build for (defaults to all supported archs)
	 ARG CUDA_DOCKER_ARCH=default
	 
	+# 1. install add-apt-repository tools
	 RUN apt-get update && \
	-    apt-get install -y gcc-14 g++-14 build-essential cmake python3 python3-pip git libssl-dev libgomp1
	+    apt-get install -y software-properties-common
	 
	-ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
	+# 2. add PPA source and install gcc12
	+RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
	+    apt-get update && \
	+    apt-get install -y gcc-12 g++-12 build-essential cmake python3 python3-pip git libssl-dev libgomp1
	+
	+# 3. setup gcc12 environment
	+ENV CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12
	 
	 WORKDIR /app


docker build --build-arg CUDA_DOCKER_ARCH=61 -t local/llama.cpp:server-cuda-12.3-20260527-mtp --target server -f .devops/cuda.Dockerfile .


mkdir -p unsloth/Qwen3.6-35B-A3B-MTP-GGUF

nohup hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF --include "Qwen3.6-35B-A3B-UD-IQ1_M.gguf" --local-dir unsloth/Qwen3.6-35B-A3B-MTP-GGUF &

nohup hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF --include "mmproj-F16.gguf" --local-dir unsloth/Qwen3.6-35B-A3B-MTP-GGUF &



###############################
#
#    --n-cpu-moe 41
#
###############################

docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260527-mtp \
  -m /models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/mmproj-F16.gguf \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 41 \
  --spec-type draft-mtp --spec-draft-n-max 2

# for mtp model
# --spec-type draft-mtp --spec-draft-n-max 2 \

# Reading: 21 tokens, 2.2s, 9.56 tokens/s
# Generation: 344 tokens, 45s, 7.49 t/s



###############################
#
#    --n-cpu-moe 5
#    due to CPU core is 6, remain one for system or GPU dispatch
###############################

docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260527-mtp \
  -m /models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/mmproj-F16.gguf \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 5 \
  --spec-type draft-mtp --spec-draft-n-max 2
  
# Reading: 23 tokens, 1.1s, 20.85 tokens/s
# Generation: 400 tokens, 1min, 6.61 t/s


###############################
#
#    KV cache q4_0
#
###############################
docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260527-mtp \
  -m /models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/mmproj-F16.gguf \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 5 \
  --spec-type draft-mtp --spec-draft-n-max 2

# Reading: 21 tokens, 1.0s, 20.39 tokens/s
# Generation: 277 tokens, 40s, 6.90 t/s
```

[huggingface Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)

## Test Qwen3.6 35B A3B

```bash
mkdir -p unsloth/Qwen3.6-35B-A3B-GGUF

nohup hf download unsloth/Qwen3.6-35B-A3B-GGUF --include "Qwen3.6-35B-A3B-UD-IQ1_M.gguf" --local-dir unsloth/Qwen3.6-35B-A3B-GGUF &

nohup hf download unsloth/Qwen3.6-35B-A3B-GGUF --include "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" --local-dir unsloth/Qwen3.6-35B-A3B-GGUF &

nohup hf download unsloth/Qwen3.6-35B-A3B-GGUF --include "mmproj-F16.gguf" --local-dir unsloth/Qwen3.6-35B-A3B-GGUF &

###############################
#
#    --n-cpu-moe 41
#
###############################

docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260527-mtp \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 41


###############################
#
#    --n-cpu-moe 5
#
###############################

docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260527-mtp \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 5


###############################
#
#    KV cache q4_0
#
###############################

docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260527-mtp \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 5
```

[huggingface Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)


## Run Qwen 3.6 35B A3B by llama.cpp Turbo Quant branch

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant

git checkout feature/turboquant-kv-cache

git log
  commit 2cbfdc62a1a047b01377948dfdede8cb6a744866 (HEAD -> feature/turboquant-kv-cache, tag: feature-turboquant-kv-cache-b9418-2cbfdc6, origin/feature/turboquant-kv-cache, origin/HEAD)
  Author: TheTom <tturney1@gmail.com>
  Date:   Tue May 19 08:26:31 2026 -0500

    docs(readme): credit Google's original TurboQuant + explain the '+'

# in .devops/cuda.Dockerfile
# modify Ubuntu to 22.04, CUDA 12.3.2, gcc12

docker build --build-arg CUDA_DOCKER_ARCH=61 -t local/llama.cpp:server-cuda-12.3-20260531-turboquant --target server -f .devops/cuda.Dockerfile .

###############################
#
#    --n-cpu-moe 41
#
###############################
docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260531-turboquant \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 41

###############################
#
#    --n-cpu-moe 5
#
###############################
docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260531-turboquant \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 5


###############################
#
#    --n-cpu-moe 41
#    KV cache turbo3
###############################
docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260531-turboquant \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 41

###############################
#
#    use Q4 model
#    
###############################
docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260531-turboquant \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --ctx-size 16384 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 41


###############################
#
#    --n-cpu-moe 41
#    KV cache turbo3, context 256K
###############################
docker run -d \
  --name llama-cpp-qwen3.6-35b-a3b \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v /mnt/d/workspace/models:/models \
  -p 11436:11436 \
  local/llama.cpp:server-cuda-12.3-20260531-turboquant \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  --mmproj /models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --ctx-size 262144 \
  --batch-size 512 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 11436 \
  --host 0.0.0.0 \
  --reasoning-budget 0 \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-mmap \
  --n-cpu-moe 41
```

## Device Info

* Windows WSL, Ubuntu 22.04
* CPU: Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz, 6 cores
* RAM: 24 GB, 22GB for WSL
* GPU: NVIDIA GeForce GTX 1050 Ti, 4GB VRAM
* CUDA 12.6


## Test Result

| Model Type| Parameters | TTFT | Tokens per second | Memory Usage |
| ---- | ---- | ---- | ---- | ---- |
| Qwen 3.6 35B A3B MTP | `--n-cpu-moe 41` <br> `--spec-type draft-mtp` <br> `--spec-draft-n-max 2` <br> `--cache-type-k q8_0 --cache-type-v q8_0` | 2.2s | 7.49 t/s | GPU VRAM: 3.569 GB, CPU RAM: 9.816 GB |
| Qwen 3.6 35B A3B MTP | `--n-cpu-moe 5` <br> `--spec-type draft-mtp` <br> `--spec-draft-n-max 2` <br> `--cache-type-k q8_0 --cache-type-v q8_0` | 1.1s | 6.61 t/s | GPU VRAM: 3.511 GB, CPU RAM: 1.87 GB |
| Qwen 3.6 35B A3B MTP | `--n-cpu-moe 5` <br> `--spec-type draft-mtp` <br> `--spec-draft-n-max 2` <br> `--cache-type-k q4_0 --cache-type-v q4_0` | 1.0s | 6.90 t/s | GPU VRAM: 3.546 GB, CPU RAM: 1.854 GB |
| Qwen 3.6 35B A3B | `--n-cpu-moe 41`<br> `--cache-type-k q8_0 --cache-type-v q8_0` | 0.9s | 11.75 t/s | GPU VRAM: 3.174 GB, CPU RAM: 8.606 GB |
| Qwen 3.6 35B A3B | `--n-cpu-moe 5`<br> `--cache-type-k q8_0 --cache-type-v q8_0` | 1.3s | 3.72 t/s | GPU VRAM: 3.566 GB, CPU RAM: 1.64 GB |
| Qwen 3.6 35B A3B | `--n-cpu-moe 5`<br> `--cache-type-k q4_0 --cache-type-v q4_0` | 1.3s | 3.67 t/s | GPU VRAM: 3.550 GB, CPU RAM: 1.651 GB |
| Qwen 3.6 35B A3B Turbo Quant | `--n-cpu-moe 41`<br> `--cache-type-k turbo4 --cache-type-v turbo4` | 0.8 s | 11.18 t/s | GPU VRAM: 3.298 GB, CPU RAM: 8.587 GB |
| Qwen 3.6 35B A3B Turbo Quant | `--n-cpu-moe 5`<br> `--cache-type-k turbo4 --cache-type-v turbo4` | 1.2 s | 3.55 t/s | GPU VRAM: 3.544 GB, CPU RAM: 1.752 GB |
| Qwen 3.6 35B A3B Turbo Quant | `--n-cpu-moe 41`<br> `--cache-type-k turbo3 --cache-type-v turbo3` <br> `--ctx-size 16384` | 0.8 s | 12.06 t/s | GPU VRAM: 3.290 GB, CPU RAM: 8.598 GB |
| Qwen 3.6 35B A3B Turbo Quant | `--n-cpu-moe 41`<br> `--cache-type-k turbo3 --cache-type-v turbo3` <br> `--ctx-size 262144` | 0.8 s | 10.85 t/s | GPU VRAM: 3.580 GB, CPU RAM: 8.638 GB |


* warning when using turbo3 for K
```
1.24.572.402 W llama_kv_cache: auto-asymmetric: GQA ratio 8:1 (n_head=16, n_head_kv=2) — upgrading K from turbo3 to q8_0 to prevent quality degradation. Disable with TURBO_AUTO_ASYMMETRIC=0
```

* ref:

1. [神人成功在 GTX 1060 6GB 上順跑 Qwen 3.6 35B A3B 模型，只需加入這五個參數](https://www.koc.com.tw/archives/642193)
2. [Running a 35B AI Model on 6GB VRAM, FAST (llama.cpp Guide)](https://www.youtube.com/watch?v=8F_5pdcD3HY)