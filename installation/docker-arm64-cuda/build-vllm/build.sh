podman build --tag vllm:v0.9.0.1-apertus-vllm-v2 --file Dockerfile-apertus-vllm .
podman build --tag vllm:v0.9.0.1-base-vllm-v2 --file Dockerfile-base-vllm .

# To save for loading
podman save -o $SCRATCH/apertus-vllm-v2.tar localhost/vllm:v0.9.0.1-apertus-vllm-v2
podman save -o $SCRATCH/base-vllm0-v2.tar localhost/vllm:v0.9.0.1-base-vllm-v2

# To save for enroot
enroot import -x mount podman://localhost/vllm:v0.9.0.1-apertus-vllm-v2
enroot import -x mount podman://localhost/vllm:v0.9.0.1-base-vllm-v2


# to load
podman load -i $SCRATCH/apertus-vllm-v2.tar
podman load -i $SCRATCH/base-vllm-v2.tar
