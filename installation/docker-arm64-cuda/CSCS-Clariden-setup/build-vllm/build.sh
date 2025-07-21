podman build --tag vllm:v0.9.0.1-apertus-vllm --file Dockerfile-apertus-vllm .
podman build --tag vllm:v0.9.0.1-base-vllm --file Dockerfile-base-vllm .

# To save for loading
podman save -o $SCRATCH/apertus-vllm.tar localhost/vllm:v0.9.0.1-apertus-vllm
podman save -o $SCRATCH/base-vllm.tar localhost/vllm:v0.9.0.1-base-vllm

# To save for enroot
enroot import -x mount podman://localhost/vllm:v0.9.0.1-apertus-vllm
enroot import -x mount podman://localhost/vllm:v0.9.0.1-base-vllm


# to load
podman load -i $SCRATCH/apertus-vllm.tar
podman load -i $SCRATCH/base-vllm.tar
