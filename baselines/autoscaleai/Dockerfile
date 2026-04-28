# AutoScaleAI inference daemon (CPU-only, no GPU on the testbed).
#
# Built locally with buildah on the CrownLabs VM:
#   buildah bud -t localhost/autoscaleai:v1 baselines/autoscaleai/
#   buildah push localhost/autoscaleai:v1 oci-archive:/tmp/autoscaleai-v1.tar
#   sudo k3s ctr images import /tmp/autoscaleai-v1.tar
#
# The k8s manifest references `localhost/autoscaleai:v1` with imagePullPolicy: Never
# so containerd serves the locally imported image without a registry.

FROM python:3.12-slim AS base

# Avoid pip user-site warnings, ensure deterministic builds, no .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime dependencies. PyTorch CPU wheel is ~200 MB; we don't need GPU.
COPY requirements-runner.txt /app/requirements-runner.txt
RUN pip install -r /app/requirements-runner.txt

# Copy the vendored upstream code + our runner in one layer.
COPY agent/ /app/agent/
COPY collector/ /app/collector/
COPY env/ /app/env/
COPY executor/ /app/executor/
COPY runner.py /app/runner.py
# autoscale_agent.pt is the upstream pretrained checkpoint. The k8s deployment
# can override this via a ConfigMap mount at /app/autoscale_agent.pt if we
# later fine-tune on muBench.
COPY autoscale_agent.pt /app/autoscale_agent.pt

# Drop privileges. The runner only needs outbound HTTP to Prometheus + the
# K8s API; the ServiceAccount is what grants RBAC, not the OS user.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 ai4k8s \
    && chown -R ai4k8s:ai4k8s /app
USER 10001

# Healthcheck via the runner's HTTP endpoint.
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8081/healthz', timeout=2).status==200 else 1)"

ENTRYPOINT ["python", "/app/runner.py"]
