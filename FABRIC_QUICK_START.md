# FABRIC Quick Start - Cheat Sheet

## 🔑 Essential Links

- **Portal**: https://portal.fabric-testbed.net
- **Jupyter Hub**: https://jupyter.fabric-testbed.net
- **Knowledge Base**: https://learn.fabric-testbed.net
- **Forums**: https://learn.fabric-testbed.net/knowledge-base/forum

## ⚡ Quick Commands

### 1. Create Kubernetes Cluster

```bash
# Install FABlib
pip3 install fabric-fablib

# Run setup script
python3 fabric_k8s_setup.py create
```

### 2. List Your Slices

```bash
python3 fabric_k8s_setup.py list
```

### 3. Delete a Slice

```bash
python3 fabric_k8s_setup.py delete ai4k8s-cluster
```

### 4. SSH to FABRIC VMs

```bash
# Configure SSH (add to ~/.ssh/config)
Host fabric-bastion
    HostName bastion-1.fabric-testbed.net
    User <your-username>
    IdentityFile ~/.ssh/fabric_bastion_key

Host k8s-*
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_slice_key
    User <your-username>

# Then SSH directly
ssh k8s-master
ssh k8s-worker1
ssh k8s-worker2
```

### 5. Install Kubernetes on VMs

```bash
# On master node
ssh k8s-master
bash fabric_install_k8s.sh master

# On each worker node
ssh k8s-worker1
bash fabric_install_k8s.sh worker
# Then run the join command from master

ssh k8s-worker2
bash fabric_install_k8s.sh worker
# Then run the join command from master
```

### 6. Get Kubeconfig

```bash
# On master node
cat ~/.kube/config

# Copy to HPC cluster
# On your local machine:
scp k8s-master:~/.kube/config ~/fabric-kubeconfig

# Then to HPC:
scp ~/fabric-kubeconfig pedramnj@hpcfund.amd.com:/home1/pedramnj/ai4k8s/kubeconfigs/
```

### 7. Add to AI4K8s Web App

1. Login to https://ai4k8s.online
2. Dashboard → Add Server
3. Name: "FABRIC K8s Cluster"
4. Kubeconfig Path: `/home1/pedramnj/ai4k8s/kubeconfigs/fabric-kubeconfig`

## 📋 Common Tasks

### Check Cluster Status

```bash
kubectl get nodes
kubectl get pods --all-namespaces
```

### Deploy Test Application

```bash
kubectl create deployment nginx --image=nginx
kubectl expose deployment nginx --port=80 --type=LoadBalancer
kubectl get services
```

### View Logs

```bash
kubectl logs <pod-name>
kubectl logs -f <pod-name>  # Follow logs
```

### Delete Everything

```bash
# Delete slice (removes all VMs)
python3 fabric_k8s_setup.py delete ai4k8s-cluster
```

## 🚨 Troubleshooting

### Cannot SSH
- Check bastion key: `ls -la ~/.ssh/fabric_bastion_key`
- Verify key permissions: `chmod 600 ~/.ssh/fabric_bastion_key`
- Check slice status in portal

### Kubernetes Not Working
- Verify all nodes: `kubectl get nodes`
- Check pods: `kubectl get pods --all-namespaces`
- Restart kubelet: `sudo systemctl restart kubelet`

### Slice Creation Fails
- Try different site (UTAH, GATECH, UCSD)
- Reduce VM size (cores/RAM)
- Check project permissions in portal

## 📚 Documentation

- Full guide: `FABRIC_GETTING_STARTED.md`
- Setup script: `fabric_k8s_setup.py`
- K8s install: `fabric_install_k8s.sh`

## ✅ Checklist

- [ ] FABRIC account created
- [ ] Project joined/created
- [ ] SSH keys generated
- [ ] API token obtained
- [ ] FABlib installed
- [ ] Slice created
- [ ] Kubernetes installed
- [ ] Cluster verified
- [ ] Kubeconfig downloaded
- [ ] Added to AI4K8s web app

---

**Need help?** Check FABRIC forums or knowledge base!

