# FABRIC Testbed - Getting Started Guide for AI4K8s Research

## 🎯 Overview

FABRIC is a national research infrastructure that provides virtual machines, networking, and storage resources for research experiments. This guide will help you set up a Kubernetes cluster on FABRIC to replace the terminated AWS EKS cluster for your AI4K8s thesis project.

## 📋 Prerequisites

- University email address (Politecnico di Torino)
- SSH client installed
- Basic knowledge of Kubernetes and Linux

## 🚀 Step-by-Step Setup

### Step 1: Create FABRIC Account

1. **Visit the FABRIC Portal**: https://portal.fabric-testbed.net
2. **Sign Up**: Click "Sign Up" or "Enroll"
3. **Login Method**: Use your **Politecnico di Torino institutional identity** (recommended)
   - Alternative: Google, GitHub, ORCID, or Microsoft (requires extra verification)
4. **Complete Enrollment**: Follow the multi-step enrollment process
5. **Important**: Use the **same email address** for all FABRIC activities

### Step 2: Join or Create a Project

1. **Access Portal**: Login to https://portal.fabric-testbed.net
2. **Project Requirements**:
   - You need to be part of a project to create slices (experiments)
   - If your professor has a project, ask to be added
   - If not, your professor can create one (see "Creating a project" in portal)
3. **Project Permissions**: Each project has specific resource permissions
   - Project owners can request additional permissions if needed

### Step 3: Generate SSH Keys

FABRIC requires **two types of SSH keys**:

#### 3.1 Bastion Keys (for VM access)
1. **Portal → SSH Keys → Bastion Keys**
2. **Generate New Key**: Click "Generate" or "Add Key"
3. **Download Private Key**: Save `fabric_bastion_key` to `~/.ssh/`
4. **Set Permissions**:
   ```bash
   chmod 600 ~/.ssh/fabric_bastion_key
   ```
5. **Note**: Bastion keys expire after 6 months

#### 3.2 Slice/Sliver Keys (for VM login)

**✅ SSH Key Already Created!**

We've already created a key for you: `~/.ssh/fabric_key`

**Your Public Key** (register this in FABRIC portal):
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIXKo9iQKyJVH0k65MnMc5lDJWEkMtOyUnGbHi+ll7c7 pedram.nikjooy@studenti.polito.it
```

**To Register in FABRIC Portal:**
1. **Portal → SSH Keys → Slice Keys**
2. **Add Key**: Click "Add Key" or "Import SSH Key"
3. **Paste Public Key**: Copy the entire line above
4. **Save**: Give it a name (e.g., "MacBook Pro") and save
5. **Note**: Slice keys expire after 2 years

**Key Details:**
- **Type**: ED25519
- **Fingerprint**: `SHA256:tV0rONgEPmz+lj8F99jKH0o3dodt2EU3ObFjHJ2PRnw`
- **Location**: `~/.ssh/fabric_key` (private), `~/.ssh/fabric_key.pub` (public)
- **Permissions**: Already set correctly (600 for private, 644 for public)

### Step 4: Get API Token

1. **Portal → API Tokens**
2. **Generate Token**: Click "Generate New Token"
3. **Copy Token**: Save it securely (you'll need it for FABlib)
4. **Note**: Tokens can expire and need renewal

### Step 5: Install FABlib (Python Library)

FABlib is the recommended way to interact with FABRIC programmatically:

```bash
# Install FABlib
pip3 install fabric-fablib

# Or using pip
pip install fabric-fablib
```

**Alternative**: Use Jupyter Notebooks in FABRIC's Jupyter Hub (recommended for beginners)

### Step 6: Create Your First Slice (Kubernetes Cluster)

#### Option A: Using Jupyter Notebooks (Recommended for First Time)

1. **Access Jupyter Hub**: https://jupyter.fabric-testbed.net
2. **Login**: Use your FABRIC credentials
3. **Open Example Notebook**: 
   - Go to "Artifact Manager"
   - Search for "Kubernetes" or "K8s" examples
   - Or use a basic VM creation example
4. **Run Notebook**: Follow the example to create your slice

#### Option B: Using FABlib from Your Laptop

Create a Python script `create_k8s_cluster.py`:

```python
from fabrictestbed_extensions.fablib.fablib import FablibManager as fablib_manager

# Initialize FABlib
fablib = fablib_manager()

# Create a slice with 3 VMs for Kubernetes cluster
slice = fablib.new_slice(name="ai4k8s-cluster")

# Add control plane node (master)
node1 = slice.add_node(name="k8s-master", site="TACC", cores=4, ram=16, disk=50)

# Add worker nodes
node2 = slice.add_node(name="k8s-worker1", site="TACC", cores=4, ram=16, disk=50)
node3 = slice.add_node(name="k8s-worker2", site="TACC", cores=4, ram=16, disk=50)

# Submit slice
slice.submit()

# Wait for slice to be ready
slice.wait_ssh(timeout=300)

print("Slice created successfully!")
print(f"Slice ID: {slice.get_slice_id()}")
```

**Run the script**:
```bash
python3 create_k8s_cluster.py
```

#### Option C: Using Portal (Visual Topology Builder)

1. **Portal → Slices → Create Slice**
2. **Add Nodes**: 
   - Click "Add Node"
   - Select site (e.g., TACC, UTAH, GATECH)
   - Configure: 4 cores, 16GB RAM, 50GB disk
   - Add 3 nodes (1 master + 2 workers)
3. **Add Network**: Connect nodes with Layer 2 network
4. **Submit Slice**: Click "Submit"

### Step 7: Install Kubernetes on FABRIC VMs

Once your slice is ready, SSH into each VM and install Kubernetes:

#### 7.1 SSH Configuration

FABRIC uses bastion hosts. Configure SSH:

```bash
# Add to ~/.ssh/config
Host fabric-bastion
    HostName bastion-1.fabric-testbed.net
    User <your-fabric-username>
    IdentityFile ~/.ssh/fabric_bastion_key
    ForwardAgent yes

Host k8s-master
    HostName <master-vm-ip>
    User <your-fabric-username>
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key

Host k8s-worker1
    HostName <worker1-vm-ip>
    User <your-fabric-username>
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key

Host k8s-worker2
    HostName <worker2-vm-ip>
    User <your-fabric-username>
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key
```

#### 7.2 Install Kubernetes (on each VM)

```bash
# SSH into master node
ssh k8s-master

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

# Install Kubernetes tools
sudo apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# Initialize Kubernetes cluster (MASTER ONLY)
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# Setup kubeconfig (MASTER ONLY)
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Install network plugin (MASTER ONLY)
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml

# Get join command (MASTER ONLY)
kubeadm token create --print-join-command
# Copy the output command
```

#### 7.3 Join Worker Nodes

```bash
# On each worker node
ssh k8s-worker1
# Run the join command from master (with sudo)
sudo kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash <hash>

# Repeat for worker2
ssh k8s-worker2
sudo kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash <hash>
```

#### 7.4 Verify Cluster

```bash
# On master node
kubectl get nodes
# Should show all 3 nodes as Ready

kubectl get pods --all-namespaces
# Should show all system pods running
```

### Step 8: Download Kubeconfig for AI4K8s

```bash
# On master node
cat ~/.kube/config

# Copy the entire output and save to a file on your HPC cluster:
# /home1/pedramnj/ai4k8s/kubeconfigs/fabric-k8s-config
```

### Step 9: Add FABRIC Cluster to AI4K8s Web App

1. **SSH to HPC**: 
   ```bash
   ssh pedramnj@hpcfund.amd.com
   ```

2. **Save Kubeconfig**:
   ```bash
   cd /home1/pedramnj/ai4k8s
   mkdir -p kubeconfigs
   nano kubeconfigs/fabric-k8s-config
   # Paste the kubeconfig content
   ```

3. **Add Server in Web App**:
   - Login to https://ai4k8s.online
   - Go to Dashboard → Add Server
   - **Name**: "FABRIC K8s Cluster"
   - **Connection String**: Leave empty or use a placeholder
   - **Kubeconfig Path**: `/home1/pedramnj/ai4k8s/kubeconfigs/fabric-k8s-config`
   - **Status**: Active

4. **Test Connection**:
   - Go to Dashboard
   - Click on "FABRIC K8s Cluster"
   - Verify it shows as "Active" (green badge)
   - Test AI Chat and Monitoring features

## 🔧 FABRIC-Specific Considerations

### Management vs Dataplane

- **Management Plane**: 
  - Used for SSH, software downloads
  - Rate-limited (~1Gbps)
  - **Do NOT use for experiment data flows**
  - IPv6 addresses (bastion handles IPv4/IPv6 translation)

- **Dataplane**:
  - High-bandwidth (100Gbps on many links)
  - For experiment traffic
  - Connect VMs with Layer 2/3 networks
  - Connect to external resources (HPC, cloud, Internet)

### Site Selection

Popular FABRIC sites:
- **TACC** (Texas Advanced Computing Center)
- **UTAH** (University of Utah)
- **GATECH** (Georgia Tech)
- **UCSD** (UC San Diego)

Choose sites close to each other for better network performance.

### Resource Limits

- **VM Sizes**: Up to 64 cores, 384GB RAM per VM
- **Storage**: Large volumes can be attached
- **Network**: 100Gbps links available
- **Slice Lifetime**: Finite, but can be extended

### Slice Management

```python
# List your slices
slices = fablib.list_slices()

# Get slice details
slice = fablib.get_slice(name="ai4k8s-cluster")
print(slice.get_slice_id())
print(slice.get_nodes())

# Extend slice lifetime
slice.renew()

# Delete slice (when done)
slice.delete()
```

## 📚 Additional Resources

### FABRIC Documentation
- **Knowledge Base**: https://learn.fabric-testbed.net
- **Portal**: https://portal.fabric-testbed.net
- **Jupyter Hub**: https://jupyter.fabric-testbed.net
- **Forums**: https://learn.fabric-testbed.net/knowledge-base/forum

### Example Notebooks
- **Artifact Manager**: Access via Jupyter Hub
- Search for: "Kubernetes", "K8s", "Multi-node", "Networking"

### Getting Help
- **Forums**: Post questions on FABRIC community forums
- **Search Knowledge Base**: Use search function
- **Contact Support**: Via portal or forums

## 🎓 For Your Thesis

### Advantages of FABRIC over AWS

1. **Free for Research**: No billing concerns
2. **High-Performance**: 100Gbps networking, powerful VMs
3. **Research-Focused**: Designed for academic experiments
4. **Reproducible**: Can share slice configurations
5. **No Cost Surprises**: Predictable resource limits

### Integration with AI4K8s

Your existing AI4K8s platform will work seamlessly with FABRIC:
- Same Kubernetes API
- Same kubectl commands
- Same monitoring and autoscaling features
- Just update the kubeconfig path

### Documentation for Thesis

You can document:
- FABRIC as the research infrastructure
- Comparison: AWS (commercial) vs FABRIC (research)
- Cost analysis: $238 AWS vs $0 FABRIC
- Performance characteristics of FABRIC networking

## ✅ Checklist

- [ ] Created FABRIC account with university email
- [ ] Joined or created a project
- [ ] Generated bastion SSH key
- [ ] Generated slice SSH key
- [ ] Obtained API token
- [ ] Installed FABlib (or accessed Jupyter Hub)
- [ ] Created first slice with 3 VMs
- [ ] Installed Kubernetes on all nodes
- [ ] Joined worker nodes to cluster
- [ ] Verified cluster is working (`kubectl get nodes`)
- [ ] Downloaded kubeconfig
- [ ] Added cluster to AI4K8s web app
- [ ] Tested monitoring and chat features

## 🚨 Troubleshooting

### Cannot SSH to VMs
- Check bastion key is correct and not expired
- Verify slice key is registered in portal
- Check SSH config has correct ProxyJump

### Kubernetes nodes not joining
- Verify network connectivity between nodes
- Check firewall rules (FABRIC may have restrictions)
- Ensure all nodes can reach master on port 6443

### Slice creation fails
- Check project has sufficient resource permissions
- Try a different site (some sites may be full)
- Reduce VM size (cores/RAM) if hitting limits

### API token expired
- Generate new token in portal
- Update FABlib configuration

## 📝 Next Steps

1. **Deploy Test Application**: Deploy nginx or redis to test autoscaling
2. **Enable Monitoring**: Install metrics-server and test monitoring
3. **Test Autoscaling**: Create HPA and test predictive autoscaling
4. **Document Results**: Update thesis with FABRIC infrastructure details

---

**Good luck with your FABRIC setup!** 🚀

If you encounter issues, check the FABRIC forums or knowledge base first, as many common questions are already answered there.

