# FABRIC Slice Setup Guide

## ✅ Current Status

- **Slice Created**: ✅
- **Bastion Key**: ✅ `~/.ssh/fabric_bastion_key` (from FABRIC dashboard)
- **Slice Key**: ✅ `~/.ssh/fabric_key` (for VM access)

## Step 1: Get Slice Information

You need to get the following information from your FABRIC slice:

1. **Slice Name/ID**: From FABRIC portal
2. **VM IP Addresses**: From slice details
3. **FABRIC Username**: Your FABRIC portal username
4. **Bastion Host**: Usually `bastion-1.fabric-testbed.net` or similar

### Get Slice Info via FABlib (if installed):

```python
from fabrictestbed_extensions.fablib.fablib import FablibManager as fablib_manager

fablib = fablib_manager()
slices = fablib.list_slices()

# Find your slice
for slice in slices:
    print(f"Slice: {slice.get_name()}")
    print(f"Slice ID: {slice.get_slice_id()}")
    nodes = slice.get_nodes()
    for node in nodes:
        print(f"  Node: {node.get_name()}")
        print(f"    Management IP: {node.get_management_ip()}")
        print(f"    Site: {node.get_site()}")
```

### Or Check FABRIC Portal:

1. Go to: https://portal.fabric-testbed.net
2. Navigate to: **Slices** → Your slice
3. View slice details to get:
   - Node names
   - Management IP addresses
   - Site locations

## Step 2: Configure SSH

Add this to your `~/.ssh/config`:

```bash
# FABRIC Bastion Host
Host fabric-bastion
    HostName bastion-1.fabric-testbed.net
    User <your-fabric-username>
    IdentityFile ~/.ssh/fabric_bastion_key
    IdentitiesOnly yes
    ForwardAgent yes

# FABRIC Slice VMs (update with your actual IPs)
Host fabric-vm1
    HostName <vm1-management-ip>
    User <your-fabric-username>
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key
    IdentitiesOnly yes

Host fabric-vm2
    HostName <vm2-management-ip>
    User <your-fabric-username>
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key
    IdentitiesOnly yes

Host fabric-vm3
    HostName <vm3-management-ip>
    User <your-fabric-username>
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key
    IdentitiesOnly yes
```

## Step 3: Test Connection

```bash
# Test bastion connection
ssh fabric-bastion "echo 'Bastion connected!'"

# Test VM connection
ssh fabric-vm1 "echo 'VM1 connected!' && uname -a"
```

## Step 4: Check System Capabilities

Once connected, check what's available:

```bash
ssh fabric-vm1 << 'EOF'
echo "=== System Information ==="
uname -a
cat /etc/os-release | head -5

echo ""
echo "=== Python ==="
python3 --version
python3 -m pip --version

echo ""
echo "=== Docker ==="
docker --version

echo ""
echo "=== Disk Space ==="
df -h /

echo ""
echo "=== Memory ==="
free -h

echo ""
echo "=== CPU ==="
lscpu | head -10
EOF
```

## Step 5: Choose Deployment Path

### Option A: Deploy Kubernetes Cluster

If you want to set up Kubernetes on FABRIC:

1. **Install Kubernetes** on all VMs (see `fabric_install_k8s.sh`)
2. **Initialize cluster** on master node
3. **Join worker nodes**
4. **Get kubeconfig** for AI4K8s

### Option B: Deploy AI4K8s Directly

If you want to run AI4K8s directly on FABRIC VMs:

1. **Transfer project files** to VM
2. **Set up Python environment**
3. **Install dependencies**
4. **Run the application**

## Quick Commands

```bash
# List your slices (if FABlib installed)
python3 -c "from fabrictestbed_extensions.fablib.fablib import FablibManager as f; print([s.get_name() for s in f().list_slices()])"

# Connect to first VM
ssh fabric-vm1

# Copy files to VM
scp -o ProxyJump=fabric-bastion file.txt <username>@<vm-ip>:~/

# Run command on all VMs
for vm in fabric-vm1 fabric-vm2 fabric-vm3; do
    ssh $vm "echo 'Running on $vm' && hostname"
done
```

## Next Steps

1. **Get slice information** (IPs, usernames)
2. **Configure SSH** with actual values
3. **Test connections**
4. **Check system capabilities**
5. **Choose deployment path** (K8s or direct AI4K8s)

---

**Need Help?** Share your slice details and I'll help configure everything!


