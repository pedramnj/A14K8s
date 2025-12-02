#!/bin/bash
# Kubernetes Installation Script for FABRIC VMs
# Run this script on each VM (master and workers)
# Usage: bash fabric_install_k8s.sh [master|worker]

set -e  # Exit on error

NODE_TYPE=${1:-worker}  # Default to worker if not specified

echo "🚀 Installing Kubernetes on FABRIC VM (Node Type: $NODE_TYPE)"
echo "============================================================"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

# Add current user to docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Install Kubernetes tools
echo "☸️  Installing Kubernetes tools..."
sudo apt-get install -y apt-transport-https curl

# Add Kubernetes repository
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list

# Install kubelet, kubeadm, kubectl
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# Configure sysctl for networking
echo "🔧 Configuring networking..."
sudo modprobe br_netfilter
echo '1' | sudo tee /proc/sys/net/bridge/bridge-nf-call-iptables
echo '1' | sudo tee /proc/sys/net/ipv4/ip_forward

# Make sysctl changes persistent
sudo tee /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables = 1
net.ipv4.ip_forward = 1
EOF

sudo sysctl --system

if [ "$NODE_TYPE" = "master" ]; then
    echo ""
    echo "👑 Initializing Kubernetes master node..."
    echo "============================================================"
    
    # Initialize cluster
    sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --ignore-preflight-errors=Swap
    
    # Setup kubeconfig for current user
    mkdir -p $HOME/.kube
    sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
    sudo chown $(id -u):$(id -g) $HOME/.kube/config
    
    # Install Flannel network plugin
    echo "🌐 Installing Flannel network plugin..."
    kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
    
    # Wait for network plugin to be ready
    echo "⏳ Waiting for network plugin to be ready..."
    sleep 30
    
    # Get join command
    echo ""
    echo "✅ Master node initialized successfully!"
    echo "============================================================"
    echo ""
    echo "📋 Join Command (run this on worker nodes):"
    echo "--------------------------------------------"
    JOIN_CMD=$(kubeadm token create --print-join-command 2>/dev/null)
    echo "$JOIN_CMD"
    echo ""
    echo "💾 Save this command to join worker nodes"
    echo ""
    echo "📋 Kubeconfig location: $HOME/.kube/config"
    echo "   Copy this file to your HPC cluster for AI4K8s integration"
    echo ""
    
    # Show cluster status
    echo "📊 Cluster Status:"
    kubectl get nodes
    kubectl get pods --all-namespaces
    
else
    echo ""
    echo "👷 Worker node setup complete!"
    echo "============================================================"
    echo ""
    echo "📝 To join this node to the cluster:"
    echo "   1. Get the join command from the master node"
    echo "   2. Run: sudo <join-command>"
    echo ""
    echo "   Example:"
    echo "   sudo kubeadm join <master-ip>:6443 --token <token> \\"
    echo "     --discovery-token-ca-cert-hash sha256:<hash>"
    echo ""
fi

echo ""
echo "✅ Kubernetes installation complete!"
echo "============================================================"

