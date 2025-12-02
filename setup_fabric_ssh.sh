#!/bin/bash

# FABRIC SSH Configuration Setup Script

echo "=== FABRIC SSH Configuration Setup ==="
echo ""

# Check if keys exist
if [ ! -f ~/.ssh/fabric_bastion_key ]; then
    echo "❌ Bastion key not found: ~/.ssh/fabric_bastion_key"
    exit 1
fi

if [ ! -f ~/.ssh/fabric_key ]; then
    echo "❌ Slice key not found: ~/.ssh/fabric_key"
    exit 1
fi

echo "✅ Keys found"
echo ""

# Get user input
read -p "Enter your FABRIC username: " FABRIC_USER
read -p "Enter bastion hostname (default: bastion-1.fabric-testbed.net): " BASTION_HOST
BASTION_HOST=${BASTION_HOST:-bastion-1.fabric-testbed.net}

echo ""
echo "Enter VM information (press Enter to skip):"
read -p "VM1 name (e.g., fabric-vm1): " VM1_NAME
read -p "VM1 IP address: " VM1_IP

read -p "VM2 name (e.g., fabric-vm2): " VM2_NAME
read -p "VM2 IP address: " VM2_IP

read -p "VM3 name (e.g., fabric-vm3): " VM3_NAME
read -p "VM3 IP address: " VM3_IP

# Create SSH config
cat >> ~/.ssh/config << CONFIG

# FABRIC Configuration
Host fabric-bastion
    HostName $BASTION_HOST
    User $FABRIC_USER
    IdentityFile ~/.ssh/fabric_bastion_key
    IdentitiesOnly yes
    ForwardAgent yes

CONFIG

# Add VMs if provided
if [ -n "$VM1_IP" ]; then
    cat >> ~/.ssh/config << CONFIG
Host $VM1_NAME
    HostName $VM1_IP
    User $FABRIC_USER
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key
    IdentitiesOnly yes

CONFIG
fi

if [ -n "$VM2_IP" ]; then
    cat >> ~/.ssh/config << CONFIG
Host $VM2_NAME
    HostName $VM2_IP
    User $FABRIC_USER
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key
    IdentitiesOnly yes

CONFIG
fi

if [ -n "$VM3_IP" ]; then
    cat >> ~/.ssh/config << CONFIG
Host $VM3_NAME
    HostName $VM3_IP
    User $FABRIC_USER
    ProxyJump fabric-bastion
    IdentityFile ~/.ssh/fabric_key
    IdentitiesOnly yes

CONFIG
fi

echo ""
echo "✅ SSH config updated!"
echo ""
echo "Test connections with:"
if [ -n "$VM1_IP" ]; then
    echo "  ssh $VM1_NAME"
fi
if [ -n "$VM2_IP" ]; then
    echo "  ssh $VM2_NAME"
fi
if [ -n "$VM3_IP" ]; then
    echo "  ssh $VM3_NAME"
fi
