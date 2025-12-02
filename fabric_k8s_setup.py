#!/usr/bin/env python3
"""
FABRIC Kubernetes Cluster Setup Script
Creates a Kubernetes cluster on FABRIC testbed for AI4K8s research project.

Prerequisites:
1. FABRIC account created and enrolled
2. FABlib installed: pip install fabric-fablib
3. API token obtained from portal
4. SSH keys generated (bastion and slice keys)
"""

from fabrictestbed_extensions.fablib.fablib import FablibManager as fablib_manager
import time
import sys

def create_k8s_cluster():
    """Create a 3-node Kubernetes cluster on FABRIC"""
    
    print("🚀 Starting FABRIC Kubernetes Cluster Creation...")
    print("=" * 60)
    
    # Initialize FABlib
    try:
        fablib = fablib_manager()
        print("✅ FABlib initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing FABlib: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed FABlib: pip install fabric-fablib")
        print("   2. Set up your API token")
        print("   3. Configured your credentials")
        return False
    
    # Create slice
    slice_name = "ai4k8s-cluster"
    print(f"\n📦 Creating slice: {slice_name}")
    
    try:
        # Check if slice already exists
        try:
            existing_slice = fablib.get_slice(name=slice_name)
            print(f"⚠️  Slice '{slice_name}' already exists!")
            response = input("Delete existing slice and create new one? (yes/no): ")
            if response.lower() == 'yes':
                print("🗑️  Deleting existing slice...")
                existing_slice.delete()
                time.sleep(5)
            else:
                print("❌ Aborted. Please delete the existing slice manually or use a different name.")
                return False
        except:
            pass  # Slice doesn't exist, continue
        
        # Create new slice
        slice = fablib.new_slice(name=slice_name)
        
        # Add control plane node (master)
        print("\n🖥️  Adding Kubernetes master node...")
        node1 = slice.add_node(
            name="k8s-master",
            site="TACC",  # You can change to UTAH, GATECH, UCSD, etc.
            cores=4,
            ram=16,  # 16GB RAM
            disk=50,  # 50GB disk
            image="default_ubuntu_22"  # Ubuntu 22.04
        )
        
        # Add worker nodes
        print("🖥️  Adding Kubernetes worker node 1...")
        node2 = slice.add_node(
            name="k8s-worker1",
            site="TACC",
            cores=4,
            ram=16,
            disk=50,
            image="default_ubuntu_22"
        )
        
        print("🖥️  Adding Kubernetes worker node 2...")
        node3 = slice.add_node(
            name="k8s-worker2",
            site="TACC",
            cores=4,
            ram=16,
            disk=50,
            image="default_ubuntu_22"
        )
        
        # Submit slice
        print("\n📤 Submitting slice to FABRIC...")
        slice.submit()
        
        print("⏳ Waiting for slice to be provisioned...")
        print("   (This may take 5-10 minutes)")
        
        # Wait for slice to be ready
        slice.wait_ssh(timeout=600)  # 10 minute timeout
        
        print("\n✅ Slice created successfully!")
        print("=" * 60)
        print(f"📋 Slice ID: {slice.get_slice_id()}")
        print(f"📋 Slice Name: {slice_name}")
        print("\n🖥️  Nodes:")
        
        # Get node information
        nodes = slice.get_nodes()
        for node in nodes:
            node_name = node.get_name()
            node_ip = node.get_management_ip()
            print(f"   - {node_name}: {node_ip}")
        
        print("\n" + "=" * 60)
        print("📝 Next Steps:")
        print("1. SSH into the master node:")
        print(f"   ssh -i ~/.ssh/fabric_slice_key -J <bastion> {nodes[0].get_management_ip()}")
        print("\n2. Install Kubernetes (see FABRIC_GETTING_STARTED.md)")
        print("\n3. Get kubeconfig and add to AI4K8s web app")
        print("\n4. Test the cluster connection")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating slice: {e}")
        print("\n💡 Common issues:")
        print("   - Insufficient project permissions")
        print("   - Site resources unavailable (try different site)")
        print("   - Network connectivity issues")
        return False


def list_slices():
    """List all your FABRIC slices"""
    try:
        fablib = fablib_manager()
        slices = fablib.list_slices()
        
        if not slices:
            print("📭 No slices found")
            return
        
        print(f"\n📦 Found {len(slices)} slice(s):")
        print("=" * 60)
        
        for slice in slices:
            print(f"\n📋 Name: {slice.get_name()}")
            print(f"   ID: {slice.get_slice_id()}")
            print(f"   State: {slice.get_state()}")
            
            nodes = slice.get_nodes()
            print(f"   Nodes: {len(nodes)}")
            for node in nodes:
                print(f"      - {node.get_name()}: {node.get_management_ip()}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error listing slices: {e}")


def delete_slice(slice_name):
    """Delete a FABRIC slice"""
    try:
        fablib = fablib_manager()
        slice = fablib.get_slice(name=slice_name)
        
        print(f"🗑️  Deleting slice: {slice_name}...")
        slice.delete()
        print("✅ Slice deleted successfully")
        
    except Exception as e:
        print(f"❌ Error deleting slice: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_slices()
        elif command == "delete" and len(sys.argv) > 2:
            delete_slice(sys.argv[2])
        elif command == "create":
            create_k8s_cluster()
        else:
            print("Usage:")
            print("  python3 fabric_k8s_setup.py create    - Create new K8s cluster")
            print("  python3 fabric_k8s_setup.py list       - List all slices")
            print("  python3 fabric_k8s_setup.py delete <name> - Delete a slice")
    else:
        # Default: create cluster
        create_k8s_cluster()

