#!/bin/bash

echo "=========================================="
echo "AWS Resource Check - Complete Report"
echo "=========================================="
echo ""

check_region() {
    local region=$1
    local region_name=$2
    
    echo "=== $region_name ($region) ==="
    export AWS_DEFAULT_REGION=$region
    
    # Elastic IPs
    eip_count=$(aws ec2 describe-addresses --query 'length(Addresses)' --output text 2>/dev/null || echo "0")
    echo "Elastic IPs: $eip_count"
    
    # NAT Gateways (all states)
    nat_count=$(aws ec2 describe-nat-gateways --query 'length(NatGateways)' --output text 2>/dev/null || echo "0")
    echo "NAT Gateways: $nat_count"
    if [ "$nat_count" != "0" ]; then
        aws ec2 describe-nat-gateways --query 'NatGateways[*].[NatGatewayId,State]' --output table
    fi
    
    # EC2 Instances (all states)
    instance_count=$(aws ec2 describe-instances --query 'length(Reservations[*].Instances[*])' --output text 2>/dev/null || echo "0")
    echo "EC2 Instances: $instance_count"
    
    # VPCs (non-default)
    vpc_count=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=false" --query 'length(Vpcs)' --output text 2>/dev/null || echo "0")
    echo "Non-default VPCs: $vpc_count"
    
    # Unattached Volumes
    volume_count=$(aws ec2 describe-volumes --filters "Name=status,Values=available" --query 'length(Volumes)' --output text 2>/dev/null || echo "0")
    echo "Unattached Volumes: $volume_count"
    
    # Load Balancers
    lb_count=$(aws elbv2 describe-load-balancers --query 'length(LoadBalancers)' --output text 2>/dev/null || echo "0")
    echo "Load Balancers: $lb_count"
    
    # EKS Clusters
    eks_count=$(aws eks list-clusters --query 'length(clusters)' --output text 2>/dev/null || echo "0")
    echo "EKS Clusters: $eks_count"
    
    echo ""
}

check_region "us-east-1" "US East (N. Virginia)"
check_region "eu-central-1" "Europe (Frankfurt)"

echo "=========================================="
echo "Check Complete"
echo "=========================================="
