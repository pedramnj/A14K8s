#!/bin/bash

echo "=========================================="
echo "VPC Resource Deep Check"
echo "=========================================="
echo ""

check_vpc_resources() {
    local region=$1
    local region_name=$2
    
    echo "=== $region_name ($region) ==="
    export AWS_DEFAULT_REGION=$region
    
    # All VPCs (including default)
    echo "All VPCs:"
    aws ec2 describe-vpcs --query 'Vpcs[*].[VpcId,CidrBlock,IsDefault]' --output table 2>/dev/null || echo "Error checking VPCs"
    echo ""
    
    # Internet Gateways
    echo "Internet Gateways:"
    igw_count=$(aws ec2 describe-internet-gateways --query 'length(InternetGateways)' --output text 2>/dev/null || echo "0")
    echo "Count: $igw_count"
    if [ "$igw_count" != "0" ]; then
        aws ec2 describe-internet-gateways --query 'InternetGateways[*].[InternetGatewayId,Attachments[0].VpcId,Attachments[0].State]' --output table 2>/dev/null
    fi
    echo ""
    
    # Subnets
    echo "Subnets:"
    subnet_count=$(aws ec2 describe-subnets --query 'length(Subnets)' --output text 2>/dev/null || echo "0")
    echo "Count: $subnet_count"
    if [ "$subnet_count" != "0" ]; then
        aws ec2 describe-subnets --query 'Subnets[*].[SubnetId,VpcId,AvailabilityZone]' --output table 2>/dev/null | head -20
    fi
    echo ""
    
    # Route Tables
    echo "Route Tables:"
    rt_count=$(aws ec2 describe-route-tables --query 'length(RouteTables)' --output text 2>/dev/null || echo "0")
    echo "Count: $rt_count"
    echo ""
    
    # Security Groups (non-default)
    echo "Security Groups (non-default):"
    sg_count=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=!default" --query 'length(SecurityGroups)' --output text 2>/dev/null || echo "0")
    echo "Count: $sg_count"
    echo ""
    
    # Network ACLs (non-default)
    echo "Network ACLs (non-default):"
    acl_count=$(aws ec2 describe-network-acls --filters "Name=default,Values=false" --query 'length(NetworkAcls)' --output text 2>/dev/null || echo "0")
    echo "Count: $acl_count"
    echo ""
    
    # VPC Endpoints
    echo "VPC Endpoints:"
    endpoint_count=$(aws ec2 describe-vpc-endpoints --query 'length(VpcEndpoints)' --output text 2>/dev/null || echo "0")
    echo "Count: $endpoint_count"
    echo ""
    
    # Elastic IPs (detailed)
    echo "Elastic IPs (all):"
    eip_count=$(aws ec2 describe-addresses --query 'length(Addresses)' --output text 2>/dev/null || echo "0")
    echo "Count: $eip_count"
    if [ "$eip_count" != "0" ]; then
        aws ec2 describe-addresses --query 'Addresses[*].[AllocationId,PublicIp,AssociationId,NetworkInterfaceId]' --output table 2>/dev/null
    fi
    echo ""
    
    echo "---"
    echo ""
}

check_vpc_resources "us-east-1" "US East (N. Virginia)"
check_vpc_resources "eu-central-1" "Europe (Frankfurt)"

echo "=========================================="
echo "VPC Resource Check Complete"
echo "=========================================="
