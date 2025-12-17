#!/usr/bin/env python3
"""Test API response directly from HPC cluster by calling the Python functions"""

import sys
import os
import json
import sqlite3

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from autoscaling_integration import AutoscalingIntegration
    
    deployment_name = "test-app-autoscaling"
    namespace = "ai4k8s-test"
    server_id = 3  # Same as web UI
    
    print("=" * 80)
    print(f"Testing API response for: {deployment_name} in {namespace}")
    print(f"Using server_id: {server_id}")
    print("=" * 80)
    
    # Get kubeconfig path from database (same as web UI does)
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'ai4k8s.db')
    kubeconfig_path = None
    
    if os.path.exists(db_path):
        print(f"Reading kubeconfig from database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First, check what columns exist
        cursor.execute("PRAGMA table_info(server)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"Server table columns: {columns}")
        
        # Try to get kubeconfig (stored as blob) and determine path
        cursor.execute("SELECT kubeconfig FROM server WHERE id = ?", (server_id,))
        result = cursor.fetchone()
        if result and result[0]:
            # kubeconfig is stored as blob, need to write to temp file
            import tempfile
            kubeconfig_data = result[0]
            if isinstance(kubeconfig_data, bytes):
                kubeconfig_data = kubeconfig_data.decode('utf-8')
            
            # Write to temp file (same as web UI does)
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
            temp_file.write(kubeconfig_data)
            temp_file.close()
            kubeconfig_path = temp_file.name
            print(f"✅ Found kubeconfig in database, wrote to temp file: {kubeconfig_path}")
        else:
            # Try well-known paths (same as web UI)
            candidate_paths = [
                '/app/instance/kubeconfig_admin',
                os.path.expanduser('~/.kube/config'),
                '/home1/pedramnj/ai4k8s/instance/kubeconfig_admin',
            ]
            for path in candidate_paths:
                if os.path.exists(path):
                    kubeconfig_path = path
                    print(f"✅ Found kubeconfig at: {kubeconfig_path}")
                    break
            if not kubeconfig_path:
                print("⚠️  No kubeconfig found, using default")
        conn.close()
    else:
        print(f"⚠️  Database not found at {db_path}, trying default paths")
        candidate_paths = [
            os.path.expanduser('~/.kube/config'),
            '/home1/pedramnj/ai4k8s/instance/kubeconfig_admin',
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                kubeconfig_path = path
                print(f"✅ Found kubeconfig at: {kubeconfig_path}")
                break
    
    # Initialize autoscaling integration with kubeconfig
    print(f"Initializing AutoscalingIntegration with kubeconfig: {kubeconfig_path or 'default'}...")
    autoscaling = AutoscalingIntegration(kubeconfig_path=kubeconfig_path)
    
    if not autoscaling:
        print("❌ Failed to initialize AutoscalingIntegration")
        sys.exit(1)
    
    print("✅ AutoscalingIntegration initialized")
    print("=" * 80)
    
    # First, call get_scaling_recommendation directly to see what it returns
    print(f"\n1. Calling get_scaling_recommendation directly...")
    direct_result = autoscaling.predictive_autoscaler.get_scaling_recommendation(deployment_name, namespace)
    print(f"   Direct result success: {direct_result.get('success')}")
    if not direct_result.get('success'):
        print(f"   Direct result error: {direct_result.get('error')}")
        print(f"   Full direct result: {json.dumps(direct_result, indent=2, default=str)}")
    else:
        rec = direct_result.get('recommendation', {})
        print(f"   Direct result recommendation target_replicas: {rec.get('target_replicas')}")
        print(f"   Direct result recommendation action: {rec.get('action')}")
    
    # Then call get_scaling_recommendations (same as API endpoint)
    print(f"\n2. Calling get_scaling_recommendations (same as API endpoint)...")
    result = autoscaling.get_scaling_recommendations(deployment_name, namespace)
    
    print("\n" + "=" * 80)
    print("FULL API RESPONSE (same format as web API):")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    
    # Check predictive recommendation
    if result.get('recommendations', {}).get('predictive'):
        pred = result['recommendations']['predictive']
        print("\n" + "=" * 80)
        print("PREDICTIVE RECOMMENDATION:")
        print("=" * 80)
        print(json.dumps(pred, indent=2, default=str))
        
        if pred.get('success') and pred.get('recommendation'):
            rec = pred['recommendation']
            target_replicas = rec.get('target_replicas')
            action = rec.get('action')
            scaling_type = rec.get('scaling_type')
            
            print("\n" + "=" * 80)
            print("KEY FIELDS:")
            print("=" * 80)
            print(f"  target_replicas: {target_replicas}")
            print(f"  action: {action}")
            print(f"  scaling_type: {scaling_type}")
            print(f"  current_replicas: {rec.get('current_replicas')}")
            print("=" * 80)
            
            # Check if target_replicas is within expected bounds
            if target_replicas is not None:
                # Try to get max_replicas from the recommendation or deployment
                max_replicas = rec.get('max_replicas')
                if not max_replicas:
                    # Try to get from deployment annotation
                    try:
                        deployment_status = autoscaling.hpa_manager.get_deployment_replicas(deployment_name, namespace)
                        if deployment_status.get('success'):
                            annotations = deployment_status.get('annotations', {})
                            config_str = annotations.get('ai4k8s.io/predictive-autoscaling-config', '{}')
                            config = json.loads(config_str) if config_str else {}
                            max_replicas = config.get('max_replicas')
                            min_replicas = config.get('min_replicas', 2)
                            
                            print(f"\n  Deployment Config: min={min_replicas}, max={max_replicas}")
                            
                            if max_replicas and target_replicas > max_replicas:
                                print(f"\n❌ ERROR: target_replicas={target_replicas} exceeds max_replicas={max_replicas}")
                            elif target_replicas > 10:
                                print(f"\n⚠️  WARNING: target_replicas={target_replicas} might exceed expected max (max_replicas not found)")
                            else:
                                print(f"\n✅ OK: target_replicas={target_replicas} is within bounds")
                    except Exception as e:
                        print(f"\n⚠️  Could not check deployment config: {e}")
            else:
                print("\n⚠️  WARNING: target_replicas is None")
        else:
            print("\n⚠️  No recommendation in predictive response")
            if pred.get('error'):
                print(f"Error: {pred['error']}")
    else:
        print("\n⚠️  No predictive recommendation in response")
        print("Response structure:", list(result.keys()))
    
    print("\n" + "=" * 80)
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the ai4k8s directory")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

