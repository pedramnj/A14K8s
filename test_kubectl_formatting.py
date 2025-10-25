#!/usr/bin/env python3
"""
Test script to demonstrate the improved kubectl response formatting
"""

from simple_kubectl_executor import SimpleKubectlExecutor

def test_kubectl_formatting():
    """Test the new kubectl response formatting"""
    executor = SimpleKubectlExecutor()
    
    # Sample kubectl outputs to test formatting
    test_cases = [
        {
            'command': 'get pods --all-namespaces',
            'output': '''NAMESPACE     NAME                                    READY   STATUS    RESTARTS   AGE
default       nginx                                   1/1     Running   0          2d22h
kube-system   coredns-5d78c9869d-5bnpr               1/1     Running   0          2d22h
kube-system   coredns-5d78c9869d-5bqjb               1/1     Running   0          2d22h
kube-system   etcd-ai4k8s-control-plane              1/1     Running   0          2d22h
web           mcp-bridge-756789bdc-s98jl              1/1     Running   0          2d22h'''
        },
        {
            'command': 'top pods',
            'output': '''NAME                                    CPU(cores)   MEMORY(bytes)
nginx                                   1m           2Mi
mcp-bridge-756789bdc-s98jl              5m           10Mi
coredns-5d78c9869d-5bnpr                2m           5Mi'''
        },
        {
            'command': 'get nodes',
            'output': '''NAME                    STATUS   ROLES           AGE   VERSION
ai4k8s-control-plane   Ready    control-plane   2d22h   v1.28.0'''
        },
        {
            'command': 'get events --all-namespaces',
            'output': '''NAMESPACE   LAST SEEN   TYPE      REASON      OBJECT              MESSAGE
default     2d22h       Normal    Scheduled   pod/nginx           Successfully assigned default/nginx to ai4k8s-control-plane
kube-system 2d22h       Normal    Created     pod/coredns         Created container coredns'''
        },
        {
            'command': 'get namespaces',
            'output': '''NAME                 STATUS   AGE
default              Active   2d22h
kube-system          Active   2d22h
kube-public          Active   2d22h
kube-node-lease      Active   2d22h
local-path-storage   Active   2d22h
web                  Active   2d22h'''
        },
        {
            'command': 'get services',
            'output': '''NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   2d22h
nginx        ClusterIP   10.96.0.100  <none>        80/TCP    2d22h'''
        }
    ]
    
    print("🧪 Testing Improved Kubectl Response Formatting\n")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['command']}")
        print("-" * 40)
        
        # Format the output using our new formatter
        formatted_output = executor._format_kubectl_output(test_case['output'], test_case['command'])
        
        # Save to HTML file for preview
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Kubectl Formatting Test - {test_case['command']}</title>
    <link rel="stylesheet" href="static/css/style.css">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 2rem;
            background: #f8fafc;
        }}
        .test-container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        .test-title {{
            background: #1e293b;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem 0.5rem 0 0;
            margin: 0;
        }}
        .test-content {{
            background: white;
            padding: 1rem;
            border: 1px solid #e2e8f0;
            border-top: none;
            border-radius: 0 0 0.5rem 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="test-container">
        <h2 class="test-title">Command: {test_case['command']}</h2>
        <div class="test-content">
            {formatted_output}
        </div>
    </div>
</body>
</html>
"""
        
        filename = f"test_output_{i}_{test_case['command'].replace(' ', '_').replace('--', '')}.html"
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Formatted output saved to: {filename}")
        print(f"📊 Raw output length: {len(test_case['output'])} chars")
        print(f"🎨 Formatted output length: {len(formatted_output)} chars")
        print(f"🔍 Preview: {formatted_output[:100]}...")
    
    print(f"\n🎉 All {len(test_cases)} test cases completed!")
    print("📁 Check the generated HTML files to see the beautiful formatting")

if __name__ == "__main__":
    test_kubectl_formatting()
