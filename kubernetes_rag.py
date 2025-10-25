#!/usr/bin/env python3
"""
Kubernetes RAG (Retrieval-Augmented Generation) System
Lightweight, CPU-only, works with any LLM
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class KubernetesRAG:
    """
    Lightweight RAG system for Kubernetes manifest generation
    No GPU required - uses simple BM25 + keyword matching
    """
    
    def __init__(self, knowledge_base_path: str = "kb_kubernetes"):
        self.kb_path = knowledge_base_path
        self.knowledge_base = {}
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize with curated Kubernetes knowledge"""
        
        # Create knowledge base directory
        os.makedirs(self.kb_path, exist_ok=True)
        
        # Load or create knowledge base
        kb_file = os.path.join(self.kb_path, "knowledge.json")
        if os.path.exists(kb_file):
            with open(kb_file, 'r') as f:
                self.knowledge_base = json.load(f)
            print(f"✅ Loaded knowledge base: {len(self.knowledge_base)} entries")
        else:
            self.knowledge_base = self._create_default_knowledge()
            self._save_knowledge_base()
            print(f"✅ Created default knowledge base: {len(self.knowledge_base)} entries")
    
    def _create_default_knowledge(self) -> Dict[str, Any]:
        """Create default Kubernetes knowledge base"""
        
        return {
            # Deployment Templates
            "deployment_basic": {
                "type": "template",
                "category": "deployment",
                "keywords": ["deployment", "create", "app", "container"],
                "content": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{name}}
  namespace: {{namespace}}
  labels:
    app: {{name}}
    managed-by: ai4k8s
spec:
  replicas: {{replicas}}
  selector:
    matchLabels:
      app: {{name}}
  template:
    metadata:
      labels:
        app: {{name}}
    spec:
      containers:
      - name: {{name}}
        image: {{image}}
        ports:
        - containerPort: {{port}}
          name: http
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /healthz
            port: {{port}}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {{port}}
          initialDelaySeconds: 5
          periodSeconds: 5""",
                "description": "Production-ready deployment with health probes and resource limits",
                "best_practices": [
                    "Always define resource requests and limits",
                    "Include liveness and readiness probes",
                    "Use meaningful labels",
                    "Set initialDelaySeconds appropriately"
                ]
            },
            
            # Service Templates
            "service_clusterip": {
                "type": "template",
                "category": "service",
                "keywords": ["service", "expose", "internal", "clusterip"],
                "content": """apiVersion: v1
kind: Service
metadata:
  name: {{name}}
  namespace: {{namespace}}
  labels:
    app: {{name}}
spec:
  type: ClusterIP
  selector:
    app: {{name}}
  ports:
  - port: {{port}}
    targetPort: {{target_port}}
    protocol: TCP
    name: http""",
                "description": "Internal service for cluster-only access",
                "best_practices": [
                    "ClusterIP for internal services",
                    "LoadBalancer for external access",
                    "Use consistent naming"
                ]
            },
            
            # HPA Templates
            "hpa_standard": {
                "type": "template",
                "category": "autoscaling",
                "keywords": ["autoscale", "hpa", "scale", "horizontal"],
                "content": """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{name}}-hpa
  namespace: {{namespace}}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{name}}
  minReplicas: {{min_replicas}}
  maxReplicas: {{max_replicas}}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 15
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max""",
                "description": "Horizontal Pod Autoscaler with CPU and memory metrics",
                "best_practices": [
                    "Set stabilization windows to prevent flapping",
                    "Use conservative scaleDown (300s)",
                    "Aggressive scaleUp (0s) for responsiveness",
                    "Monitor metrics server availability"
                ]
            },
            
            # Best Practices
            "resources_practice": {
                "type": "best_practice",
                "category": "resources",
                "keywords": ["resources", "cpu", "memory", "limits", "requests"],
                "content": """Resource Management Best Practices:

1. Always define requests and limits:
   - requests: Guaranteed resources (scheduling)
   - limits: Maximum allowed (prevent noisy neighbors)

2. CPU:
   - request = expected average usage
   - limit = burst capacity (2-3x request)
   - CPU is throttled, not OOM

3. Memory:
   - request = limit (avoid OOM kills)
   - Memory cannot be throttled
   - OOM = pod killed

4. Example:
   resources:
     requests:
       cpu: 100m      # 0.1 CPU core
       memory: 128Mi
     limits:
       cpu: 500m      # 0.5 CPU core burst
       memory: 128Mi  # Same as request""",
                "description": "How to properly set resource requests and limits"
            },
            
            "probes_practice": {
                "type": "best_practice",
                "category": "health",
                "keywords": ["probe", "health", "liveness", "readiness", "startup"],
                "content": """Health Probes Best Practices:

1. Liveness Probe:
   - Checks if pod is alive
   - Failed = restart pod
   - Use for deadlock detection
   - initialDelaySeconds: 30-60s

2. Readiness Probe:
   - Checks if pod can serve traffic
   - Failed = remove from service
   - Use for dependency checks
   - initialDelaySeconds: 5-10s

3. Startup Probe:
   - For slow-starting containers
   - Disables other probes until succeeds
   - initialDelaySeconds: 0
   - failureThreshold: 30

4. Example:
   livenessProbe:
     httpGet:
       path: /healthz
       port: 8080
     initialDelaySeconds: 30
   readinessProbe:
     httpGet:
       path: /ready
       port: 8080
     initialDelaySeconds: 5""",
                "description": "How to properly configure health probes"
            },
            
            "autoscaling_practice": {
                "type": "best_practice",
                "category": "autoscaling",
                "keywords": ["autoscaling", "hpa", "scaling", "metrics"],
                "content": """Autoscaling Best Practices:

1. HPA Metrics:
   - CPU: Good for compute-bound apps
   - Memory: Good for memory-bound apps
   - Custom: Use for application-specific metrics

2. Scaling Behavior:
   - scaleDown stabilization: 300s (prevent flapping)
   - scaleUp stabilization: 0s (fast response)
   - Use policies to control rate

3. Replica Settings:
   - minReplicas: At least 2 for HA
   - maxReplicas: Based on cluster capacity
   - Target: 70-80% utilization

4. Common Issues:
   - Missing metrics server
   - No resource requests defined
   - Too aggressive thresholds
   - Insufficient stabilization""",
                "description": "How to properly configure autoscaling"
            },
            
            # Security Best Practices
            "security_practice": {
                "type": "best_practice",
                "category": "security",
                "keywords": ["security", "securitycontext", "privileged", "root"],
                "content": """Security Best Practices:

1. Never run as root:
   securityContext:
     runAsNonRoot: true
     runAsUser: 1000
     fsGroup: 1000

2. Drop capabilities:
   securityContext:
     capabilities:
       drop:
       - ALL

3. Read-only filesystem:
   securityContext:
     readOnlyRootFilesystem: true

4. No privileged:
   securityContext:
     privileged: false
     allowPrivilegeEscalation: false

5. Network policies for isolation
6. Use secrets for sensitive data
7. Enable RBAC""",
                "description": "Security hardening for Kubernetes pods"
            }
        }
    
    def retrieve_relevant_context(
        self, 
        query: str, 
        category: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge base entries using keyword matching
        Simple but effective - no embeddings needed!
        """
        
        query_lower = query.lower()
        results = []
        
        for doc_id, doc in self.knowledge_base.items():
            # Skip if category filter doesn't match
            if category and doc.get("category") != category:
                continue
            
            # Calculate relevance score
            score = 0
            
            # Keyword matching
            for keyword in doc.get("keywords", []):
                if keyword in query_lower:
                    score += 2
            
            # Category match
            if doc.get("category", "") in query_lower:
                score += 1
            
            # Type match
            if doc.get("type", "") in query_lower:
                score += 1
            
            if score > 0:
                results.append({
                    "id": doc_id,
                    "score": score,
                    "content": doc.get("content", ""),
                    "description": doc.get("description", ""),
                    "best_practices": doc.get("best_practices", []),
                    "type": doc.get("type", ""),
                    "category": doc.get("category", "")
                })
        
        # Sort by score and return top N
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n_results]
    
    def build_rag_prompt(
        self, 
        user_query: str,
        relevant_docs: List[Dict[str, Any]]
    ) -> str:
        """Build enhanced prompt with RAG context"""
        
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"""
=== {doc['type'].upper()}: {doc['id']} ===
{doc['content']}

Description: {doc['description']}
""")
            if doc.get('best_practices'):
                context_parts.append("Best Practices:")
                for bp in doc['best_practices']:
                    context_parts.append(f"  - {bp}")
        
        rag_prompt = f"""You are a Kubernetes expert assistant. Generate a production-ready manifest based on the user's request.

USER REQUEST:
{user_query}

RELEVANT KUBERNETES KNOWLEDGE:
{''.join(context_parts)}

INSTRUCTIONS:
1. Use the templates and best practices provided above as reference
2. Generate VALID Kubernetes YAML
3. Include ALL best practices (resources, probes, labels, security)
4. Replace {{variable}} placeholders with appropriate values
5. Add helpful comments in the YAML
6. Ensure proper indentation (2 spaces)

Generate the manifest now:
"""
        return rag_prompt
    
    def learn_from_manifest(self, manifest: str, metadata: Dict[str, Any]):
        """Learn from user-created or cluster manifests"""
        
        doc_id = f"learned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.knowledge_base[doc_id] = {
            "type": "learned",
            "category": metadata.get("category", "custom"),
            "keywords": metadata.get("keywords", []),
            "content": manifest,
            "description": metadata.get("description", "User-provided manifest"),
            "source": "cluster" if metadata.get("from_cluster") else "user",
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_knowledge_base()
        print(f"✅ Learned new manifest: {doc_id}")
    
    def _save_knowledge_base(self):
        """Save knowledge base to disk"""
        kb_file = os.path.join(self.kb_path, "knowledge.json")
        with open(kb_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        
        stats = {
            "total_entries": len(self.knowledge_base),
            "by_type": {},
            "by_category": {}
        }
        
        for doc in self.knowledge_base.values():
            doc_type = doc.get("type", "unknown")
            doc_category = doc.get("category", "unknown")
            
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
            stats["by_category"][doc_category] = stats["by_category"].get(doc_category, 0) + 1
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = KubernetesRAG()
    
    # Test retrieval
    query = "create a deployment with autoscaling"
    results = rag.retrieve_relevant_context(query)
    
    print(f"\n🔍 Query: {query}")
    print(f"📚 Found {len(results)} relevant documents:\n")
    
    for result in results:
        print(f"  - {result['id']} (score: {result['score']})")
        print(f"    {result['description']}\n")
    
    # Build RAG prompt
    prompt = rag.build_rag_prompt(query, results)
    print("\n" + "="*80)
    print("📝 RAG-Enhanced Prompt:")
    print("="*80)
    print(prompt)
    
    # Show stats
    print("\n" + "="*80)
    print("📊 Knowledge Base Stats:")
    print("="*80)
    stats = rag.get_stats()
    print(json.dumps(stats, indent=2))


