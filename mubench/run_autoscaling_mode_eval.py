#!/usr/bin/env python3
"""
AutoSage Autoscaling Mode Selection Test
=========================================
Ground truth:
  ingest  → stateless → HPA expected (scale-out: add replicas)
  process → stateless → HPA expected (scale-out: add replicas)
  analyze → stateful  → VPA expected (scale-up: increase CPU/memory limits)

Test flow:
  1. Set annotations (ingest/process=stateless, analyze=stateful)
  2. Pin maxReplicas=2 (freeze reactive HPA so we can observe predictive decisions)
  3. Apply 48-connection load for 120s
  4. Sample CPU at T+30s
  5. Fire AutoSage (LLM+MCDA) for all three services with real hpa_manager
  6. Compare result against ground truth → pass/fail
  7. ACT: apply VPA patch for analyze; restore maxReplicas=4 for ingest/process
  8. Verify final cluster state
"""

import sys, os, time, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
for _line in open(_env_path):
    _line = _line.strip()
    if '=' in _line and not _line.startswith('#'):
        _k, _v = _line.split('=', 1)
        os.environ.setdefault(_k.strip(), _v.strip())

from llm_autoscaling_advisor import LLMAutoscalingAdvisor
from autoscaling_engine import HorizontalPodAutoscaler
from vpa_engine import VerticalPodAutoscaler

import logging
logging.disable(logging.WARNING)   # suppress debug noise in output

# ── Ground truth ──────────────────────────────────────────────────────
GROUND_TRUTH = {
    'ingest':  {'state': 'stateless', 'expected_type': 'hpa', 'cpu_limit_m': 500, 'mem_limit': '256Mi'},
    'process': {'state': 'stateless', 'expected_type': 'hpa', 'cpu_limit_m': 500, 'mem_limit': '256Mi'},
    'analyze': {'state': 'stateful',  'expected_type': 'vpa', 'cpu_limit_m': 300, 'mem_limit': '256Mi'},
}

def kubectl(*args, silent=False):
    r = subprocess.run(['kubectl'] + list(args), capture_output=True, text=True)
    if not silent and r.returncode != 0:
        print(f'  [kubectl error] {" ".join(args)}: {r.stderr.strip()[:200]}')
    return r.stdout.strip()

def get_svc_ip(name):
    return kubectl('get', 'svc', name, '-o', 'jsonpath={.spec.clusterIP}')

def get_cpu_millis(svc):
    out = kubectl('top', 'pods', '-l', f'app={svc}', '--no-headers', silent=True)
    vals = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals.append(int(parts[1].replace('m', '')))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else 0.0

def set_annotation(svc, key, value):
    kubectl('annotate', 'deployment', svc, f'{key}={value}', '--overwrite')

def patch_hpa_max(svc, max_val):
    patch = json.dumps({'spec': {'maxReplicas': max_val}})
    kubectl('patch', 'hpa', svc, '--type=merge', f'-p={patch}')

def get_current_resources(svc):
    out = kubectl('get', 'deployment', svc,
                  '-o', 'jsonpath={.spec.template.spec.containers[0].resources.limits}')
    try:
        d = json.loads(out)
        return d.get('cpu', 'N/A'), d.get('memory', 'N/A')
    except Exception:
        return 'N/A', 'N/A'

def separator(title=''):
    print('\n' + '─' * 60)
    if title:
        print(f'  {title}')
        print('─' * 60)

# ── Main ──────────────────────────────────────────────────────────────
def main():
    hpa = HorizontalPodAutoscaler()
    vpa = VerticalPodAutoscaler()
    advisor = LLMAutoscalingAdvisor()

    separator('STEP 1 — Set state-management annotations')
    for svc, gt in GROUND_TRUTH.items():
        set_annotation(svc, 'autosage.ai4k8s/state-management', gt['state'])
        info = advisor._detect_state_management(svc, 'default', hpa)
        ok = '✓' if info['type'] == gt['state'] else '✗'
        print(f'  {ok} {svc}: detected={info["type"]} (expected={gt["state"]}, confidence={info["confidence"]})')

    separator('STEP 2 — Pin maxReplicas=2 (freeze reactive HPA)')
    for svc in GROUND_TRUTH:
        patch_hpa_max(svc, 2)
    print(kubectl('get', 'hpa'))

    separator('STEP 3 — Start load (48 connections, 120s)')
    ingest_ip = get_svc_ip('ingest')
    print(f'  ingest ClusterIP: {ingest_ip}')
    wrk = subprocess.Popen(
        ['/usr/bin/wrk', '-t', '4', '-c', '48', '-d', '120s',
         f'http://{ingest_ip}:8080/api/v1'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    t0 = time.time()
    print(f'  wrk PID={wrk.pid}')

    separator('STEP 4 — Wait 30s, sample CPU')
    time.sleep(30)
    cpu_snapshot = {}
    for svc, gt in GROUND_TRUTH.items():
        cpu = round(get_cpu_millis(svc) / gt['cpu_limit_m'] * 100, 1)
        cpu_snapshot[svc] = cpu
        print(f'  {svc}: {cpu:.1f}%  (limit={gt["cpu_limit_m"]}m, state={gt["state"]})')

    separator('STEP 5 — Fire AutoSage LLM+MCDA for each service')
    results = {}
    for svc, gt in GROUND_TRUTH.items():
        cpu = cpu_snapshot[svc]
        limit_m = gt['cpu_limit_m']
        peak = min(cpu * 2.5, 350.0)
        preds = [round(cpu * (1 + 0.35 * i), 1) for i in range(6)]

        print(f'\n  [{svc}] cpu={cpu}% state={gt["state"]} expected={gt["expected_type"].upper()}')
        t1 = time.time()
        result = advisor.get_intelligent_recommendation(
            deployment_name=svc,
            namespace='default',
            current_metrics={
                'cpu_usage': cpu,
                'memory_usage': 28.0,
                'pod_count': 2,
                'running_pod_count': 2,
            },
            forecast={
                'cpu': {
                    'current': cpu,
                    'peak': peak,
                    'trend': 'rapidly_increasing',
                    'predictions': preds,
                },
                'memory': {
                    'current': 28.0,
                    'peak': 33.0,
                    'trend': 'stable',
                    'predictions': [28.0] * 6,
                },
            },
            hpa_status={
                'exists': True,
                'current_replicas': 2,
                'desired_replicas': 2,
                'target_cpu': 70,
                'target_memory': 80,
                'scaling_status': 'active',
            },
            vpa_status={'exists': False},
            current_resources={
                'cpu_request': '125m',
                'cpu_limit': f'{limit_m}m',
                'memory_request': '128Mi',
                'memory_limit': gt['mem_limit'],
            },
            current_replicas=2,
            min_replicas=2,
            max_replicas=4,
            hpa_manager=hpa,   # real hpa_manager so state detection reads annotations
        )
        elapsed = time.time() - t1
        rec = result.get('recommendation', {})
        scaling_type = rec.get('scaling_type', 'hpa')
        action = rec.get('action', '?')
        target_replicas = rec.get('target_replicas')
        target_cpu = rec.get('target_cpu')
        target_memory = rec.get('target_memory')
        confidence = rec.get('confidence', 0)
        mcda = rec.get('mcda_validation', {})
        model = result.get('llm_model', '?')

        passed = scaling_type == gt['expected_type']
        status = 'PASS' if passed else 'FAIL'

        print(f'  [{status}] scaling_type={scaling_type} action={action}')
        if target_replicas is not None:
            print(f'         target_replicas={target_replicas}')
        if target_cpu or target_memory:
            print(f'         target_cpu={target_cpu}  target_memory={target_memory}')
        print(f'         conf={confidence:.2f}  model={model}  ({elapsed:.0f}s)')
        print(f'         MCDA: agreement={mcda.get("agreement","?")} gap={mcda.get("score_gap",0):.4f} override={mcda.get("should_override",False)}')
        print(f'         reason: {rec.get("reasoning","")[:180]}')

        results[svc] = {
            'expected_type': gt['expected_type'],
            'scaling_type': scaling_type,
            'action': action,
            'target_replicas': target_replicas,
            'target_cpu': target_cpu,
            'target_memory': target_memory,
            'passed': passed,
            'inference_s': round(elapsed, 1),
        }

    separator('STEP 6 — ACT on decisions')
    for svc, r in results.items():
        if r['scaling_type'] == 'vpa' and r['target_cpu'] and r['target_memory']:
            cpu_before, mem_before = get_current_resources(svc)
            patch_result = vpa.patch_deployment_resources(
                svc, 'default',
                cpu_request=r['target_cpu'],
                memory_request=r['target_memory'],
                cpu_limit=r['target_cpu'],
                memory_limit=r['target_memory'],
            )
            cpu_after, mem_after = get_current_resources(svc)
            ok = '✓' if patch_result.get('success') else '✗'
            print(f'  {ok} VPA patch {svc}: {cpu_before}/{mem_before} → {cpu_after}/{mem_after}')
        elif r['scaling_type'] == 'hpa':
            # Restore maxReplicas=4 so HPA can scale
            patch_hpa_max(svc, 4)
            print(f'  ✓ HPA {svc}: maxReplicas restored to 4 (HPA will scale reactively)')
        else:
            patch_hpa_max(svc, 4)
            print(f'  ~ {svc}: restored maxReplicas=4 (no VPA target values to apply)')

    separator('STEP 7 — Final cluster state')
    time.sleep(5)
    print(kubectl('get', 'hpa'))
    print()
    for svc in GROUND_TRUTH:
        cpu_lim, mem_lim = get_current_resources(svc)
        print(f'  {svc}: cpu_limit={cpu_lim}  mem_limit={mem_lim}')

    separator('RESULTS SUMMARY')
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(results)
    print(f'\n  {"Service":<10} {"Expected":<10} {"Got":<10} {"Action":<14} {"Pass?":<6} {"Time":>6}')
    print(f'  {"-"*58}')
    for svc, r in results.items():
        icon = '✓' if r['passed'] else '✗'
        target = (f'replicas={r["target_replicas"]}' if r['target_replicas'] is not None
                  else f'cpu={r["target_cpu"]}')
        print(f'  {icon} {svc:<9} {r["expected_type"]:<10} {r["scaling_type"]:<10} {target:<14} {str(r["passed"]):<6} {r["inference_s"]:>5}s')
    print(f'\n  Score: {passed}/{total} correct')

    wrk.terminate()
    elapsed_total = time.time() - t0
    print(f'\n  Total test time: {elapsed_total:.0f}s')
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
