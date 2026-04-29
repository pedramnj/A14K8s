import unittest
from agent.train import SimulatedEnvironment

class IntegrationTests(unittest.TestCase):
    def test_sim_environment_step(self):
        env = SimulatedEnvironment()
        s = env.reset()
        self.assertEqual(len(s), 3)
        next_s, r, done, info = env.step(1)
        self.assertIn('latency', info)

#    def test_prometheus_query(self):
#        pc = PrometheusCollector(base_url='http://localhost:9090')
#        try:
#            res = pc.query('up')
#        except Exception as e:
#            raise unittest.SkipTest('Prometheus not available: ' + str(e))
#        self.assertIn('status', res)

if __name__ == '__main__':
    unittest.main()