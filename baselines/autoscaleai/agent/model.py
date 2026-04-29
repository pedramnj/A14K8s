from agent.policy_network import PolicyNetwork
import torch
import numpy as np

class AutoScaleAgent:
    def __init__(self, state_dim, action_dim, hidden=128, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.policy = PolicyNetwork(state_dim, hidden, action_dim).to(self.device)

    def get_action_probs(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).flatten()
        state = state.to(self.device).unsqueeze(0)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).detach().cpu().numpy()

    def act(self, state):
        probs = self.get_action_probs(state)
        action = np.random.choice(len(probs), p=probs)
        return int(action)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))