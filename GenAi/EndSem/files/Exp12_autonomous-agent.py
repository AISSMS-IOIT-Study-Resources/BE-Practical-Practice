class SimpleAgent:
    def __init__(self, name):
        self.name = name
        self.state = {'energy': 10}

    def decide(self, observation):
        # Very simple rule-based policy
        if observation.get('food') and self.state['energy'] < 5:
            action = 'eat'
        elif observation.get('threat'):
            action = 'run'
        else:
            action = 'explore'
        return action

def simulate_multi_agent(agents, steps=5):
    env = [{'food': True, 'threat': False}, {'food': False, 'threat': True}, {'food': False, 'threat': False}]
    history = []
    for t in range(steps):
        for i, agent in enumerate(agents):
            obs = env[i % len(env)]
            action = agent.decide(obs)
            # Update simple state
            if action == 'eat':
                agent.state['energy'] += 3
            elif action == 'run':
                agent.state['energy'] -= 2
            else:
                agent.state['energy'] -= 1
            history.append((agent.name, t, action, dict(agent.state)))
    return history

if __name__ == '__main__':
    a1 = SimpleAgent('A1')
    a2 = SimpleAgent('A2')
    h = simulate_multi_agent([a1, a2], steps=3)
    
    print("Simulation History:")
    for item in h:
        print(f"Agent {item[0]} at step {item[1]}: action='{item[2]}', state={item[3]}")
    
    # Analysis
    final_energies = [h[-2][3]['energy'], h[-1][3]['energy']]  # Last states of A1 and A2
    avg_energy = sum(final_energies) / len(final_energies)
    actions_count = {}
    for item in h:
        action = item[2]
        actions_count[action] = actions_count.get(action, 0) + 1
    
    print("\nAnalysis:")
    print(f"Final energies: {final_energies}")
    print(f"Average final energy: {avg_energy:.2f}")
    print(f"Action distribution: {actions_count}")
    print("Agents survived with positive energy, demonstrating basic decision-making.")