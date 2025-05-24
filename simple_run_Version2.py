"""
Simple script to run the Rebellion Model simulation without advanced visualization.
"""

from rebellion_model import RebellionModel, Visualization
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Create model with default parameters (similar to NetLogo model)
    model = RebellionModel(
        width=40,
        height=40,
        vision=7,
        initial_cop_density=4,
        initial_agent_density=70,
        government_legitimacy=0.83,
        max_jail_term=30
    )
    
    print("Setting up model...")
    model.setup()
    
    # Option 1: Run the model for a specific number of steps without animation
    steps = 200
    print(f"Running model for {steps} steps...")
    stats = model.run(steps)
    
    # Plot final statistics
    plt.figure(figsize=(10, 6))
    x = range(len(stats['active_agents']))
    plt.plot(x, stats['active_agents'], 'r-', label='Active')
    plt.plot(x, stats['jailed_agents'], 'k-', label='Jailed')
    plt.plot(x, stats['quiet_agents'], 'g-', label='Quiet')
    plt.title('Agent States Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Agents')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Option 2: Display visualization of the final state
    print("Displaying final state...")
    vis = Visualization(model)
    # vis.animate(frames=100)
    vis.show_final_state()
    
    # Option 3: Run animation (uncomment to use)
    # print("Running animation...")
    # vis = Visualization(model)
    # vis.animate(frames=100)

if __name__ == "__main__":
    main()