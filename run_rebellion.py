"""
Main script to run the Rebellion Model simulation.
"""

from rebellion_model import RebellionModel, Visualization

def main():
    # Create model with default parameters (similar to NetLogo model)
    model = RebellionModel(
        width=40,
        height=40,
        vision=7,
        initial_cop_density=4,
        initial_agent_density=70,
        government_legitimacy=0.8,
        max_jail_term=30
    )
    
    # Set up the model (create agents and cops)
    model.setup()
    
    # Create visualization and run animation
    vis = Visualization(model)
    animation = vis.animate(frames=200)
    
    # Alternative: Run model for fixed number of steps without animation
    # model.run(200)
    # vis.show_final_state()

if __name__ == "__main__":
    main()