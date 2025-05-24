"""
Rebellion Model: A Python implementation of the Civil Violence model by Epstein (2002).

This model simulates the dynamics of civil violence between citizens and law enforcement.
Citizens have individual grievance levels based on perceived hardship and government legitimacy,
and decide whether to rebel based on their grievance and risk assessment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.patches import Circle, RegularPolygon
import random
from collections import defaultdict

class Cell:
    """
    Represents a single location in the grid where agents and cops can be located.
    Acts as a container for turtles at a specific x,y coordinate.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.agents = []  # List of agents in this cell
        self.cops = []    # List of cops in this cell
        self.neighborhood = []  # Will be populated with nearby cells

    def is_empty(self):
        """Returns True if the cell has no active agents or cops."""
        return (len(self.cops) == 0 and 
                all(agent.jail_term > 0 for agent in self.agents))
    
    def get_active_agents(self):
        """Returns a list of actively rebelling agents in this cell."""
        return [agent for agent in self.agents if agent.active and agent.jail_term == 0]


class Agent:
    """
    Represents a citizen in the model that can decide to rebel based on grievance and risk.
    """
    def __init__(self, x, y, model, risk_aversion=None, perceived_hardship=None):
        self.x = x
        self.y = y
        self.model = model
        # Initialize with random values if not provided
        self.risk_aversion = risk_aversion if risk_aversion is not None else random.random()
        self.perceived_hardship = perceived_hardship if perceived_hardship is not None else random.random()
        self.active = False
        self.jail_term = 0

    def move_to(self, cell):
        """Move the agent to a new cell."""
        # Remove from current cell
        current_cell = self.model.grid[self.y][self.x]
        if self in current_cell.agents:
            current_cell.agents.remove(self)
        
        # Add to new cell
        cell.agents.append(self)
        self.x, self.y = cell.x, cell.y

    def move(self):
        """
        Implements Rule M: Move to a random site within vision if movement is enabled
        or if the agent is not in jail.
        """
        if self.jail_term == 0 and self.model.movement:
            current_cell = self.model.grid[self.y][self.x]
            targets = [cell for cell in current_cell.neighborhood 
                      if cell.is_empty() or all(agent.jail_term > 0 for agent in cell.agents)]
            
            if targets:
                new_cell = random.choice(targets)
                self.move_to(new_cell)

    def determine_behavior(self):
        """
        Implements Rule A: Determine if agent should be actively rebelling based on
        grievance and risk assessment.
        """
        if self.jail_term == 0:
            grievance = self.calculate_grievance()
            risk = self.risk_aversion * self.estimated_arrest_probability()
            self.active = (grievance - risk > self.model.threshold)

    def calculate_grievance(self):
        """
        Calculate agent's grievance based on perceived hardship and government legitimacy.
        """
        return self.perceived_hardship * (1 - self.model.government_legitimacy)

    def estimated_arrest_probability(self):
        """
        Calculate estimated probability of arrest based on local cop and active agent counts.
        """
        current_cell = self.model.grid[self.y][self.x]
        c = sum(len(cell.cops) for cell in current_cell.neighborhood)
        active_agents = sum(len([a for a in cell.agents if a.active]) 
                           for cell in current_cell.neighborhood)
        a = 1 + active_agents  # Add 1 to prevent division by zero
        
        # Implementation of P = 1 - exp(-k * floor(C/A))
        return 1 - np.exp(-self.model.k * np.floor(c / a))


class Cop:
    """
    Represents a law enforcement officer that arrests actively rebelling agents.
    """
    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self.model = model

    def move_to(self, cell):
        """Move the cop to a new cell."""
        # Remove from current cell
        current_cell = self.model.grid[self.y][self.x]
        if self in current_cell.cops:
            current_cell.cops.remove(self)
        
        # Add to new cell
        cell.cops.append(self)
        self.x, self.y = cell.x, cell.y

    def move(self):
        """
        Implements Rule M for cops: Move to a random site within vision.
        """
        current_cell = self.model.grid[self.y][self.x]
        targets = [cell for cell in current_cell.neighborhood 
                  if cell.is_empty() or all(agent.jail_term > 0 for agent in cell.agents)]
        
        if targets:
            new_cell = random.choice(targets)
            self.move_to(new_cell)

    def enforce(self):
        """
        Implements Rule C: Look for active agents within vision and arrest one if found.
        """
        current_cell = self.model.grid[self.y][self.x]
        active_agents = []
        
        # Check all cells in neighborhood for active agents
        for cell in current_cell.neighborhood:
            active_agents.extend(cell.get_active_agents())
        
        if active_agents:
            # Arrest a random active agent
            suspect = random.choice(active_agents)
            suspect.active = False
            suspect.jail_term = random.randint(1, self.model.max_jail_term)
            # Move to the cell of the arrested agent
            suspect_cell = self.model.grid[suspect.y][suspect.x]
            self.move_to(suspect_cell)


class RebellionModel:
    """
    Main simulation class that manages the grid, agents, cops, and simulation rules.
    """
    def __init__(self, width, height, vision=7, initial_cop_density=4, 
                initial_agent_density=70, government_legitimacy=0.8, max_jail_term=30):
        self.width = width
        self.height = height
        self.vision = vision
        self.initial_cop_density = initial_cop_density
        self.initial_agent_density = initial_agent_density
        self.government_legitimacy = government_legitimacy
        self.max_jail_term = max_jail_term
        self.movement = True  # Equivalent to MOVEMENT? toggle
        self.k = 2.3  # Factor for determining arrest probability
        self.threshold = 0.1  # Threshold for rebellion
        
        # Initialize grid and populations
        self.grid = [[Cell(x, y) for x in range(width)] for y in range(height)]
        self.agents = []
        self.cops = []
        self.time = 0
        
        # Initialize neighborhood for each cell
        self._init_neighborhoods()
        
        # Statistics tracking
        self.stats = {
            'active_agents': [],
            'quiet_agents': [],
            'jailed_agents': []
        }

    def _init_neighborhoods(self):
        """
        Initialize the neighborhood for each cell based on vision radius.
        This is a performance optimization to avoid recalculating neighborhoods.
        """
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                neighborhood = []
                
                for ny in range(max(0, y - self.vision), min(self.height, y + self.vision + 1)):
                    for nx in range(max(0, x - self.vision), min(self.width, x + self.vision + 1)):
                        # Check if cell is within vision radius (using Euclidean distance)
                        if ((nx - x) ** 2 + (ny - y) ** 2) <= self.vision ** 2:
                            neighborhood.append(self.grid[ny][nx])
                
                cell.neighborhood = neighborhood

    def setup(self):
        """
        Initialize the simulation by creating agents and cops based on density settings.
        """
        # Check if densities are valid
        if self.initial_cop_density + self.initial_agent_density > 100:
            raise ValueError("Sum of cop and agent densities must not exceed 100")
        
        # Create a list of all available positions
        available_positions = [(x, y) for y in range(self.height) for x in range(self.width)]
        random.shuffle(available_positions)
        
        # Calculate number of cops and agents
        total_cells = self.width * self.height
        num_cops = round(self.initial_cop_density * 0.01 * total_cells)
        num_agents = round(self.initial_agent_density * 0.01 * total_cells)
        
        # Create cops
        for i in range(min(num_cops, len(available_positions))):
            x, y = available_positions.pop()
            cop = Cop(x, y, self)
            self.cops.append(cop)
            self.grid[y][x].cops.append(cop)
        
        # Create agents
        for i in range(min(num_agents, len(available_positions))):
            x, y = available_positions.pop()
            agent = Agent(x, y, self)
            self.agents.append(agent)
            self.grid[y][x].agents.append(agent)
        
        # Initialize statistics
        self._update_stats()

    def step(self):
        """
        Advance the simulation by one time step, applying all rules.
        """
        # Rule M: Move agents and cops
        for agent in self.agents:
            if agent.jail_term == 0:
                agent.move()
        
        for cop in self.cops:
            cop.move()
        
        # Rule A: Agents determine behavior
        for agent in self.agents:
            agent.determine_behavior()
        
        # Rule C: Cops enforce
        for cop in self.cops:
            cop.enforce()
        
        # Reduce jail terms
        for agent in self.agents:
            if agent.jail_term > 0:
                agent.jail_term -= 1
        
        # Update statistics
        self._update_stats()
        self.time += 1

    def _update_stats(self):
        """
        Update statistics about the current state of the simulation.
        """
        active_count = len([a for a in self.agents if a.active])
        jailed_count = len([a for a in self.agents if a.jail_term > 0])
        quiet_count = len(self.agents) - active_count - jailed_count
        
        self.stats['active_agents'].append(active_count)
        self.stats['jailed_agents'].append(jailed_count)
        self.stats['quiet_agents'].append(quiet_count)

    def run(self, steps):
        """Run the simulation for a specified number of steps."""
        for _ in range(steps):
            self.step()
        return self.stats


class Visualization:
    """
    Handles visualization of the Rebellion model using Matplotlib.
    """
    def __init__(self, model):
        self.model = model
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.suptitle('Civil Violence Model', fontsize=16)
        
        # Setup grid visualization
        self.ax1.set_xlim(-1, model.width)
        self.ax1.set_ylim(-1, model.height)
        self.ax1.set_title('Agent States')
        self.ax1.set_aspect('equal')
        self.ax1.axis('off')
        
        # Setup statistics plot
        self.ax2.set_xlim(0, 100)  # Will be updated dynamically
        self.ax2.set_ylim(0, len(model.agents))
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Number of Agents')
        self.ax2.set_title('Agent States Over Time')
        
        # Initialize lines for statistics
        self.active_line, = self.ax2.plot([], [], 'r-', label='Active')
        self.jailed_line, = self.ax2.plot([], [], 'k-', label='Jailed')
        self.quiet_line, = self.ax2.plot([], [], 'g-', label='Quiet')
        self.ax2.legend()
        
        # Initialize agent and cop markers
        self.agent_markers = []
        self.cop_markers = []
        self._init_markers()

    def _init_markers(self):
        """Initialize visual markers for agents and cops."""
        for agent in self.model.agents:
            color = self._get_agent_color(agent)
            marker = Circle((agent.x, agent.y), 0.4, color=color)
            self.ax1.add_patch(marker)
            self.agent_markers.append(marker)
        
        for cop in self.model.cops:
            # Fixed: proper parameters for RegularPolygon
            marker = RegularPolygon((cop.x, cop.y), numVertices=3, radius=0.5, color='cyan')
            self.ax1.add_patch(marker)
            self.cop_markers.append(marker)

    def _get_agent_color(self, agent):
        """Determine the color of an agent based on its state."""
        if agent.active:
            return 'red'
        elif agent.jail_term > 0:
            return 'black'
        else:
            # Scale color based on grievance (green -> darker green for higher grievance)
            grievance = agent.calculate_grievance()
            return mcolors.to_rgba('green', 0.3 + 0.7 * grievance)

    def update(self, frame):
        """Update the visualization for a new frame/time step."""
        # Update agent markers
        for agent, marker in zip(self.model.agents, self.agent_markers):
            marker.center = (agent.x, agent.y)
            marker.set_color(self._get_agent_color(agent))
        
        # Update cop markers
        for cop, marker in zip(self.model.cops, self.cop_markers):
            marker.xy = (cop.x, cop.y)
        
        # Update statistics plot
        x = list(range(len(self.model.stats['active_agents'])))
        self.active_line.set_data(x, self.model.stats['active_agents'])
        self.jailed_line.set_data(x, self.model.stats['jailed_agents'])
        self.quiet_line.set_data(x, self.model.stats['quiet_agents'])
        
        # Adjust plot limits if needed
        if len(x) > 1:
            self.ax2.set_xlim(0, len(x))
        
        # Step the model
        self.model.step()
        
        return self.agent_markers + self.cop_markers + [self.active_line, self.jailed_line, self.quiet_line]

    def animate(self, frames=100):
        """Create an animation of the model running."""
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=frames,
            interval=100, blit=True
        )
        plt.tight_layout()
        plt.show()
        return ani

    def show_final_state(self):
        """Show the final state of the model without animation."""
        # Update agent markers
        for agent, marker in zip(self.model.agents, self.agent_markers):
            marker.center = (agent.x, agent.y)
            marker.set_color(self._get_agent_color(agent))
        
        # Update cop markers
        for cop, marker in zip(self.model.cops, self.cop_markers):
            marker.xy = (cop.x, cop.y)
        
        # Update statistics plot
        x = list(range(len(self.model.stats['active_agents'])))
        self.active_line.set_data(x, self.model.stats['active_agents'])
        self.jailed_line.set_data(x, self.model.stats['jailed_agents'])
        self.quiet_line.set_data(x, self.model.stats['quiet_agents'])
        
        # Adjust plot limits
        if len(x) > 1:
            self.ax2.set_xlim(0, len(x))
            
        plt.tight_layout()
        plt.show()