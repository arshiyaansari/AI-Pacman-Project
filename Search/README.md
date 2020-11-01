# Part One: Search Algorithms 

In this project, the Pacman agent will find paths through his maze world, both to reach a particular location and to
collect food efficiently. In this part of the proejct, I built general search algorithms and applied them to Pacman scenarios.

Files that were edited to implement the search algorithms: 
- search.py: Where all the search algorithms will reside
- searchAgents.py: Where the search-based agents will reside 

All other files were copied from CS 188: Introduction to Artifical Intelligence at Berkeley. 
Files of importance: 
- pacman.py: Main file that runs Pacman games. Describes a Pacman GameState type. 
- game.py: The logic behind how the Pacman world works. This file describes several supporting types like AgentState, Agent, Direction, and Grid.
- util.py: Useful data structures for implementing search algorithms.

In this project, we implement a depth first search, breadth first search, uniform-cost graph search, A* search with different heuristics, and a suboptimal search using a greedy algorithm. 