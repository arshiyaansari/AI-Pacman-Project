# Part Two: Multi-Agent Search Algorithms 

In this part of the project, I designed agents for the classic version of Pacman, including ghosts. Along the way, I implemented both minimax and expectimax search and tried my hand at evaluation function design

Files that were edited to implement the search algorithms: 
- multiAgents.py: Where all the multi-agent search agents will reside. 
- pacman.py: Main file that runs Pacman games. Describes a Pacman GameState type. 
- game.py: The logic behind how the Pacman world works. This file describes several supporting types like AgentState, Agent, Direction, and Grid.
- util.py: Useful data structures for implementing search algorithms.

All other files were copied from CS 188: Introduction to Artifical Intelligence at Berkeley: http://ai.berkeley.edu/multiagent.html


In this project, we implement different methods for the creation of promising search agents. These methods include implementing a reflex agent, a minimax agent, alpha-beta pruning, an expectimax agent, and a new evalulation function. 