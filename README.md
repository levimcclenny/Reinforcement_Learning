# *Reinforcement Learning: An Introduction*

One text that is widely regarded as the "industry standard" in Reinforcement Learning is Sutton and Barto's *Reinforcement Learning: An Introduction.* Here you will find the supporting source code for the jupyter notebooks found on [my website](http://people.tamu.edu/~levimcclenny/project/reinforcement-learning/), as well as in the links below. My hope is that the code and the insights offered in these notebooks will help a causal reader better understand the power of reinforcement learning. 

### [Chapter 2 - Multi-Arm Bandits](http://people.tamu.edu/~levimcclenny/project/reinforcement-learning/Barto_Sutton_RL/Chapter2.html)
This section of the book is dedicated to framing the optimal control and award maximization principles via an illustration though the "multi-armed bandit." More is discussed in the accompanying [Multi-Arm Bandit Jupyter Notebook](Barto_Sutton_RL/Chapter2.html). The standalone python source code can be found [here](https://github.com/levimcclenny/Reinforcement_Learning).

### [Chapter 3 - Finite Markov Decision Processes](http://people.tamu.edu/~levimcclenny/project/reinforcement-learning/Barto_Sutton_RL/Chapter3.html)
Here we begin to formulate the reinforcement learning problem, starting with markov chains and moving into Markov Decision Processes. We evaluate random and optimal value functions using the Gridworld example outlined in the text. Some of the more important takeaways are outlined in the [Finite Markov Decision Processes Jupyter Notebook](Barto_Sutton_RL/Chapter3.html) file and the source code is available [here](https://github.com/levimcclenny/Reinforcement_Learning).

### [Chapter 4 - Dynamic Programming](http://people.tamu.edu/~levimcclenny/project/reinforcement-learning/Barto_Sutton_RL/Chapter4.html)
This chapter offers some insights into some popular, albeit fairly basic, dynamic programming algorithms used to evaluate MDPs when all the state-action interactions are known. Hence, in this chapter, we know all the transition probabilities, the rewards, etc, and there isnt anything to really "figure out" as far as the agent-environment interface is concerned. However, dont misconstrue that description with a lack of application of these algorithms, as you can see in the [Dynamic Programming Jupyter Notebook](http://people.tamu.edu/~levimcclenny/project/reinforcement-learning/Barto_Sutton_RL/Chapter4.html) and in the text for this chapter, we can apply these algorithms to real-life problems and handle some subtle non-linearities in problems that traditional optimization algorithms might struggle with. Thus, the efficacy of these algorithms, and hence their presentation in this chapter and widespread use, is something to be valued. As always, the source code can be found on [my github](https://github.com/levimcclenny/Reinforcement_Learning).




Obligatory disclaimer: This is not original research, but rather my insights into this incredible book.
