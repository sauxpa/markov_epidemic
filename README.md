# markov_epidemic

Markov stochastic models (SIS, SIR, SEIR...) to describe the evolution of epidemics on a network of connected individuals.

This repo contains standalone code for simulating epidemics and study the influence of the underlying graph and points to some theoretical results regarding the expected lifetime of an epidemic as a function of its infection/recovery rates and the network topology (in particular its adjacency spectrum).

Enjoy while you stay indoors helping us reduce our spectral radius :) 

**To install** : clone repo and pip install -e markov_epidemic/ (or see https://pypi.org/project/markov-epidemic/)

Also check out the Bokeh app to play with different networks and epidemic models dynamically:

**To run the app** : run bokeh serve --show markov_epidemic_app/

### France data vs SEIR

<img src="./pictures/seir_france.png"
     alt="France_SEIR"
     style="float: left; margin-right: 10px;" />

### Markov models

<img src="./pictures/description_1.png"
     alt="description_1"
     style="float: left; margin-right: 10px;" />


<img src="./pictures/description_2.png"
     alt="description_2"
     style="float: left; margin-right: 10px;" />
 
 
<img src="./pictures/description_3.png"
     alt="description_3"
     style="float: left; margin-right: 10px;" />
     
### Susceptible-Infected-Recovered (SIR)

<img src="./pictures/sir_epidemic.png"
     alt="SIR"
     style="float: left; margin-right: 10px;" />

### Susceptible-Infected-Susceptible (SIS)

<img src="./pictures/sis_epidemic.png"
     alt="SIS"
     style="float: left; margin-right: 10px;" />

### Susceptible-Exposed-Infected-Susceptible (SEIR)

<img src="./pictures/seir_epidemic.png"
     alt="SEIR"
     style="float: left; margin-right: 10px;" />
