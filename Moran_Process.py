from random import *
import Graphs

class Mutant:
   def __init__(self, fitness, id_n = 'mutant', color = 'red'):
      self.fitness = fitness
      self.color = color
      self.id_n = id_n
   def __hash__(self):
      return self.id_n
   def __cmp__(self, p):
      if self.id_n < p.id_n: return -1
      elif self.id_n == p.id_n: return 0
      else: return 1
   def __eq__(self, p):
      if p.id_n == self.id_n: return True
      else: return False
   def __repr__(self):
      return str(self.id_n)

class Resident:
   def __init__(self,fitness,id_n = 'resident',color = 'blue'):
      self.id_n = id_n
      self.fitness = fitness
      self.color = color
   def __hash__(self):
      return self.id_n
   def __cmp__(self, p):
      if self.id_n < p.id_n: return -1
      elif self.id_n == p.id_n: return 0
      else: return 1
   def __eq__(self, p):
      if p.id_n == self.id_n: return True
      else: return False
   def __repr__(self):
      return str(self.id_n)

# Mutate neighbor
def step(G): #Idk the arg might be wrong
   # Arg 1 -> Might be something different than the graph
   # Arg 2 -> Node that has been choosen for mutation based upon fitness

   # Get all neighboring nodes and walk on a edge based upond the weights

   #Choose a node based on fitness (for now it's just uniformly)
   node = randint(0,len(G.nodes())-1)
   print("Node", node)

   #Find all node neighbors
   neighbors = [n for n in G.neighbors(0)]
   print(neighbors)

   #Mutate a neighbor
   nodeToMutate = random.choice(neighbors)

   if nodeToMutate['type'] == 'resident':
      nodeType = Mutant(1)
   else:
      nodeType = Resident(1)

   G.nodes[nodeToMutate]['type'] = nodeType

   Graphs.drawGraph(G)

# Uniformly picks a node to initially mutate
def mutateARandomNode(G):
   # generate 'random' node to mutate
   node = randint(0,len(G.nodes())-1)
   nodeType = Mutant(1)
   G.nodes[node]['type'] = nodeType
   Graphs.drawGraph(G)

if __name__ == "__main__":
   G = Graphs.createCompleteGraph()
   mutateARandomNode(G)
   step(G)




