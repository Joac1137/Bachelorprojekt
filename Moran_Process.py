import networkx as nx
import matplotlib.pyplot as plt

class Mutant:
   def __init__(self, fitness, id_n = 'mutant', color = 'red'):
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




if __name__ == "__main__":
   print("haj")




