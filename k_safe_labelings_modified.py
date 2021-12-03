# library import
import networkx as nx
import operator
import random
from matplotlib import pyplot as plt
import math
from datetime import datetime
import csv
from pathlib import Path
import numpy as np

def k_safe_labeling(n,m,k,file):
  # clique size of a graph
  def large_clique_size(G):
    degrees = G.degree
    def _clique_heuristic(G, U, size, best_size):
        if not U:
            return max(best_size, size)
        u = max(U, key=degrees)
        U.remove(u)
        N_prime = {v for v in G[u] if degrees[v] >= best_size}
        return _clique_heuristic(G, U & N_prime, size + 1, best_size)

    best_size = 0
    nodes = (u for u in G if degrees[u] >= best_size)
    for u in nodes:
        neighbors = {v for v in G[u] if degrees[v] >= best_size}
        best_size = _clique_heuristic(G, neighbors, 1, best_size)
    return best_size

  E=[e for e in G.edges]                      # number of edges in inputgraph
  q=large_clique_size(G)                      # clique size of a graph
  q_now=nx.graph_clique_number(G)
  
  n=nx.number_of_nodes(G)
  G2=nx.complete_graph(n)                                                           #conversion to complete graph

  E_c=[e for e in nx.non_edges(G)]                                                  #edges in complement graph of G
  non_edge_cnt=len(E_c)
  file.write("Non edge number: "+str(len(E_c))+"\n")
  label_init={vertex:vertex*k+1  for vertex in G2.nodes}                           #labeling the complete graph using a dictionary
  max_label=max(label_init.values())                                               #finding the maximum labeled vertex
  for vertex, label in label_init.items():                                         #finding the maximum label
      if label == max_label:
          max_labeled=vertex
          break
  # sort vertices according to their degree        
  max_degree = sorted([d for n, d in G2.degree()], reverse=True)[0]
  

  #this function checks if a label violates the constraints of k-safe labeling
  def violate_constraints(G_now, label_now, k, vertex, label_this):                                       
    for node in nx.all_neighbors(G_now, vertex):
        if label_this in label_now.values() or (label_now[node] > 0 and ((label_this-label_now[node] < k 
		and label_this-label_now[node] > 0) or (label_now[node]-label_this < k and label_now[node]-label_this >0))):
            return True

  def non_over_max_cliques(G_now, max_clique_size):
    checked = []
    all_cliques=list(nx.find_cliques(G_now))
    maxm=max_clique_size
    non_overlap_cnts=0
    while maxm>=3:
        found=0
        for clique in all_cliques:
            if len(clique)==maxm:
                found=1
                flag=0
                for vertex in clique:
                    if vertex in checked:
                        flag=1
                        break
                if flag==0:
                    non_overlap_cnts+=1
                    for vertex in clique:
                        checked.append(vertex)
                all_cliques.remove(clique)
        if found==0:
          maxm=maxm-1

  #stack where removed edges are pushed
  edge_stack = []                                                                   
  same_count =0
  i=0
  edge_checked=set()
  flag_o=0
  clique_same_count=1
  curr_bound=0
  clique_unchanged=0
  clique_changed_flag=0
  clique_diff=0

  #first outer loop, label_init is the dictionary of the previous step
  while not len(E_c)==0:                                                            
      q_old=nx.graph_clique_number(G2)
      i+=1
      label_init={node: label for node, label in sorted(label_init.items(), key=lambda item: item[1], reverse=True)}
      label_p=label_init
      #sorting the vertices in decreasing order of labels                                                            
      nodes_reversed=[node for node in label_init]       
      e_incident=None
      flag_i=0
      # selection of edge to remove
      for e in E_c:
          if not e[0] in edge_checked and not e[1] in edge_checked:
              e_incident=e
              edge_checked.add(e[0])
              edge_checked.add(e[1])
              break
      
      
      if e_incident==None:
          if not len(edge_checked)==n:
              for e in E_c:
                  if e[0] in edge_checked and e[1] in edge_checked:
                      e_incident=e
                      break
          else:
              edge_checked=set()
                  
      if e_incident==None:
          edge_checked=set()
          continue

      #adding the edge to the stack        
      edge_stack.append(e_incident)                                                 
      #removing the edge from E_c
      E_c.remove(e_incident)

      if G2.has_edge(e_incident[0], e_incident[1]):
          G2.remove_edge(e_incident[0], e_incident[1])                              #removing the edge from G2
          
      elif G2.has_edge(e_incident[1], e_incident[0]):
          G2.remove_edge(e_incident[1], e_incident[0])
          
      q_now=nx.graph_clique_number(G2)

      #initializing the dictionary of current step with labels 0
      label_now={vertex:0  for vertex in nodes_reversed}                            
      checked={vertex:0 for vertex in nodes_reversed}
      #inner loop 1, taking edges in the reverse order of input to stack
      for e in edge_stack[::-1]:                                                    
          
          if checked[e[0]]==0: 
              #because a vertex can be labeled with at most the max_label of the previous step
              for label in range(1,max_label+1):      
                  #if both vertices incident to the edges are unlabeled, then label both with consecutive integers                              
                  if not violate_constraints(G2, label_now, k, e[0], label):        
                      label_now[e[0]]=label
                      break
          if checked[e[1]]==0:
              #because a vertex can be labeled with at most the max_label of the previous step
              for label in range(1,max_label+1):          
                  #if both vertices incident to the edges are unlabeled, then label both with consecutive integers                          
                  if not violate_constraints(G2, label_now, k, e[1], label):        
                      label_now[e[1]]=label
                      break
          
          checked[e[0]]=1
          checked[e[1]]=1
              
      
      not_checked=[i for i in checked if checked[i]==0]
      
      if not len(not_checked)==0:
          #inner loop 2, label the unlabeled vertices in the reverse order of labels in previous step
          for node in not_checked:                                                   
              #if checked[node]==0:
              checked[node]=1
              for label in range(3,max_label+1):
                  if not violate_constraints(G2, label_now, k, node, label):        
                      label_now[node]=label
                      break
                      
      max_degree = sorted([d for n, d in G2.degree()], reverse=True)[0]
      
      nparr=np.array(G2.degree())
      
      #to check if the current span is bigger than the previous span
      flag=0                                                                        
      for vertex, label in label_now.items():
          #if one vertex is labeled as 0, that means a label more than the previous span should have been needed
          if label == 0 and checked[vertex]==1:                                     
              flag=1
              break
      #flag=0 means the current span is not greater thab the previous span
      if flag==0:                                                                   
          #replace the previous step's dictionary with current step's dictionary
          label_init=label_now                                                      
          
      else:
          label_init=label_init
      
      #finding the maximum label in this step
      max_label=max(label_init.values())                                            
      #finding the maximum labeled vertex in this step
      for vertex, label in label_init.items():                                      
          if label == max_label:
              max_labeled=vertex
              break
      
      upper_bound=0
      upper_bound_new=0
      
      # finding the unaffected clique
      if q_now==q_old and q_now>=int(nx.number_of_nodes(G2)/2) and clique_changed_flag==0:
          clique_unchanged=q_old
          clique_diff=q_old-int(nx.number_of_nodes(G2)/2)
          clique_changed_flag=1

      # calculating upper bound    
      if q_now>=clique_unchanged:
          upper_bound=(k-1)*q_now+n+1-k
      else:
          upper_bound=(k-1)*clique_unchanged+n+1-k
      
      if q_now==q_old:
          clique_same_count+=1
      else:
          clique_same_count=1
      
      #printing for debugging and testing purposes
      file.write("Maximum labeled vertex: "+str(max_labeled)+"\n")                    
      file.write("Maximum Degree= "+str(max_degree)+"\n")
      file.write("Maximum= "+str(max_label)+"\n")
      file.write("Clique= "+str(q_now)+"\n")
      file.write("Upper Bound= "+str(upper_bound)+"\n")
      file.write("Clique_unchanged= "+ str(clique_unchanged)+"\n")
      file.write("\n")
      lst.append(max_label)
      lst2.append(upper_bound)
      if(len([e for e in G2.edges])%50==0):
        print(str(len([e for e in G2.edges]))+" "+"is done")

def main:
    # number of vertices
    n = input()
    # number of edges
    m = input()
    # random graph generating
    G=nx.gnm_random_graph(50,750)
    # minimum distance between two labeled vertices
    k = input()
    # file path,output will be written there
    filepath = input()
    file = open(filepath,'a')
    k_safe_labeling(n,m,k,file)
    

if __name__ == "__main__":
    main()
