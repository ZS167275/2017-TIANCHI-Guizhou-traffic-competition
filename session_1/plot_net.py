import pandas as pd
import matplotlib.pyplot as  plt
import networkx as nx

linktop=pd.read_csv('./gy_contest_link_top.txt',sep=';')

linklist=[]

for i in range(len(linktop)):

    temp=linktop['in_links'].astype('str').iloc[i]

    if temp.lower()=='nan':

        continue

    else:

        temp2=temp.split('#')

        for item in temp2:

            linklist.append((linktop['link_ID'].iloc[i],item))

for i in range(len(linktop)):

    temp=linktop['out_links'].astype('str').iloc[i]

    if temp.lower()=='nan':

        continue

    else:

        temp2=temp.split('#')

        for item in temp2:

            linklist.append((item,linktop['link_ID'].iloc[i]))

G=nx.DiGraph()

for node in set(linktop['link_ID']):

    G.add_node(node)

for edge in linklist:

    G.add_edge(*edge)

nx.draw(G,pos=nx.spring_layout(G),alpha=0.3)
plt.show()