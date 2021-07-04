import pandas as pd
import matplotlib.pyplot as plt

import Graphs
import Moran_Process


def plot_heuristic_comparison_from_csv():
    #path_to_csv = 'C:\\Users\\joac1\\Downloads\\davis_southern_women_f_1.5_32_f_1.5.csv'
    #path_to_csv = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Experiments\\heuristic_expriments_on_larger_graphs\\Erdos Renyi\\with vertex cover\\erdos_renyi_p_0_1_1_50.csv'
    path_to_csv = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Experiments\\heuristic_expriments_on_larger_graphs\\barabasi_albert_graph\\barabasi_albert_n50_m3_f_1_50.csv'
    path_to_csv = r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\barabasi_albert_graph\barabasi_albert_n50_m3_f_100_50.csv'
    path_to_csv.replace('\\','\\\\')

    df = pd.read_csv(path_to_csv)

    nodes_list = df.iloc[:,0]
    high_fixation_probabilities = df['High Degree']
    low_fixation_probabilities = df['Low Degree']
    centrality_fixation_probabilities = df['Centrality']
    temperature_fixation_probabilities = df['Temparature']
    random_fixation_probabilities = df['Random']
    vertex_fixation_probabilities = df['Vertex Cover']

    plt.plot(nodes_list,high_fixation_probabilities, label='High Degree',color='b', marker='.', markersize = 4)
    plt.plot(nodes_list,low_fixation_probabilities, label='Low degree', color='y', marker='v', markersize = 4)
    plt.plot(nodes_list,centrality_fixation_probabilities, label='Centrality', color='g', marker='^', markersize = 4)
    plt.plot(nodes_list,temperature_fixation_probabilities, label='Temperature', color='r', marker='s', markersize = 4)
    plt.plot(nodes_list,random_fixation_probabilities, label='Random', color='purple', marker='*', markersize = 4)
    plt.plot(nodes_list,vertex_fixation_probabilities, label='Vertex Cover', color='brown', marker='D', markersize = 4)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.title('Erdós Rényi', fontsize = 14)
    plt.legend(loc=2, prop={'size': 9})
    plt.savefig(r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\barabasi_albert_graph\Final\barabasi_albert_n50_m3_f_100_50.csv'.replace('\\','\\\\') + ".png")

    plt.show()

def plot_star_data():
    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Star_Graph_Experiments\\star_experiments0.01_g_size_201.txt'
    df001 = pd.read_csv(path_to_txt, header = None)

    active_list = list(range(0,200))
    f_001_fixation_probabilities = df001.loc[1].values.tolist()
    f_001_fixation_probabilities[0] = float(f_001_fixation_probabilities[0][24:])


    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Star_Graph_Experiments\\star_experiments0.05_g_size_201.txt'
    df005 = pd.read_csv(path_to_txt, header = None)

    f_005_fixation_probabilities = df005.loc[1].values.tolist()
    f_005_fixation_probabilities[0] = float(f_005_fixation_probabilities[0][24:])
    #f_005_fixation_probabilities = [float(x) for x in f_005_fixation_probabilities[0][24:].split(',')]


    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Star_Graph_Experiments\\star_experiments0.1_g_size_201.txt'
    df01 = pd.read_csv(path_to_txt, header = None)

    f_01_fixation_probabilities = df01.loc[1].values.tolist()
    f_01_fixation_probabilities[0] = float(f_01_fixation_probabilities[0][24:])
    #f_01_fixation_probabilities = [float(x) for x in f_01_fixation_probabilities[0][24:].split(',')]


    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Star_Graph_Experiments\\star_experiments0.2_g_size_201.txt'
    df02 = pd.read_csv(path_to_txt, header = None)

    f_02_fixation_probabilities = df02.loc[1].values.tolist()
    f_02_fixation_probabilities[0] = float(f_02_fixation_probabilities[0][24:])
    #f_02_fixation_probabilities = [float(x) for x in f_02_fixation_probabilities[0][24:].split(',')]


    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Star_Graph_Experiments\\star_experiments0.3_g_size_201.txt'
    df03 = pd.read_csv(path_to_txt, header = None)

    f_03_fixation_probabilities = df03.loc[1].values.tolist()
    f_03_fixation_probabilities[0] = float(f_03_fixation_probabilities[0][24:])
    #f_03_fixation_probabilities = [float(x) for x in f_03_fixation_probabilities[0][24:].split(',')]

    plt.plot(active_list,f_001_fixation_probabilities, label='0.01',color='b', marker='.',markersize = 7, markevery=10)
    plt.plot(active_list,f_005_fixation_probabilities, label='0.05', color='y', marker='v', markersize = 7, markevery=10)
    plt.plot(active_list,f_01_fixation_probabilities, label='0.1', color='g', marker='^', markersize = 7, markevery=10)
    plt.plot(active_list,f_02_fixation_probabilities, label='0.2', color='r', marker='s', markersize = 7, markevery=10)
    plt.plot(active_list,f_03_fixation_probabilities, label='0.3', color='purple', marker='*', markersize = 7, markevery=10)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    plt.legend(loc=2, prop={'size': 12})
    plt.show()


def plot_complete_data():
    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Complete_Graph_Experiments\\complete_experiments0.01_g_size_200.txt'
    df001 = pd.read_csv(path_to_txt, header = None)

    active_list = list(range(0,201))
    f_001_fixation_probabilities = df001.loc[1].values.tolist()
    f_001_fixation_probabilities[0] = float(f_001_fixation_probabilities[0][24:])


    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Complete_Graph_Experiments\\complete_experiments0.05_g_size_200.txt'
    df005 = pd.read_csv(path_to_txt, header = None)

    f_005_fixation_probabilities = df005.loc[1].values.tolist()
    f_005_fixation_probabilities[0] = float(f_005_fixation_probabilities[0][24:])
    #f_005_fixation_probabilities = [float(x) for x in f_005_fixation_probabilities[0][24:].split(',')]


    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Complete_Graph_Experiments\\complete_experiments0.1_g_size_200.txt'
    df01 = pd.read_csv(path_to_txt, header = None)

    f_01_fixation_probabilities = df01.loc[1].values.tolist()
    f_01_fixation_probabilities[0] = float(f_01_fixation_probabilities[0][24:])
    #f_01_fixation_probabilities = [float(x) for x in f_01_fixation_probabilities[0][24:].split(',')]


    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Complete_Graph_Experiments\\complete_experiments0.2_g_size_200.txt'
    df02 = pd.read_csv(path_to_txt, header = None)

    f_02_fixation_probabilities = df02.loc[1].values.tolist()
    f_02_fixation_probabilities[0] = float(f_02_fixation_probabilities[0][24:])
    #f_02_fixation_probabilities = [float(x) for x in f_02_fixation_probabilities[0][24:].split(',')]


    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Complete_Graph_Experiments\\complete_experiments0.3_g_size_200.txt'
    df03 = pd.read_csv(path_to_txt, header = None)

    f_03_fixation_probabilities = df03.loc[1].values.tolist()
    f_03_fixation_probabilities[0] = float(f_03_fixation_probabilities[0][24:])
    #f_03_fixation_probabilities = [float(x) for x in f_03_fixation_probabilities[0][24:].split(',')]

    plt.plot(active_list,f_001_fixation_probabilities, label='0.01',color='b', marker='.',markersize = 7, markevery=10)
    plt.plot(active_list,f_005_fixation_probabilities, label='0.05', color='y', marker='v', markersize = 7, markevery=10)
    plt.plot(active_list,f_01_fixation_probabilities, label='0.1', color='g', marker='^', markersize = 7, markevery=10)
    plt.plot(active_list,f_02_fixation_probabilities, label='0.2', color='r', marker='s', markersize = 7, markevery=10)
    plt.plot(active_list,f_03_fixation_probabilities, label='0.3', color='purple', marker='*', markersize = 7, markevery=10)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    plt.legend(loc=2, prop={'size': 12})
    plt.show()


def plot_from_txt():
    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Circle_Graph_Experiments\\cycle_experiments20_g_size_4.txt'
    df = pd.read_csv(path_to_txt, delimiter = "\t")

    active_list = df.loc[0]
    active_list = [int(x) for x in active_list[0][8:].split(',')]

    continous_list = df.loc[1].values.tolist()
    continous_list = [float(x) for x in continous_list[0][24:].split(',')]

    evenly_distributed_list = df.loc[4].values.tolist()
    evenly_distributed_list = [float(x) for x in evenly_distributed_list[0][24:].split(',')]

    every_other_list = df.loc[7].values.tolist()
    every_other_list = [float(x) for x in every_other_list[0][24:].split(',')]

    plt.plot(active_list,continous_list, label='Continuous')
    plt.plot(active_list,evenly_distributed_list, label='Evenly Distributed')
    plt.plot(active_list,every_other_list, label='Every other')

    plt.xlabel('Active Nodes')
    plt.ylabel('Fixation Probability')
    plt.title('Cycle Graphs')
    plt.legend()
    plt.show()


def plot_circle_choosing_strategies_txt():
    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Circle_Graph_Experiments\\cycle_experiments5_g_size_15.txt'
    df = pd.read_csv(path_to_txt, delimiter = "\t")

    active_list = df.loc[0]
    active_list = [int(x) for x in active_list[0][8:].split(',')]

    continous_list = df.loc[1].values.tolist()
    continous_list = [float(x) for x in continous_list[0][24:].split(',')]

    evenly_distributed_list = df.loc[4].values.tolist()
    evenly_distributed_list = [float(x) for x in evenly_distributed_list[0][24:].split(',')]

    every_other_list = df.loc[7].values.tolist()
    every_other_list = [float(x) for x in every_other_list[0][24:].split(',')]

    plt.plot(active_list,continous_list, label='Continuous',color='b', marker='.',markersize = 7, markevery=2)
    plt.plot(active_list,evenly_distributed_list, label='Evenly Distributed', color='y', marker='v', markersize = 7, markevery=2)
    plt.plot(active_list,every_other_list, label='Every other',color='g', marker='^', markersize = 7, markevery=2)

    plt.xlabel('Active Nodes', fontsize = 14)
    plt.ylabel('Fixation Probability', fontsize = 14)
    plt.legend(loc=2, prop={'size': 15})
    plt.savefig('C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Circle_Graph_Experiments\\Choosing Strategy Performance\\cycle_experiments5_g_size_15' + '.png')
    plt.show()


def plot_circle_choosing_strategy_performance_txt():
    path_to_txt = r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Circle_Graph_Experiments\cycle_fitness_experiment_new3\cycle_experiments_f_[0.1, 0.2, 0.5, 1, 1.5, 10, 100]_g_size_50_setup_evenly_distributed.txt'
    path_to_txt.replace('\\','\\\\')
    df = pd.read_csv(path_to_txt, header = None)

    active_list = list(range(0,51))

    f_01 = df.loc[0].values.tolist()
    f_01[0] = float(f_01[0][24:])

    f_02 = df.loc[3].values.tolist()
    f_02[0] = float(f_02[0][24:])

    f_05 = df.loc[6].values.tolist()
    f_05[0] = float(f_05[0][24:])

    f_1 = df.loc[9].values.tolist()
    f_1[0] = float(f_1[0][24:])

    f_15 = df.loc[12].values.tolist()
    f_15[0] = float(f_15[0][24:])

    f_10 = df.loc[15].values.tolist()
    f_10[0] = float(f_10[0][24:])

    f_100 = df.loc[18].values.tolist()
    f_100[0] = float(f_100[0][24:])

    plt.plot(active_list,f_01, label='0.1',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,f_02, label='0.2', color='y', marker='v', markersize = 7, markevery=5)
    plt.plot(active_list,f_05, label='0.5', color='g', marker='^', markersize = 7, markevery=5)
    plt.plot(active_list,f_1, label='1', color='r', marker='s', markersize = 7, markevery=5)
    plt.plot(active_list,f_15, label='1.5', color='grey', marker='>', markersize = 7, markevery=5)
    plt.plot(active_list,f_10, label='10', color='purple', marker='*', markersize = 7, markevery=5)
    plt.plot(active_list,f_100, label='100', color='k', marker='x', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.axis([0, 50, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()


def compose_plots():
    path_to_vertex_csv = r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\Florentine Family\florentine_families_vertex_f_1_15.csv'
    path_to_vertex_csv.replace('\\', '\\\\')



    path_to_csv_big = r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\Florentine Family\florentine_families_f_1_15_f_1.csv'
    path_to_csv_big.replace('\\', '\\\\')

    df_vertex = pd.read_csv(path_to_vertex_csv)
    df_big = pd.read_csv(path_to_csv_big)

    frames = [df_big,df_vertex]

    df = pd.concat(frames)

    nodes_list = df.iloc[:,0]
    high_fixation_probabilities = df['High Degree']
    low_fixation_probabilities = df['Low Degree']
    centrality_fixation_probabilities = df['Centrality']
    temperature_fixation_probabilities = df['Temparature']
    random_fixation_probabilities = df['Random']
    vertex_fixation_probabilities = df['Vertex Cover']

    plt.plot(nodes_list,high_fixation_probabilities, label='High Degree',color='b', marker='.', markersize = 4)
    plt.plot(nodes_list,low_fixation_probabilities, label='Low degree', color='y', marker='v', markersize = 4)
    plt.plot(nodes_list,centrality_fixation_probabilities, label='Centrality', color='g', marker='^', markersize = 4)
    plt.plot(nodes_list,temperature_fixation_probabilities, label='Temperature', color='r', marker='s', markersize = 4)
    plt.plot(nodes_list,random_fixation_probabilities, label='Random', color='purple', marker='*', markersize = 4)
    plt.plot(nodes_list,vertex_fixation_probabilities, label='Vertex Cover', color='brown', marker='D', markersize = 4)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.title('Erdós Rényi', fontsize = 14)
    plt.legend(loc=2, prop={'size': 12})
    plt.savefig(r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\Florentine Family\Final\florentine_families_f_1_15_f_1.csv'.replace('\\','\\\\') + ".png")
    plt.show()


def compare_individual_choosing_strategies():
    path_to_continuous = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Circle_Graph_Experiments\\cycle_fitness_experiment\\cycle_experiments_f_[0.1, 0.2, 0.5, 1, 1.5, 10, 100]_g_size_50_setup_continuous.txt'
    df_continuous = pd.read_csv(path_to_continuous, header = None)

    path_to_dist = r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Circle_Graph_Experiments\cycle_fitness_experiment_new3\cycle_experiments_f_[0.1, 0.2, 0.5, 1, 1.5, 10, 100]_g_size_50_setup_evenly_distributed.txt'
    path_to_dist.replace('\\','\\\\')
    df_dist = pd.read_csv(path_to_dist, header = None)

    active_list = list(range(0,51))

    f_con_01 = df_continuous.loc[0].values.tolist()
    f_con_01[0] = float(f_con_01[0][24:])

    f_dist_01 = df_dist.loc[0].values.tolist()
    f_dist_01[0] = float(f_dist_01[0][24:])

    plt.plot(active_list,f_con_01, label='Continuous',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,f_dist_01, label='Evenly Distributed', color='y', marker='v', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()



    f_con_02 = df_continuous.loc[3].values.tolist()
    f_con_02[0] = float(f_con_02[0][24:])

    f_dist_02 = df_dist.loc[3].values.tolist()
    f_dist_02[0] = float(f_dist_02[0][24:])

    plt.plot(active_list,f_con_02, label='Continuous',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,f_dist_02, label='Evenly Distributed', color='y', marker='v', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()


    f_con_05 = df_continuous.loc[6].values.tolist()
    f_con_05[0] = float(f_con_05[0][24:])

    f_dist_05 = df_dist.loc[6].values.tolist()
    f_dist_05[0] = float(f_dist_05[0][24:])

    plt.plot(active_list,f_con_05, label='Continuous',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,f_dist_05, label='Evenly Distributed', color='y', marker='v', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()



    f_con_1 = df_continuous.loc[9].values.tolist()
    f_con_1[0] = float(f_con_1[0][24:])

    f_dist_1 = df_dist.loc[9].values.tolist()
    f_dist_1[0] = float(f_dist_1[0][24:])

    plt.plot(active_list,f_con_1, label='Continuous',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,f_dist_1, label='Evenly Distributed', color='y', marker='v', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()



    f_con_15 = df_continuous.loc[12].values.tolist()
    f_con_15[0] = float(f_con_15[0][24:])

    f_dist_15 = df_dist.loc[12].values.tolist()
    f_dist_15[0] = float(f_dist_15[0][24:])

    plt.plot(active_list,f_con_15, label='Continuous',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,f_dist_15, label='Evenly Distributed', color='y', marker='v', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()


    f_con_10 = df_continuous.loc[15].values.tolist()
    f_con_10[0] = float(f_con_10[0][24:])

    f_dist_10 = df_dist.loc[15].values.tolist()
    f_dist_10[0] = float(f_dist_10[0][24:])

    plt.plot(active_list,f_con_10, label='Continuous',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,f_dist_10, label='Evenly Distributed', color='y', marker='v', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()



    f_con_100 = df_continuous.loc[18].values.tolist()
    f_con_100[0] = float(f_con_100[0][24:])

    f_dist_100 = df_dist.loc[18].values.tolist()
    f_dist_100[0] = float(f_dist_100[0][24:])

    plt.plot(active_list,f_con_100, label='Continuous',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,f_dist_100, label='Evenly Distributed', color='y', marker='v', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    #plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()


    plt.plot(active_list,[x-y for x,y in zip(f_dist_01,f_con_01)], label='0.1',color='b', marker='.',markersize = 7, markevery=5)
    plt.plot(active_list,[x-y for x,y in zip(f_dist_02,f_con_02)], label='0.2', color='y', marker='v', markersize = 7, markevery=5)
    plt.plot(active_list,[x-y for x,y in zip(f_dist_05,f_con_05)], label='0.5', color='g', marker='^', markersize = 7, markevery=5)
    plt.plot(active_list,[x-y for x,y in zip(f_dist_1,f_con_1)], label='1', color='r', marker='s', markersize = 7, markevery=5)
    plt.plot(active_list,[x-y for x,y in zip(f_dist_15,f_con_15)], label='1.5', color='grey', marker='>', markersize = 7, markevery=5)
    plt.plot(active_list,[x-y for x,y in zip(f_dist_10,f_con_10)], label='10', color='purple', marker='*', markersize = 7, markevery=5)
    plt.plot(active_list,[x-y for x,y in zip(f_dist_100,f_con_100)], label='100', color='k', marker='x', markersize = 7, markevery=5)

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability Difference', fontsize = 12)
    #plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=1, prop={'size': 12})
    plt.show()




if __name__ == "__main__":

    #plot_heuristic_comparison_from_csv()
    #plot_from_txt()
    #plot_complete_data()
    #plot_star_data()
    #plot_circle_choosing_strategies_txt()
    #plot_circle_choosing_strategy_performance_txt()
    #compose_plots()
    #compare_individual_choosing_strategies()

    G = Graphs.create_complete_graph(3)
    G = Graphs.initialize_nodes_as_resident(G)
    Moran_Process.numeric_fixation_probability(G,1)

    """test = [0.04715050718143668634363407932141853962093591690063, 0.04738556841259058960424965789570705965161323547363, 0.04762051876233320407694193932002235669642686843872, 0.04785535816644663487107180799284833483397960662842, 0.04809008657379824913657984097881126217544078826904, 0.04832470394132138552523159091833804268389940261841, 0.04855921023348165610489246546421782113611698150635, 0.04879360542362243952085876230739813763648271560669, 0.04902788949245869692949995055641920771449804306030, 0.04926206243040594928705999677731597330421209335327, 0.04949612423293968388460228879921487532556056976318, 0.04973007490498514149290798513902700506150722503662, 0.04996391445391484137372728469017602037638425827026, 0.05019764289806728785325873332112678326666355133057, 0.05043126025974604964563496878326986916363239288330, 0.05066476656599604860842234188567090313881635665894, 0.05089816185187383962285068150777078699320554733276, 0.05113144615556593219896086566222948022186756134033, 0.05136461952157841959198947279219282791018486022949, 0.05159768199866181237478457433098810724914073944092, 0.05183063364041510384661037846854014787822961807251, 0.05206347450582923808148905209236545488238334655762, 0.05229620465498319159536322331405244767665863037109, 0.05252882415572204116704924103942175861448049545288, 0.05276133307650700265956089651808724738657474517822, 0.05299373149179548159182218114438001066446304321289, 0.05322601947866854865676344843450351618230342864990, 0.05345819711850746669412615119654219597578048706055, 0.05369026449240967002740632096902118064463138580322, 0.05392222168878316179707610444893362000584602355957, 0.05415406879630176723017243034519196953624486923218, 0.05438580590786742940956344227743102237582206726074, 0.05461743311640232523274107734323479235172271728516, 0.05484895052061516257380091587947390507906675338745, 0.05508035821853054375685232457726669963449239730835, 0.05531165631391185555365552772855153307318687438965, 0.05554284490826220249681810514630342368036508560181, 0.05577392410965470448402925285336095839738845825195, 0.05600489402511826808606087979569565504789352416992, 0.05623575476234105724460121678021096158772706985474, 0.05646650643491748361446624926429649349302053451538, 0.05669714915517370945607922294584568589925765991211, 0.05692768303709970501014225874314433895051479339600, 0.05715810819727838720805124239632277749478816986084, 0.05738842475178736884933172746059426572173833847046, 0.05761863282108196232522701052403135690838098526001, 0.05784873252322554615378891185173415578901767730713, 0.05807872398046210610234041382682335097342729568481, 0.05830860731412338543444562333206704352051019668579, 0.05853838264824946790998438927999814040958881378174, 0.05876805010604070622637351561934337951242923736572, 0.05899760981357996675322752366810163948684930801392, 0.05922706189558776812553730906074633821845054626465, 0.05945640648077842038699003524016006849706172943115, 0.05968564369487542459191686816666333470493555068970, 0.05991477366672523507951098054036265239119529724121, 0.06014379652568831863268528081789554562419652938843, 0.06037271240076442363475806018868752289563417434692, 0.06060152142313154910446826306724688038229942321777, 0.06083022372225821428060044127050787210464477539062, 0.06105881943060053468341763505122798960655927658081, 0.06128730867891034328476251857864554040133953094482, 0.06151569160021582577391185964188480284065008163452, 0.06174396832700741050148351973803073633462190628052, 0.06197213899165660011547629437700379639863967895508, 0.06220020372782358170082872561579279135912656784058, 0.06242816266912667305666317929535580333322286605835, 0.06265601595016569158591579480344080366194248199463, 0.06288376370426779216060708677105139940977096557617, 0.06311140606545946607486285984123242087662220001221, 0.06333894316967908499904638119915034621953964233398, 0.06356637515170075747317213199494290165603160858154, 0.06379370214540153183335746689408551901578903198242, 0.06402092428715203120681564996630186215043067932129, 0.06424804171142457775989242918512900359928607940674, 0.06447505455382405992192929033990367315709590911865, 0.06470196295004033981701496713867527432739734649658, 0.06492876703570847618518513399976654909551143646240, 0.06515546694631349500248518324951874092221260070801, 0.06538206281671805897293126008662511594593524932861, 0.06560855478463843704073354956562980078160762786865, 0.06583494298370122510544177885094541124999523162842, 0.06606122755088023046354805956070777028799057006836, 0.06628740862141527245832151038484880700707435607910, 0.06651348633155626521862302524823462590575218200684, 0.06673946081636068783371484869348932988941669464111, 0.06696533221123955603726329854907817207276821136475, 0.06719110065129051123644643439547508023679256439209, 0.06741676627337508942439114889566553756594657897949, 0.06764232921180590718535086125484667718410491943359, 0.06786778960281114603247942795860581099987030029297, 0.06809314758045605098235597552047693170607089996338, 0.06831840328046033528064384654499008320271968841553, 0.06854355683698522272440101232859888114035129547119, 0.06876860838609018256040172900611651130020618438721, 0.06899355806136610314016621714472421444952487945557, 0.06921840599779678637748503433613223023712635040283, 0.06944315232949480054536195439141010865569114685059, 0.06966779719151960925671573932049795985221862792969, 0.06989234071719549468149068616185104474425315856934, 0.07011678304110147141869902043254114687442779541016, 0.07034112429677392974713967532807146199047565460205, 0.07056536461712691055137014473075396381318569183350, 0.07078950413608842140433807799126952886581420898438, 0.07101354298666642805848425723524997010827064514160, 0.07123748130225224239886472332727862522006034851074, 0.07146131921536139075534777020948240533471107482910, 0.07168505685844192565348009793524397537112236022949, 0.07190869436516378332413523821742273867130279541016, 0.07213223186645406626027465790684800595045089721680, 0.07235566949515362389355743744090432301163673400879, 0.07257900738244939253807075374425039626657962799072, 0.07280224566005435993254479853931115940213203430176, 0.07302538445937459266055213902291143313050270080566, 0.07324842391278120479203295190018252469599246978760, 0.07347136414937130433067125068191671743988990783691, 0.07369420530133348468115883633799967356026172637939, 0.07391694749785346352233261768560623750090599060059, 0.07413959087042586160176682597011676989495754241943, 0.07436213554879846088230266332175233401358127593994, 0.07458458166171028080881910682364832609891891479492, 0.07480692934050650610977584165084408596158027648926, 0.07502917871328147747522763211236451752483844757080, 0.07525132990917278441589388648935710079967975616455, 0.07547338305865143592843224951138836331665515899658, 0.07569533828853165779992906436746125109493732452393, 0.07591719572770043444887022587863611988723278045654, 0.07613895550446125870536207003169693052768707275391, 0.07636061774723118533714227851305622607469558715820, 0.07658218258297724845284903949504951015114784240723, 0.07680365013957987652126746525027556344866752624512, 0.07702502054384062490921536436871974729001522064209, 0.07724629392268611560723456932464614510536193847656, 0.07746747040339602152769771237217355519533157348633, 0.07768855011191948267690321472400682978332042694092, 0.07790953317413905343524760382933891378343105316162, 0.07813041971632400661817996478930581361055374145508, 0.07835120986514204632911173575848806649446487426758, 0.07857190374560744927645572488472680561244487762451, 0.07879250148260227848417969198635546490550041198730, 0.07901300320078731564965579536874429322779178619385, 0.07923340902465415835909112729495973326265811920166, 0.07945371907931859933871265866400790400803089141846, 0.07967393348910084549530097319802735000848770141602, 0.07989405237838835549535332347659277729690074920654, 0.08011407586938339153714849771859007887542247772217, 0.08033400408686292015758567686134483665227890014648, 0.08055383715337054162386465350209618918597698211670, 0.08077357519258604456879879762709606438875198364258, 0.08099321832667465137500784067015047185122966766357, 0.08121276667808827176564534511271631345152854919434, 0.08143222036927695584029862629904528148472309112549, 0.08165157952220485071403999199901591055095195770264, 0.08187084425876711701874910431797616183757781982422, 0.08209001470019637591857275538131943903863430023193, 0.08230909096710667394170002353348536416888236999512, 0.08252807318146154758053967270825523883104324340820, 0.08274696146404005026031569514088914729654788970947, 0.08296575593429188211214153625405742786824703216553, 0.08318445671396654250706603761500446125864982604980, 0.08340306392211821462367993262887466698884963989258, 0.08362157767806095709506308821801212616264820098877, 0.08383999810205694513953744717582594603300094604492, 0.08405832531282275699169304061797447502613067626953, 0.08427655943058409704349287494551390409469604492188, 0.08449470057246999399325204649358056485652923583984, 0.08471274885773577567604775140353012830018997192383, 0.08493070440498154405339903405547374859452247619629, 0.08514856733158647494885684636756195686757564544678, 0.08536633775609843433063161910467897541821002960205, 0.08558401579502318556880169353462406434118747711182, 0.08580160156621696732059945134096778929233551025391, 0.08601909518755167238790448891450068913400173187256, 0.08623649677482841946485336848127190023660659790039, 0.08645380644446819251580649279276258312165737152100, 0.08667102431361649317320683394427760504186153411865, 0.08688815049772585785703427063708659261465072631836, 0.08710518511317029421991975368655403144657611846924, 0.08732212827478859351604256744394660927355289459229, 0.08753898009913292643791038472045329399406909942627, 0.08775574070064870446650218127615517005324363708496, 0.08797241019321574984068945468607125803828239440918, 0.08818898869344456159513612192313303239643573760986, 0.08840547631464806821455226781836245208978652954102, 0.08862187317181835499724229521234519779682159423828, 0.08883817937705824896443829175041173584759235382080, 0.08905439504709235232216002486893557943403720855713, 0.08927052029220512718854507738797110505402088165283, 0.08948655522747203661104720140428980812430381774902, 0.08970249996594048536024956774781458079814910888672, 0.08991835462015868030150045342452358454465866088867, 0.09013411930316057862899725705574383027851581573486, 0.09034979412572403945080878884255071170628070831299, 0.09056537920151093434828482031662133522331714630127, 0.09078087464168142772802383433372597210109233856201, 0.09099628055761149747482363636663649231195449829102, 0.09121159706109957521213971176621271297335624694824, 0.09142682426266837691919420194608392193913459777832, 0.09164196227578676190450579497337457723915576934814, 0.09185701120848806244811868282340583391487598419189]
    number = (test[20]-test[0]) - (test[50] - test[20])
    print(number)"""


