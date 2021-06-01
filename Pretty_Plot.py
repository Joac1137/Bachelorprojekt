import pandas as pd
import matplotlib.pyplot as plt


def plot_heuristic_comparison_from_csv():
    #path_to_csv = 'C:\\Users\\joac1\\Downloads\\davis_southern_women_f_1.5_32_f_1.5.csv'
    #path_to_csv = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Experiments\\heuristic_expriments_on_larger_graphs\\Erdos Renyi\\with vertex cover\\erdos_renyi_p_0_1_1_50.csv'
    path_to_csv = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Experiments\\heuristic_expriments_on_larger_graphs\\barabasi_albert_graph\\barabasi_albert_n50_m3_f_1_50.csv'
    path_to_csv = r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\barabasi_albert_graph\barabasi_albert_n50_m3_f_1_50.csv'
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
    plt.savefig(r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\barabasi_albert_graph\Final\barabasi_albert_n50_m3_f_1_50.csv'.replace('\\','\\\\') + ".png")

    plt.legend(loc=2, prop={'size': 12})
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
    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Circle_Graph_Experiments\\cycle_experiments5_g_size_10.txt'
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

    plt.xlabel('Active Nodes', fontsize = 12)
    plt.ylabel('Fixation Probability', fontsize = 12)
    plt.legend(loc=2, prop={'size': 12})
    plt.show()


def plot_circle_choosing_strategy_performance_txt():
    path_to_txt = 'C:\\Users\\joac1\\Documents\\Universitet\\6. Semester\\Bachelorprojekt\\Moran Process\\Circle_Graph_Experiments\\cycle_fitness_experiment\\cycle_experiments_f_[0.1, 0.2, 0.5, 1, 1.5, 10, 100]_g_size_50_setup_evenly_distributed.txt'
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
    plt.axis([0, 30, 0, 0.8])
    plt.legend(loc=2, prop={'size': 12})
    plt.show()


def compose_plots():
    path_to_vertex_csv = r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\Davis Southern Woman\davis_southern_women_vertex_f_1_32.csv'
    path_to_vertex_csv.replace('\\', '\\\\')



    path_to_csv_big = r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\Davis Southern Woman\davis_southern_women_f_1_32_f_1.csv'
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
    plt.savefig(r'C:\Users\joac1\Documents\Universitet\6. Semester\Bachelorprojekt\Moran Process\Experiments\heuristic_expriments_on_larger_graphs\Davis Southern Woman\Final\davis_southern_women_f_1_32_f_1.csv'.replace('\\','\\\\') + ".png")
    plt.show()



if __name__ == "__main__":

    plot_heuristic_comparison_from_csv()
    #plot_from_txt()
    #plot_complete_data()
    #plot_star_data()
    #plot_circle_choosing_strategies_txt()
    #plot_circle_choosing_strategy_performance_txt()
    #compose_plots()