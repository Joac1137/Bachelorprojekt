import pandas as pd
import matplotlib.pyplot as plt


def plot_from_csv():
    path_to_csv = 'C:\\Users\\joac1\\Downloads\\davis_southern_women_f_1.5_32_f_1.5.csv'
    df = pd.read_csv(path_to_csv)

    nodes_list = df.iloc[:,0]
    high_fixation_probabilities = df['High Degree']
    low_fixation_probabilities = df['Low Degree']
    centrality_fixation_probabilities = df['Centrality']
    temperature_fixation_probabilities = df['Temparature']
    random_fixation_probabilities = df['Random']

    plt.plot(nodes_list,high_fixation_probabilities, label='High Degree')
    plt.plot(nodes_list,low_fixation_probabilities, label='Low degree')
    plt.plot(nodes_list,centrality_fixation_probabilities, label='Centrality')
    plt.plot(nodes_list,temperature_fixation_probabilities, label='Temperature')
    plt.plot(nodes_list,random_fixation_probabilities, label='Random')

    plt.xlabel('Active Nodes')
    plt.ylabel('Fixation Probability')
    plt.title('Davis Southern Women')
    plt.legend()
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

if __name__ == "__main__":
    
    #plot_from_csv()
    plot_from_txt()