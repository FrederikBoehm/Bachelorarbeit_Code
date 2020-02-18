
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# def convertToPng(data_files, x_column, y_column, graph_labels, x_label, y_label, output_file):
#     data_frames = []
#     for data_file in data_files:
#         data_frames.append(pd.read_csv(data_file))

#     plt.style.use('seaborn-whitegrid')
#     fig = plt.figure()
#     ax = plt.axes()
#     ax.axvline(x=185000, ymin=0, ymax=1, color='#808080')
#     for index, data_frame in enumerate(data_frames):
#         plt.plot(data_frame[x_column], data_frame[y_column], label=graph_labels[index])


#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()

#     plt.savefig(output_file, dpi=1000)

def convertToPng(data_files, x_column, y_columns, graph_labels, x_label, y_label, output_file, log_scale_x, log_scale_y):
    matplotlib.rcParams.update({'font.size': 14})
    data_frames = []
    for data_file in data_files:
        data_frames.append(pd.read_csv(data_file))

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    if log_scale_x:
        ax.set_xscale('log')
    if log_scale_y:
        ax.set_yscale('log')
    # ax.axvline(x=185000, ymin=0, ymax=1, color='#808080')
    for index, data_frame in enumerate(data_frames):
        plt.plot(data_frame[x_column], data_frame[y_columns[index]], label=graph_labels[index])


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    plt.savefig(output_file, dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings for the diagram.')
    parser.add_argument('--data_files', dest='data_files', type=str)
    parser.add_argument('--x_column', dest='x_column', type=str)
    parser.add_argument('--y_columns', dest='y_columns', type=str)
    parser.add_argument('--graph_labels', dest='graph_labels', type=str)
    parser.add_argument('--x_label', dest='x_label', type=str)
    parser.add_argument('--y_label', dest='y_label', type=str)
    parser.add_argument('--output_file', dest='output_file', type=str)
    parser.add_argument('--log_scale_x', dest='log_scale_x', action='store_true')
    parser.add_argument('--log_scale_y', dest='log_scale_y', action='store_true')
    args = parser.parse_args()
    if args.data_files and args.x_column and args.y_columns and args.graph_labels and args.x_label and args.y_label and args.output_file:
        data_files = args.data_files.split(';')
        graph_labels = args.graph_labels.split(';')
        y_columns = args.y_columns.split(';')
        convertToPng(data_files, args.x_column, y_columns, graph_labels, args.x_label, args.y_label, args.output_file, args.log_scale_x, args.log_scale_y)
    else:
        print(f'Not all parameters provided.')