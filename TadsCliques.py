#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Class   :
# @description:
# @Time    : 2022/8/8 下午10:33
# @Author  : Duan Ran


import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)


def mcl(data, inflation_param, max_iterations):
    """
    Perform Markov Clustering on the input data

    :param data: A 2-D array-like object containing the data to be clustered
    :param inflation_param: The inflation parameter to use during the clustering process
    :param max_iterations: The maximum number of iterations to perform before stopping
    :return: A list of clusters, where each cluster is a list of indices into the input data
    """
    # Convert the input data to a NumPy array
    data = np.array(data)

    # Normalize the input data so that each row sums to 1
    data /= np.sum(data, axis=1, keepdims=True)

    # Initialize the similarity matrix
    similarity_matrix = np.copy(data)

    # Perform the MCL algorithm
    for i in range(max_iterations):
        # Raise the similarity matrix to the power of the inflation parameter
        similarity_matrix = np.linalg.matrix_power(similarity_matrix, inflation_param)

        # Normalize the rows of the similarity matrix so that they sum to 1
        row_sums = np.sum(similarity_matrix, axis=1, keepdims=True)
        similarity_matrix /= row_sums

    # Find the clusters by thresholding the similarity matrix
    clusters = []
    visited = np.zeros(data.shape[0], dtype=bool)
    for i in range(data.shape[0]):
        if not visited[i]:
            cluster = np.where(similarity_matrix[i, :] > 0)[0]
            visited[cluster] = True
            clusters.append(cluster)

    return clusters


def expand_matrix(orign_mat, probability_mat, power):
    expand_mat = orign_mat
    for i in range(power):
        expand_mat = expand_mat @ expand_mat
    return expand_mat


def inflate(expand_mat, inflation):
    inflate_mat = expand_mat
    inflate_mat = np.power(inflate_mat, inflation)
    inflate_column_sum = np.sum(inflate_mat, axis=0)
    inflate_mat_result = inflate_mat / inflate_column_sum
    return inflate_mat_result


def markov_cluster(adjacency_mat, inflation, power):
    column_sum = np.sum(adjacency_mat, axis=0)
    M1 = adjacency_mat / column_sum
    flag = True
    diedai = 0
    while flag:
        diedai +=1
        M2 = expand_matrix(M1, M1, power)
        M1 = inflate(M2, inflation)
        if (M1 == M2).all():
            flag = False
        if diedai == 100:
            print('do convergence')
            flag = False
    return M1


def mctad_result_format(result_path, threshold):
    print('run')
    result_list = []
    for index in range(1,23):
        hold_list = []
        filepath = "{0}/chr{1}_min_score.txt".format(result_path, index)
        valuepath = "{0}/chr{1}_min_value.txt".format(result_path, index)
        reloca = np.loadtxt(filepath, dtype=np.int)
        rescore = np.loadtxt(valuepath)
        for index in range(len(rescore)):
            if rescore[index] < threshold:
                hold_list.append(reloca[index])
        print('lens of mctad:{0}'.format(len(hold_list)))
        result_list.append(hold_list)

    return result_list


def get_background_true(tad_one, tad_two, dense_mat):
    r1 = int(pow(pow(tad_two[0] - tad_one[1], 2) + pow(tad_two[0] - tad_one[1], 2), 0.5)/2)
    r2 = int(pow(pow(tad_two[1] - tad_one[0], 2) + pow(tad_two[1] - tad_one[0], 2), 0.5)/2)
    diag_sum = 0
    diag_area = 0
    for diag in range(r1, r2+1):
        diag_sum += np.sum(np.diagonal(dense_mat, offset=diag))
        diag_area += np.size(np.diagonal(dense_mat, offset=diag))
    exp_count = diag_sum/diag_area

    count_tad_ij = np.sum(dense_mat[tad_two[0]:tad_two[1], tad_two[0]:tad_two[1]], dtype=np.int)
    count_area = (tad_two[0] - tad_one[0])*(tad_two[1] - tad_one[1])
    true_count = count_tad_ij/count_area
    count_tre = 0
    p_value = 0
    if true_count > exp_count:
        p_value = 1
    return p_value, exp_count, true_count


def get_cliques_result():
    dense_mat = np.loadtxt('/mnt/disk1/duanran/GM12878dense/GM128782')
    celline = 'GM12878'
    mctad_result_list = mctad_result_format('/home/rduan/utad/algorithm/{0}re'.format(celline), 500)
    chr2 = mctad_result_list[1]
    print(mctad_result_list[1])
    for vvv in mctad_result_list[1]:
        print(vvv)
    tad_list = []
    begin = 0
    for vv in chr2:
        end = int(vv)
        tad_list.append(np.array([begin, end]))
        begin = end
    # print(hy_N)
    count_tru = 0
    p_value_list = []
    graph_list = []
    for tad_index_1 in range(len(tad_list)):
        for tad_index_2 in range(tad_index_1+1,len(tad_list)):
            p_value, exp_count, true_count = get_background_true(tad_list[tad_index_1],
                                                                 tad_list[tad_index_2], dense_mat)
            if p_value == 1:
                graph_list.append(np.array([tad_index_1, tad_index_2, true_count-exp_count]))

    fin_list = graph_list

    first_list = []
    secend_list = []
    value_list = []
    for node in fin_list:
        first_list.append(int(node[0]))
        secend_list.append(int(node[1]))
        value_list.append(node[2])

    all_result = []
    all_result.append(first_list)
    all_result.append(secend_list)
    all_result.append(value_list)
    all_result = np.array(all_result)
    re = pd.DataFrame({'Source': all_result[0], 'Target': all_result[1], 'Weight':all_result[2]}, dtype=np.int64)
    # re.to_csv('./TadsCliques.csv',sep=' ',index=0)

    # build adjacent graph
    max1 = max(re['Source'])
    max2 = max(re['Target'])
    max3 = max(max1, max2)+1

    ad_graph = np.zeros((max3, max3))
    for i, row in re.iterrows():
        ad_graph[int(row['Source']), int(row['Target'])] = row['Weight']
    ad_graph = ad_graph+ad_graph.T
    np.fill_diagonal(ad_graph, 1)
    rr = markov_cluster(ad_graph, 1.6, 1)
    # np.savetxt('./TadsCliques_result.txt', rr, fmt='%d')
    co = 0
    tad_cliques_result = []
    for lin in rr:
        clists = []
        for index in range(len(lin)):
            if lin[index] > 0:
                clists.append(index)
        if len(clists) > 0:
            tad_cliques_result.append(clists)
            co += 1
    return dense_mat, tad_cliques_result, tad_list


def matrix_demo(input_matrix):
    dense_mat = input_matrix
    diag_mat = np.ones(shape=dense_mat.shape)
    dense_mat = dense_mat + diag_mat
    print(dense_mat)
    offset_flage = 0
    for offset_value in range(np.shape(diag_mat)[0]-1):
        diag_list = np.sort(dense_mat.diagonal(offset=offset_value))
        begin = int(np.size(diag_list)*0.1)
        end = int(np.size(diag_list)*0.9)
        fvalue = int(np.average(diag_list[begin:end]))

        for index2 in range(len(diag_list)):
            diag_mat[index2,index2+offset_flage] = fvalue

        offset_flage += 1

    print(diag_mat)
    print(dense_mat)
    re = np.triu(dense_mat/diag_mat)
    re = re + re.T - np.diag(np.diag(re))
    return re


def get_cliques_result_nmat(nmat, result_list):
    celline = 'GM12878'
    dense_mat = nmat
    mctad_result_list = result_list
    chr2 = mctad_result_list[1]
    print(mctad_result_list[1])
    tad_list = []
    begin = 0
    for vv in chr2:
        end = int(vv)
        tad_list.append(np.array([begin, end]))
        begin = end
    count_tru = 0
    p_value_list = []
    graph_list = []
    for tad_index_1 in range(len(tad_list)):
        for tad_index_2 in range(tad_index_1+1, len(tad_list)):
            p_value, exp_count, true_count = get_background_true(tad_list[tad_index_1],
                                                                 tad_list[tad_index_2], dense_mat)
            if p_value == 1:
                graph_list.append(np.array([tad_index_1, tad_index_2, true_count-exp_count]))

    fin_list = graph_list

    first_list = []
    secend_list = []
    value_list = []
    for node in fin_list:
        first_list.append(int(node[0]))
        secend_list.append(int(node[1]))
        value_list.append(node[2])

    all_result = []
    all_result.append(first_list)
    all_result.append(secend_list)
    all_result.append(value_list)
    all_result = np.array(all_result)
    re = pd.DataFrame({'Source': all_result[0], 'Target': all_result[1], 'Weight':all_result[2]}, dtype=np.int64)
    # re.to_csv('./TadsCliques.csv',sep=' ',index=0)

    # build adjacent graph
    max1 = max(re['Source'])
    max2 = max(re['Target'])
    max3 = max(max1, max2)+1

    ad_graph = np.zeros((max3, max3))
    for i, row in re.iterrows():
        ad_graph[int(row['Source']), int(row['Target'])] = row['Weight']
    ad_graph = ad_graph+ad_graph.T
    np.fill_diagonal(ad_graph, 1)
    rr = markov_cluster(ad_graph, 1.3, 1)
    np.savetxt('/mnt/disk3/duanran/mactop/tadcliqu/cliquresult.txt', rr, fmt='%d')
    co = 0
    tad_cliques_result = []
    for lin in rr:
        clists = []
        for index in range(len(lin)):
            if lin[index] > 0:
                clists.append(index)
        if len(clists) > 0:
            tad_cliques_result.append(clists)
            co += 1
    return dense_mat, tad_cliques_result, tad_list


if __name__ == '__main__':
    # dense_mat, tad_cliques_result, tad_list = get_cliques_result()
    # print(tad_cliques_result)
    # length_list = []
    # for sub in tad_cliques_result:
    #     length_list.append(len(sub))
    # for vv in length_list:
    #     print(vv)

    # GM12878_hg19
    celline = 'GM12878'
    hic_mat = np.loadtxt('/mnt/disk3/duanran/mactop/tadcliqu/original_chr2.txt')
    mctad_result_list = mctad_result_format('/home/rduan/utad/algorithm/{0}re'.format(celline), 999)
    dense_mat, tad_cliques_result, tad_list = get_cliques_result_nmat(hic_mat, mctad_result_list)
    print(tad_cliques_result)
    for sub in tad_cliques_result:
        if len(sub) > 2:
            print(sub)

    # GM12878_hg38
    # mctad_result_list = mctad_result_format('/mnt/disk3/duanran/mactop/GM12878_result/mactop_hg38_result', 500)
    # celline = 'GM12878'
    # hic_mat= np.loadtxt('/mnt/disk3/duanran/pore_c_data/chr{0}-icemat.txt'.format(2))
    # dense_mat, tad_cliques_result, tad_list = get_cliques_result_nmat(hic_mat, mctad_result_list)
    # print(tad_cliques_result)
    # for vv in tad_cliques_result:
    #     if len(vv) >2:
    #         print(vv)

