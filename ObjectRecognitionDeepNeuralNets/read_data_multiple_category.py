import read_data2 as rd
import numpy as np

def read_data_multiple_category(start_stop_list, category_list, size):
	
	# start_stop_list --> [[start, stop], [start, stop]]

	total_data = []
	for i,j in zip(start_stop_list, category_list):
		data_list = rd.read_data(i[0], i[1], j, size)
		for data in data_list:
			total_data.append(data)

	
	return total_data			

