from general_func.file_wav import get_wav_list
import numpy as np

def get_interval_list(filepath):
	coarse_dict, _ =get_wav_list(filepath)

	for key, value in coarse_dict.items():
		line = value.split(';')
		line = line[:-1]
		norm = np.zeros( (len(line),2),dtype=float )
		for index, zhi in enumerate(line):
			try:
				tmp1,tmp2 = zhi.split(',')
				norm[index][0] = float(tmp1)
				norm[index][1] = float(tmp2)
			except:
				print('[error]unpaired data')
				print('key:%s\nvalue:%s'%(key,value))
				assert(0)
		for i in range(len(line)-1):
			if norm[i][1]!= norm[i+1][0]:
				print('[error]unequal data')
				assert(0)
		coarse_dict[key]=norm

	return coarse_dict


if (__name__=='__main__'):
	filepath='E:\workspace\stdenv\small_dataset\datalist\ICBHI\intervals_index.txt'
	a=get_interval_list(filepath)
	b=a