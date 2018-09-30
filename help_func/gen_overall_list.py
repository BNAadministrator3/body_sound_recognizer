import os
from help_func.gen_wav_index import  dict_generate

class gen_overall_index(dict_generate):
	def __init__(self,datapath,dest_path):
		super(gen_overall_index, self).__init__(datapath=datapath, outpath=dest_path)
		self.common_path = ''
		self.dict = {}

	def gen_set(self,type):
		
		if type == 'train':
			self.common_path = self.datapath + 'train'+self.slash
		if type == 'eval':
			self.common_path = self.datapath + 'train'+self.slash
		
		s = set([])
		rover = ('00','01','10','11')
		for i in rover:
			list_name_folder = os.listdir(self.common_path + i)
			for j in list_name_folder:
				str = self.common_path + i + self.slash + j
				s.add(str)