

from help_func.gen_wav_index import  dict_generate
import os


class interval_generate(dict_generate):
	def __init__(self,datapath, outpath):
		super(interval_generate, self).__init__(datapath=datapath,outpath=outpath)
		# self.slash = super(interval_generate, self).slash
		# self.datapath =super(interval_generate, self).datapath
		# self.outpath = super(interval_generate, self).outpath

	def get_intervals(self):
		dir = self.datapath + 'files'

		files = os.listdir(dir)
		outpath = self.outpath + 'intervals_index' + '.txt'
		with open(outpath, mode='w', encoding='utf-8') as txt_obj:
			for file in files:
				filename = file[:-4]
				intervals = ''
				with open(dir+self.slash+file,mode='r') as inflow:
					lines = inflow.readlines()
					for line in lines:
						linelist = line.split('\t')
						intervals += linelist[0]+','+linelist[1]+';'
				txt = filename + '\t' + intervals + '\n'
				txt_obj.write(txt)


if(__name__=='__main__'):

	inpath = 'E:\workspace\Data\\rs_sdegrom\ICBHI_final_database_downsample'
	oupath = 'E:\workspace\stdenv\small_dataset\datalist'

	hello2 = interval_generate(inpath,oupath)
	hello2.get_intervals()