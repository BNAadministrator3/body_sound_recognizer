#coding = utf-8
import platform as plat
import os, sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

from general_function.file_wav import get_wav_list



lookup_table = { '00':'0',
                 '01':'1',
                 '10':'2',
                 '11':'3'}

class dict_generate():

    def __init__(self,datapath,outpath):
        system_type = plat.system() #vairables within the method; cant be accessed by other method, let alone the instance
        self.datapath = datapath
        self.type = ''
        self.outpath = outpath

        self.slash = ''
        if system_type == 'Windows':
            self.slash = '\\'
        elif system_type == 'Linux':
            self.slash = '/'
        else:
            print('*[Warning] Unknown System\n')
            self.slash = '\\'  # 正斜杠

        if self.datapath[-1]!=self.slash:
            self.datapath += self.slash

        if self.outpath[-1]!=self.slash:
            self.outpath += self.slash

    # generate file_name-path pair dictionary
    def generate_index(self,type):
        self.type = type
        inpath = self.datapath
        if (self.type == 'train'):
            inpath += 'train'
        elif (self.type == 'eval'):
            inpath += 'validation'
        else:
            print('*[Error] Unknown Set Type\n')
            assert (0)

        files = os.listdir(inpath)
        outpath = self.outpath+self.type+'_index'+'.txt'
        with open(outpath, mode='w', encoding='utf-8') as txt_obj:
            for file in files:
                filename = file[:-4]
                wholepath = self.type + self.slash + file
                txt = filename+'\t'+wholepath+'\n'
                txt_obj.write(txt)

    def generate_symbol(self,subpath):
        inpath = self.datapath + subpath
        outpath = self.outpath + 'symbolics' + '_index' + '.txt'
        files = os.listdir(inpath)
        with open(outpath, mode='w', encoding='utf-8') as txt_obj:
            for file in files:
                #1 read the content of the file
                filepath = inpath + self.slash + file
                lines = []
                with open(filepath,mode='r') as inflow:
                    lines = inflow.readlines()
                #2 encode the label
                label = ''
                for line in lines:
                    char = line[-2]+line[-4]
                    code = lookup_table[char]
                    label += code+' '
                #3 concatenate
                filename = file[:-4]
                txt = filename + '\t' + label + '\n'
                txt_obj.write(txt)

    def symbol_split(self):
        train_dict, _ = get_wav_list(self.outpath+'train_index.txt','\t')
        val_dict, _ = get_wav_list(self.outpath+ 'eval_index.txt','\t')
        symbol_dict = {}
        lines = []
        with open(self.outpath+ 'symbolics_index.txt','r') as whole_symbols:
            lines=whole_symbols.readlines()
        for line in lines:
            if line != '':
                tmp = line.split('\t')
                symbol_dict[tmp[0]] = tmp[1]
        del lines
        # dump content
        with open(self.outpath+ 'symboltrain_index.txt','w') as sybotrain:
            for line in train_dict:
                correspond = symbol_dict[line]
                txt = line + '\t' + correspond
                sybotrain.write(txt)
        with open(self.outpath + 'symbolval_index.txt', 'w') as syboval:
            for line in val_dict:
                correspond = symbol_dict[line]
                txt = line + '\t' + correspond
                syboval.write(txt)


if(__name__=='__main__'):
    # inpath = '/ssd/1/zhaok_folder3/small_dataset/dataset/'
    # oupath = '/ssd/1/zhaok_folder3/small_dataset/datalist/ICBHI/'

    inpath = 'E:\workspace\Data\\rs_sdegrom\ICBHI_final_database_downsample'
    oupath = 'E:\workspace\stdenv\small_dataset\datalist'

    a = dict_generate(datapath = inpath, outpath=oupath)
    a.generate_index('train')
    a.generate_index('eval')
    a.generate_symbol('files')
    a.symbol_split()

