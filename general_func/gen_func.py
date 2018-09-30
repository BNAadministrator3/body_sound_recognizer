#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
一些通用函数
'''

import difflib

def GetEditDistance(str1, str2):
	leven_cost = 0
	s = difflib.SequenceMatcher(None, str1, str2)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		#print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
		if tag == 'replace':
			leven_cost += max(i2-i1, j2-j1)
		elif tag == 'insert':
			leven_cost += (j2-j1)
		elif tag == 'delete':
			leven_cost += (i2-i1)
	return leven_cost

def Comapare(list_pre, list_true):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for index,true in enumerate(list_true):
		if true == 0 :
			if list_pre[index] == true:
				tn = tn + 1
			else:
				fp = fp + 1
		else:
			if list_pre[index] == true:
				tp = tp + 1
			else:
				fn = fn + 1
	
	return tp,fp,tn,fn


def Comapare2(pre, true):
	'''
	:param pre: single sparse predictions
	:param true: single sparse labels
	:return:
	'''
	tp = 0
	tn = 0
	fp = 0
	fn = 0

	if true == 0:
		if pre == true:
			tn = tn + 1
		else:
			fp = fp + 1
	else:
		if pre == true:
			tp = tp + 1
		else:
			fn = fn + 1

	return tp, fp, tn, fn

