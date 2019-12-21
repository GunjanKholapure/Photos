import pickle
import os
import shutil
import numpy as np
import time

def cluster_faces(upd=False):
	start_time = time.time()
	dr = os.path.dirname(os.path.abspath(__file__))
	
	if upd:
		file = open("ue.pk","rb")
	else:
		file = open("pte.pk","rb")

	embeds = pickle.load(file)
	embeds_copy = embeds
	lst = []
	for key,value in embeds.items():
		lst.append(key)
	#print(len(key))
	file.close()
		
	f = open("diff.txt","w")
	threshold = .40#most important variable
	filter_limit = 2
	print(threshold)
	cluster = {}
	itr = 0

	flag = True
	while True:
		length = len(lst)
		cnt = 0
		clus = {}
		if not flag:
			break
	
		print(threshold)
		flag = False

		visit = [0]*length
		new_lst = []
			
		while  cnt!=length:
			for i in range(length):
				if visit[i] == 0:
					visit[i] = 1
					new_lst.append(lst[i])
					cnt += 1
					if itr == 0:
						cluster[lst[i]] = []
					break
			
			for j in range(i+1,length):
					if visit[j] == 0:
						diff = 0
						#print(len(embeds[lst[i]]))
						
						for k in range(128):
							diff +=  (embeds[lst[i]][k] - embeds[lst[j]][k])**2
							f.write(str(itr) + " " + str(diff) + "\n")
						"""
						else:
							diff = 0
							for a in cluster[lst[i]]:
								for b in cluster[lst[j]]:
									diff += dis(embeds[a],embeds[b])

							for a in cluster[lst[i]]:
								diff += dis(embeds[a],embeds[lst[j]])

							for b in cluster[lst[j]]:
								diff += dis(embeds[lst[i]],embeds[b])

							diff = diff/ ( ( len(cluster[lst[i]]) +1)*(len(cluster[lst[j]]) + 1 ) )
						"""
						
						if diff < threshold:	
							flag = True
							cluster[lst[i]].append(lst[j])
							visit[j] = 1
							cnt += 1
	
							if itr > 0:
								if lst[j] in cluster:
									for vecs in cluster[lst[j]]:
										cluster[lst[i]].append(vecs)
									
									
									cluster[lst[j]] = []
								
								
		if itr==0:  
			for key,value in cluster.items():
				lnt = len(cluster[key])
				if lnt>0:
					diff = embeds[key]
					for j in value:
						diff += embeds[j]
					diff = diff/(len(cluster[key])+1)
					embeds[key] = diff
			
		if itr>0:
			for key,value in cluster.items():
				lnt = len(cluster[key])

				if lnt>0:
					diff = embeds_copy[key]
					for j in cluster[key]:
						diff += embeds_copy[j]
					diff = diff/(len(cluster[key])+1)
					embeds[key] = diff
			
		lst = new_lst
		print(itr)
		itr += 1
	


	f.close()
	final = {}
	print(len(cluster),itr)
	cnt = 0
	cnt1 = 0

	file = open(dr+"/../../../im2txt/ploc.pk","rb")
	ploc = pickle.load(file)
	file.close()
	if not os.path.exists(dr + "/keys"):
			os.makedirs(dr+"/keys")
	
	if upd:
		f = open("kemb.pk","rb")
		kemb = pickle.load(f)
		f.close()

		f = open("agcls.pk","rb")
		old_cluster = pickle.load(f)
		f.close()

		for key, values in old_cluster.items():
			for k,v in cluster.items():
				if dis(kemb[key],embeds[k]) < 0.4:
					ind = k.rfind("_")
					name = k[:ind]
					old_cluster[key].append(ploc[name])
					for elements in v:
						ind = elements.rfind("_")
						name = elements[:ind]
						old_cluster[key].append(ploc[name])
						cluster[k] = []


		for key,value in cluster.items():
			if len(cluster[key]) > filter_limit:
				old_cluster[key] = []
				ind = key.rfind("_")
				name = key[:ind]
				old_cluster[key].append(ploc[name])
				kemb[key] = embeds[key]
				if os.path.isfile(dr+"/tmp/"+key):
					shutil.copy(dr+"/tmp/"+key,dr+"/keys/")
				for elements in cluster[key]: 
					ind = elements.rfind("_")
					name = elements[:ind]
					old_cluster[key].append(ploc[name])


		with open("agcls.pk","wb") as f:
			pickle.dump(old_cluster,f)

		with open("kemb.pk","wb") as f:
			pickle.dump(kemb,f)

	else :
		kemb = {}
		for key, val in cluster.items():
			if len(cluster[key])>filter_limit:
				kemb[key] = embeds[key]
				final[key] = []
				ind = key.rfind("_")
				name = key[:ind]
				final[key].append(ploc[name])
				if os.path.isfile(dr+"/tmp/"+key):
					shutil.copy(dr+"/tmp/"+key,dr+"/keys/")
				for i in cluster[key]:
					#print(i)
					ind = i.rfind("_")
					name = i[:ind]
					final[key].append(ploc[name])
				cnt +=1
			elif len(cluster[key])>0:
				cnt1+=1


		print(cnt,cnt1)
		print(time.time()-start_time)
		with open("agcls.pk","wb") as f:
			pickle.dump(final,f)

		with open("kemb.pk","wb") as f:
			pickle.dump(kemb,f)

	if os.path.exists(dr+"/tmp"):
			shutil.rmtree(dr+"/tmp")
		


def dis(vec1,vec2):
	diff = 0
	for i in range(128):
		diff += (vec1[i] - vec2[i])**2

	return diff


def prop_clustering():
	dr = os.path.dirname(os.path.abspath(__file__))
	lst = os.listdir(dr+"/tmp")
		
	print(len(lst))

	file = open(dr + "/../pte.pk","rb")
	info = pickle.load(file)
	embeds = info["ind_embed"]
	file.close()

	cnt = 0
	itn = [0]*len(embeds)
	vec = [0]*len(embeds)
	for key,value in embeds.items():
		itn[cnt] = key
		vec[cnt] = value
		cnt += 1


	threshold = 0.65 #most important variable
	filter_limit = 1
	print(threshold)
	cluster = {}
	itr = 0

	flag = True
	min_val = [1000]*len(lst)
	min_node = [-1]*len(lst)
	visit = [0]*len(lst)
	cluster = []
	itr = 0
	while True:
		print(itr)
		mval = 1000
		print(type(mval))
		mnode = -1
		var = [False]*len(lst)
	
		for i in range(len(lst)):
			for j in range(i+1,len(lst)):
				if visit[i] == 0 and visit[j] == 0:
					dist = dis(vec[i],vec[j])
					if dist<threshold and dist<min_val[i]:
						var[i] = True
						min_val[i] = j
						min_node[i] = j

			if var[i]:
				if min_val[i] < min_val[min_node[i]]:
					min_val[min_node[i]] = min_val[i]
					min_node[min_node[i]] = i
					var[min_node[i]] = True

				#print(type(min_val[i]),type(mval))

				if min_val[i] < mval:
					mval = min_val[i]
					mnode = i
		
		itr += 1

		if mnode != -1:
			vec1 = vec[mnode]
			vec2 = vec[min_node[mnode]]
			
			vec[mnode] = (vec1 + vec2)/2
			visit[min_node[mnode]] = 1
			if lst[mnode] not in cluster:
				cluster[mnode] = []

			cluster[mnode].append(min_node[mnode])

		else:
			break


	final = {}
	print(len(cluster))
	cnt = 0
	if os.path.exists(dr + "/keys"):
		shutil.rmtree(dr+"/keys")
	os.makedirs(dr+"/keys")
	file = open(dr+"/../../../im2txt/ploc.pk","rb")
	ploc = pickle.load(file)

	"""
	for key, val in cluster.items():
		if len(cluster[key])>filter_limit:
			final[key] = []
			ind = key.rfind("_")
			name = key[:ind]
			final[key].append(ploc[name])
			shutil.copy(dr+"/tmp/"+key,dr+"/keys/")
			for i in cluster[key]:
				print(i)
				ind = i.rfind("_")
				name = i[:ind]
				final[key].append(ploc[name])
			cnt +=1

	#shutil.rmtree(dr+"/tmp")
	print("clsters = "  + str(cnt))

	with open("agcls.pk","wb") as f:
		pickle.dump(final,f)
	"""
def find_par(node,arr):
	p = arr[node]
	if p!=node:
		p = find_par(arr[node],arr)
	return p



def min_cluster():
	dr = os.path.dirname(os.path.abspath(__file__))
	lst = os.listdir(dr+"/tmp")
		
	print(len(lst))

	file = open(dr + "/../../../im2txt/pte.pk","rb")
	#embeds = pickle.load(file)
	info = pickle.load(file)
	embeds = info["ind_embed"]
	embeds_copy = {}
	file.close()


	threshold = .7#most important variable
	filter_limit = 2
	print(threshold)
	cluster = {}
	copy_embeds = {}
	itr = 0
	flag = True
	
	while  True:
	
		parent = [0]*len(lst)
		rank = [0]*len(lst)
		min_node = [-1]*len(lst)
		min_val = [threshold]*len(lst)
		new_lst = []

		if not flag:
			break

		flag = False
		#threshold -= itr*.05
		

		print(itr)		
		clus = {}

		for i in range(len(lst)):
			parent[i] = i

		for i in range(len(lst)):
			for j in range(i+1,len(lst)):
				diff = dis(embeds[lst[i]],embeds[lst[j]])
				if diff < min_val[i]:
					min_val[i] = diff
					min_node[i] = j

					if diff< min_val[j]:
						min_val[j] = diff
						min_node[j] = i

			if min_val[i] != threshold:
				flag = True
				p1 = find_par(i,parent)
				p2 = find_par(min_node[i],parent)
				if p1 != p2:
					if rank[p1]<rank[p2]:
						parent[p1] = p2
					elif rank[p1]>rank[p2] :
						parent[p2] = p1
					else:
						parent[p1] = p2
						rank[p2] += 1

		for i in range(len(lst)):
			p = find_par(i,parent)
			if itr ==0: 
				if p!=i:
					if lst[p] not in cluster:
						cluster[lst[p]] = []
				
					cluster[lst[p]].append(lst[i])
					if lst[p] not in new_lst:
						new_lst.append(lst[p])	
			else:
				if p!=i:
					if lst[p] not in clus:
						clus[lst[p]] = []
					clus[lst[p]].append(lst[i])
					cluster[lst[p]].append(lst[i])
					for vecs in cluster[lst[i]]:
						cluster[lst[p]].append(vecs)
					cluster[lst[i]] = []
					if lst[p] not in new_lst:
						new_lst.append(lst[p])	

						

		if itr == 0:
			for key,value in cluster.items():
				diff = embeds[key]
				copy_embeds[key] = embeds[key]
				for embs in value:
					diff += embeds[embs]
				embeds[key] = diff/(len(value)+1)

		else:
			for key,value in clus.items():
				diff = copy_embeds[key]
				for vecs in cluster[key]:
					diff += embeds[vecs]
				embeds[key] = diff/(len(cluster[key])+1)



		lst = new_lst
		itr += 1 

	final = {}
	print(len(cluster),itr)
	cnt = 0
	if os.path.exists(dr + "/keys"):
		shutil.rmtree(dr+"/keys")
	os.makedirs(dr+"/keys")
	file = open(dr+"/../../../im2txt/ploc.pk","rb")
	ploc = pickle.load(file)


	for key, val in cluster.items():
		if len(cluster[key])>filter_limit:
			final[key] = []
			ind = key.rfind("_")
			name = key[:ind]
			final[key].append(ploc[name])
			shutil.copy(dr+"/tmp/"+key,dr+"/keys/")
			for i in cluster[key]:
				#print(i)
				ind = i.rfind("_")
				name = i[:ind]
				final[key].append(ploc[name])
			cnt +=1

	#shutil.rmtree(dr+"/tmp")
	print(cnt)

	with open("agcls.pk","wb") as f:
		pickle.dump(final,f)
