#program to calculate how many messages are needed to shuffle in various mapreduce schemes

import numpy as np
import scipy as sp
import os
import math
import itertools as it
from operator import itemgetter




Q = 26 #number of keys
N = 1000 #number of chapters



input_file = "./random_text.txt"
possible_keys = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#reads in text, separates by line (each line is a block of data).  
#returns a list of blocks, each with a series of words
def read_in_data(file):
	text_file = open(file, "r")
	lines = text_file.read().split(',')
	#each line in line should be an array
	lines_array = dict()
	for line in lines:
		linesplit = list(line)
		lines_array[lines.index(line)] = linesplit
	return lines_array



#assigns mappers to blocks based on disjoint separation of the text. assigns keys to blocks.  
#returns list of chapters for each server to map
def vanilla_assign(input_data, p, K):
	size_of_block = math.ceil(len(input_data)/K)
	assignments_blocks = dict()

	size_of_keyblock = math.ceil(Q/K)
	assignments_keys = dict()
	for server in range(K):
		assignments_blocks[server] = list()
		for chapter in range(server*size_of_block, min(N, (server + 1)*size_of_block)):
			assignments_blocks[server].append(chapter)
		
		assignments_keys[server] = list()
		for letter in range(server*size_of_keyblock, min(Q, (server + 1)*size_of_keyblock)):
			assignments_keys[server].append(possible_keys[letter])
	return assignments_blocks, assignments_keys

#assigns mappers to blocks based on redundant separation of the text, parameter p
def coded_assign(input_data, p, K):
	#each block is randomly assigned to pK servers
	#each server has at most pN blocks
	assignments = dict()
	possible_servers = dict()
	for server in range(K):
		assignments[server] = list()
		possible_servers[server] = 0
	for block in range(N):
		#randomly pick pK servers in possible_servers
		choices_servers = np.random.choice(list(possible_servers.keys()), size = int(p*K), replace = True)
		#put this block in the assigned blocks for these servers
		#increase the count for this server
		for choice in choices_servers:
			assignments[choice].append(block)
			possible_servers[choice] = possible_servers[choice] + 1
			
		for choice in choices_servers:
			if choice in possible_servers:
				if possible_servers[choice] >= p*N:
					possible_servers.pop(choice, None)

	#assign keys to each
	size_of_keyblock = math.ceil(Q/K)
	assignments_keys = dict()
	for server in range(K):
		assignments_keys[server] = list()
		for letter in range(server*size_of_keyblock, min(Q, (server + 1)*size_of_keyblock)):
			assignments_keys[server].append(possible_keys[letter])


	return assignments, assignments_keys


#naively assigns mappers to blocks dependent on parameter p
def redundant_assign(input_data, p, K):
	#naive -- first pN blocks to each of the first pK servers, etc
	size_of_block = math.ceil(p*N) #how many blocks each server group gets
	size_of_server_group = math.ceil(p*K) #how many servers in each server group
	num_server_groups = math.ceil(K/size_of_server_group) #how many distinct groups
	assignments = dict()
	#create list of list of assignments:
	for server in range(K):
		blocks_assigned = list()
		group_num = int(server/size_of_server_group)
		for block in range(group_num*size_of_block, min(N, (group_num + 1)*size_of_block)):
			blocks_assigned.append(block)
		assignments[server] = blocks_assigned

	#assign keys to each
	size_of_keyblock = math.ceil(Q/K)
	assignments_keys = dict()
	for server in range(K):
		assignments_keys[server] = list()
		for letter in range(server*size_of_keyblock, min(Q, (server + 1)*size_of_keyblock)):
			assignments_keys[server].append(possible_keys[letter])

	return assignments, assignments_keys


def map(input_data, server, block_assignment):
	#gets lettercount for each key in assigned block assignment for a single server
	maps = list()
	for block in block_assignment:
		#print(input_data)
		line_to_map = list(input_data[block])
		for letter in possible_keys:
			#create dict of chapter, key, value
			freq = input_data[block].count(letter)
			maps.append((block, letter))
	return maps

#creates list of all values needed for each server, gets length of this list
def vanilla_shuffle(input_data, mapped_data, assigned_keys, p, K):
	#for each server, create a list of all values for this specific key.
	needed_vals_master = dict()
	total_communication_lag = 0
	for server in mapped_data:
		keys_needed = assigned_keys[server]
		#get full list of needed vals, then cross off ones already have
		needed_vals_list = list()
		for chapter in input_data:
			for letter in keys_needed:
				created_tup = (chapter, letter)
				needed_vals_list.append(created_tup)
		#delete what already have:
		have_vals_list = mapped_data[server]
		todel = list()
		for needed_val in needed_vals_list:
			if needed_val in have_vals_list:
				todel.append(needed_val)
		for delitem in todel:
			needed_vals_list.remove(delitem)
		total_communication_lag = total_communication_lag + len(needed_vals_list)
		needed_vals_master[server] = needed_vals_list


	return total_communication_lag

#creates list of all values needed for each server, gets length of this list
def redundant_shuffle(input_data, mapped_data, assigned_keys, p, K):
	#for each server, create a list of all values for this specific key.
	needed_vals_master = dict()
	total_communication_lag = 0
	for server in mapped_data:
		keys_needed = assigned_keys[server]
		#get full list of needed vals, then cross off ones already have
		needed_vals_list = list()
		for chapter in input_data:
			for letter in keys_needed:
				created_tup = (chapter, letter)
				needed_vals_list.append(created_tup)
		#delete what already have:
		have_vals_list = mapped_data[server]
		todel = list()
		for needed_val in needed_vals_list:
			if needed_val in have_vals_list:
				todel.append(needed_val)
		for delitem in todel:
			needed_vals_list.remove(delitem)
		total_communication_lag = total_communication_lag + len(needed_vals_list)
		needed_vals_master[server] = needed_vals_list


	return total_communication_lag


#creates list of all values needed for each server, 
#runs coded shuffle scheme to create list of recieved messages for each server, gets length of this list
def coded_shuffle(input_data, mapped_data, assigned_keys, p, K):
	total_communication_lag = 0
	S = p*K + 1 #size of sets
	subsets_list = list(it.combinations(range(K), int(S))) #get all subsets of this size
	for subset in subsets_list:
		#for each server k in the set, get the list of values exclusively known by all other servers in S and needed by k
		subset = list(subset)
		max_len_message = 0
		for server in subset:
			other_guys = [item for item in list(subset) if item != server]
			needed_vals_list = list()
			needed_vals_list_exclusive = list() #This is v_S\k^k
			keys_needed = assigned_keys[server]
			for chapter in input_data:
				for letter in keys_needed:
					created_tup = (chapter, letter)
					needed_vals_list.append(created_tup)
			for tup in needed_vals_list:
				check = 0
				for other_server in other_guys:
					if (tup not in mapped_data[other_server]):
						check = check +1 
				if (check == 0):
					needed_vals_list_exclusive.append(tup)
			#divide by the number of people in the group to see how much longer their message will be -- use this for num messages total
			if int(len(needed_vals_list_exclusive)/(p*K)) > max_len_message:
				max_len_message = int(len(needed_vals_list_exclusive)/(p*K))
			#adds max_len_message*(pK +1) pieces of information
		total_communication_lag = total_communication_lag + math.ceil((max_len_message)*(p*K + 1))

	return total_communication_lag






input_data = read_in_data(input_file)

fout = open("mapreduce_comparison.txt", "w+")
fout.write("p\tK\treduce_mode\tcommunication_lag\n")

#generate for various values of K
for K in (5, 8, 10, 15, 20):
	#K is number of servers
	p = 3/K #redundancy
	assigned_vals, assigned_keys = vanilla_assign(input_data, p, K)
	mapped_data = dict()
	for server in range(K):
		mapped_data[server] = map(input_data, server, assigned_vals[server])
	total_lag = vanilla_shuffle(input_data, mapped_data, assigned_keys, p, K)
	print("Vanilla: ", total_lag)
	fout.write(str(p) + "\t" + str(K) + "\t" + "vanilla\t" + str(total_lag) + "\n")

	assigned_vals, assigned_keys = redundant_assign(input_data, p, K)
	mapped_data = dict()
	for server in range(K):
		mapped_data[server] = map(input_data, server, assigned_vals[server])
	total_lag = redundant_shuffle(input_data, mapped_data, assigned_keys, p, K)
	print("Redundant: ", total_lag)
	fout.write(str(p) + "\t" + str(K) + "\t" + "redundant\t" + str(total_lag) + "\n")


	assigned_vals, assigned_keys = coded_assign(input_data, p, K)
	mapped_data = dict()
	for server in range(K):
		mapped_data[server] = map(input_data, server, assigned_vals[server])
	total_lag = coded_shuffle(input_data, mapped_data, assigned_keys, p, K)
	print("Coded: ", total_lag)
	fout.write(str(p) + "\t" + str(K) + "\t" + "coded\t" + str(total_lag) + "\n")


