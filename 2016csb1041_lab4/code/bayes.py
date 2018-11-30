import operator
import math


m = -1

# setting m - estimate parameters :
t = input('Enter 1 for using default m OR Enter 2 for using m of your choice : ')
if(t == '2'):	
	m = float(input('Enter value of m parameter : '))



# bayes function :
def bayes(m):

	Ham_count = 0
	Spam_count = 0
	Vocabulary = set()

	ham_word_count = {}
	spam_word_count = {}
	likelihood_for_spam = {}
	likelihood_for_ham = {}
	dictionary = {}


	# training purpose :

	# read from file :
	with open('nbctrain', 'r') as infile:
		for line in infile:
			split_arr = line.split()
			if(split_arr[1] == 'ham'):
				Ham_count = Ham_count + 1
				for i in range(2,len(split_arr),2):
					Vocabulary.add(split_arr[i])
					dictionary[split_arr[i]] = 1
					if(split_arr[i] in ham_word_count):
						ham_word_count[split_arr[i]] = ham_word_count[split_arr[i]] + int(split_arr[i+1])
					else:
						ham_word_count[split_arr[i]] = int(split_arr[i+1])	

			else:
				Spam_count = Spam_count + 1
				for i in range(2,len(split_arr),2):
					Vocabulary.add(split_arr[i])
					dictionary[split_arr[i]] = 1
					if(split_arr[i] in spam_word_count):
						spam_word_count[split_arr[i]] = spam_word_count[split_arr[i]] + int(split_arr[i+1])
					else:
						spam_word_count[split_arr[i]] = int(split_arr[i+1])	



	# total number of words in classes spam & ham (repeat allowed):
	N_spam = sum(spam_word_count.values())
	N_ham = sum(ham_word_count.values())


	# total training instances :
	Total_count = Spam_count + Ham_count
	
	if(m == -1):
		m = len(Vocabulary)
	p = 1/m


	# getting likelihood probabilities :
	for key, value in dictionary.items():
		tmp = m * p
		if(key in ham_word_count):
			likelihood_for_ham[key] = (ham_word_count[key] + tmp)/(N_ham + m)
		else:
			likelihood_for_ham[key] = (tmp)/(N_ham + m)
			

	for key, value in dictionary.items():
		tmp2 = m * p
		if(key in spam_word_count):
			likelihood_for_spam[key] = (spam_word_count[key] + tmp2)/(N_spam + m)
		else:
			likelihood_for_spam[key] = (tmp2)/(N_spam + m)
			


	# getting top 5 maxm probability words :
	top_5_ham = dict(sorted(likelihood_for_ham.items(), key=operator.itemgetter(1), reverse=True)[:5])
	top_5_spam = dict(sorted(likelihood_for_spam.items(), key=operator.itemgetter(1), reverse=True)[:5])


	#----------------------------------------------------------------------------------
 
	# testing purpose :

	correct_outputs = []
	predicted_outputs = []

	# read from file :
	with open('nbctest', 'r') as inf:
		for line in inf:
			split_arr = line.split()
			if(split_arr[1] == 'ham'):
				correct_outputs.append(1)
			else:
				correct_outputs.append(-1)

			# for Ham :	
			temp = 1
			Ham_prob = math.log(Ham_count/Total_count)
			for i in range(2,len(split_arr),2):
				if(split_arr[i] in likelihood_for_ham):
					temp = likelihood_for_ham[split_arr[i]]
				else:
					temp = p
				temp = math.log(temp) * int(split_arr[i+1])
				Ham_prob = Ham_prob + temp


			# for Spam :	
			temp2 = 1
			Spam_prob = math.log(Spam_count/Total_count)
			for i in range(2,len(split_arr),2):
				if(split_arr[i] in likelihood_for_spam):
					temp2 = likelihood_for_spam[split_arr[i]]
				else:
					temp2 = p
				temp2 = math.log(temp2) * int(split_arr[i+1])
				Spam_prob = Spam_prob + temp2


			# Naive Bayes Condition :	
			if(Spam_prob > Ham_prob):
				predicted_outputs.append(-1)
			else :
				predicted_outputs.append(1)	

	final_count = 0
	for j in range(0,len(predicted_outputs)):
		if(predicted_outputs[j] == correct_outputs[j]):
			final_count = final_count + 1

	# Accuracy :	
	Accuracy = final_count/(len(predicted_outputs))
	return (Accuracy,top_5_ham,top_5_spam)	




# printing outputs :
(Accuracy,top_5_ham,top_5_spam) = bayes(m)

print()
print('5 Most frequently words indicative of Spam email & their likelihood probabilities are : ')
print()
for key, value in top_5_spam.items():
	print(key + ' : '+ str(value))

print('----------------------------------------------------------------------------------------')
print()
print('5 Most frequently words indicative of Ham email & their likelihood probabilities are : ')
print()
for key, value in top_5_ham.items():
	print(key + ' : '+ str(value))
print('----------------------------------------------------------------------------------------')
print()

print('======> Accuracy for Naive bayes model is : ' + str(Accuracy*100) +'%' )	







			
				


				

			
				




















			
		
