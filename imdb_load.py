# import pymysql
import os

# Connect to the database
# connection = pymysql.connect(host='',
#                              user='',
#                              password='',
#                              db='',
#                              charset='')

def load_imdb_data(data_folder):
    d = {}
    with open('pos.txt', errors='ignore') as f:
        pos_data = f.read()
        pos_most_frequent = find_top_100(pos_data)
        d["pos"] = review_filter(pos_data, pos_most_frequent)

    with open('neg.txt', errors='ignore') as f:
        neg_data = f.read()
        neg_most_frequent = find_top_100(neg_data)
        d["neg"] = review_filter(neg_data, neg_most_frequent)



    stop_words = stopwords.words('english')
    stop_words_+ = ["br", "", "nt"]
    stop_words.extend(stop_words_+)

# def get_mostfreq_words(data_dict):
# 	freqdict = {}
# 	# your code for creating a frequency distribution of the words in the dataset
# 	# can be useful to pre-select a vocabulary
# 	return freqdict
#
# def create_tables():
# 	# your code for creating the tables 'corpus', 'dictionary' and 'sentiment'
# 	with connection.cursor() as cur:
# 		q = """
# 			CREATE TABLE dictionary(
# 			  # some columns here and pk constraint here
# 			);
# 		"""
# 		cur.execute(q)
# 		connection.commit()
#
# 	with connection.cursor() as cur:
# 		q = """
# 			CREATE TABLE sentiment_corpus(
# 			# some columns and pk constraint here
# 			);
# 		"""
# 		cur.execute(q)
# 		connection.commit()
#
# 	with connection.cursor() as cur:
# 		q = """
# 			CREATE TABLE mentions(
# 			  # some columns and pk/fk constraints here
# 			);
# 		"""
# 		cur.execute(q)
# 		connection.commit()
#
#
# def populate_dictionary(most_freq_words):
# 	# columns: wordId, word
# 	rows = []
# 	idx = 1
# 	for idx,mfw in most_freq_words:
# 		# your code for populating the 'rows' list and incrementing the counter goes here
# 	with connection.cursor() as cur:
# 		q = """
# 				YOUR INSERT STATEMENT HERE
# 		"""
# 		cur.executemany(q, rows)
# 		connection.commit()
#
# def populate_sentiment_corpus(data_dictionary):
# 	#  columns: textID, sentimentValue
# 	rows = []
# 	text_idx = 1
# 	for label in data_dictionary:
# 		for document in data_dictionary[label]:
# 			# your code for populating the 'rows' list and incrementing the counter goes here
# 	with connection.cursor() as cur:
# 		q = """
# 				# YOUR INSERT STATEMENT HERE
# 		"""
# 		cur.executemany(q, rows)
# 		connection.commit()
#
# def populate_mentions(data_dictionary, most_freq_words):
# 	#  columns: textID, ID, wordID
# 	rows = []
# 	hit_idx = 1
# 	text_idx = 1
# 	# your (big) chunk of code goes here to populate the rows list
# 	with connection.cursor() as cur:
# 		q = """
# 				YOUR INSERT STATEMENT HERE
# 		"""
# 		cur.executemany(q, rows)
# 		connection.commit()
#
# def query_32():
# 	results = []
# 	# your code here
# 	return results
#
if __name__ == '__main__':

    data_folder = 'imdb/train'

    data = load_imdb_data(data_folder)
    # freqdist = get_mostfreq_words(data)
	# we only keep the 100 most frequent words
	# most_freq_words = [a for a,b in sorted(freqdist.items(), key=lambda x:x[1], reverse=True)][:100]
    #
	# try:
	# 	# this is the workflow
	# 	create_tables()
	# 	print('populating the dictionary table')
	# 	populate_dictionary(most_freq_words)
	# 	print('populating the sentiment corpus table')
	# 	populate_sentiment_corpus(data)
	# 	print('populating the mentions table')
	# 	populate_mentions(data, most_freq_words)
	# finally:
	# 	connection.close()
    #
	# print('== Attempting query for question 3.2 ==')
	# for r in query_32():
	# 	print(r)

    load_imdb_data
