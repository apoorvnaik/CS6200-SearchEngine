import urllib
import re
import math
import time
import sys
import os
import linecache


class SearchEngine:
    'Project 3 class file that processes the queries'
    stoplist = "stoplist.txt"
    queries_file = "processed-CACM.txt"
    stop_words = ""
    all_ctf, all_df = [], []
    doc_ranked_list = {}
    term_inv_list = {}
    doc_map = {}
    offset_map = {}
    query_map = {}
    query = ""
    avg_doc_len = {0: 493.0, 1: 45, 2: 288.0, 3: 288.0}
    avg_query_len = 0.0
    # Need to set these values by analysing the indexer output
    # INDEX1: STOP STEM IR Class Specification
    # INDEX0: NOSTOP NOSTEM
    # INDEX2: NO STOP STEM
    # INDEX3: STOP NO STEM
    unique_words = {0: 207615, 1: 7969, 2: 207224, 3: 166054}
    num_terms = {0: 41802513, 1: 144637, 2: 24401877, 3: 24401877}

    def __init__(self, index_type, query):
        self.query = query
        self.index_type = index_type
        #Read Stop words
        FH = open(self.stoplist, "r")
        data = FH.read()
        self.stop_words = re.compile(r'\w+', re.DOTALL).findall(data)
        #Read queries
        FH.close()
        FH = open(self.queries_file)
        data = FH.readlines()
        # Mapping Query number
        for line in data:
            data = re.compile(r'\w+', re.DOTALL).findall(line)
            self.query_map[int(data[0])] = data[1:]
        FH.close()
        self.avg_query_len = self.avg_query_len()
        #self.populate_index()

    # Read created index into memory
    def populate_index(self):
        FH = open("index1-offset.txt")
        data = FH.read()
        FH.close()
        for line in data.split("\n"):
            self.offset_map[line[0]] = int(line[1])
        #for term in terms:
        #    FH = open(prefix + term)
        #    numbers = re.compile(r'\w+', re.DOTALL).findall(FH.read())
        #    self.term_inv_list[term] = {'ctf': int(numbers[0]),
        #                             'df': int(numbers[1]),
        #                             'rows': map(lambda x: (numbers[2 + 4 * x] + "-" + numbers[3 + 4 * x],
        #                               float(numbers[4 + 4 * x]),
        #                               float(numbers[5 + 4 * x])),
        #                                range(0, (len(numbers) - 2) / 4))}

    # List to plain string
    def list_to_str(self, lst):
        return str(lst).replace("[", "").replace("]", "").replace(",", "")

    # Remove periods hyphens and other characters that cause trouble with lemur
    def sanitize_query(self, query):
        query = query.lower().replace("-", " ").strip()
        for c in "!\"#$%&'()*+,./:;<=>?@[\]^_{|}~":
            query = query.replace(c, "")
        return query

    # Remove stop words from the query
    def parse_query(self, query):
        return list(set(filter(lambda x: x not in self.stop_words, self.sanitize_query(query))))

    # Get inverted list for given term
    def get_term_inv_list(self, term):
        
        if term in self.offset_map.keys():
            FH = open("index1.txt")
            line = FH.read()
            vals = line.split()
            count = 0
            while count < vals[1]
        else:
            return 0

    # OPtimized term data for laplace and jelinek
    def get_optimized_term_inv_list(self, terms):
        results = {}
        df_map = {}
        ctf_map = {}
        results_list = []
        for term in self.term_inv_list.keys():
            # Construct an inverted list for processinG
            ctf, df = float(self.term_inv_list[term]['ctf']), float(self.term_inv_list[term]['df'])
            inverted_list = self.term_inv_list[term]['rows']
            temp = {}
            for (docid, doclen, tf) in inverted_list:
                temp[docid] = [doclen, tf]
            results[term] = temp
            df_map[term] = df
            ctf_map[term] = ctf
            results_list.append(results)
            results_list.append(df_map)
            results_list.append(ctf_map)
        return results_list

    #COmpute avg query length
    def avg_query_len(self):
        total_terms = 0
        for query_terms in self.query_map.values():
            total_terms += len(query_terms)
        return (total_terms / len(self.query_map.keys()))

    # Write output to a file
    def write_output(self, model_name, q_no):
        FH = open(model_name + ".txt", "a")
        for rank, row in zip(range(1, 1001), self.doc_ranked_list):
            line = str(q_no) + " Q0 " + row[0] + " " + str(rank) + " " + str(row[1]) + " Exp"
            FH.write(line + "\n")
        FH.close()
        print "Query " + str(q_no) + " done.\n"

    # COmpute OKAPI for each term
    def term_okapi(self, tf, query_len):
        return (tf / (tf + 0.5 + 1.5 * query_len / self.avg_query_len))

    def compute_okapi(self):
        for q_no in sorted(self.query_map.keys()):
            terms = self.get_query_terms(q_no)
            for term in terms:
                term_inv_list = self.get_term_inv_list(term)
                if term_inv_list != 0:
                    self.compute_okapi_helper(q_no, term_inv_list)
                else:
                    continue
            self.sort_doc_ranked_list()
            self.write_output("okapi", q_no)
            self.clear_doc_ranked_list()

    #Compute OKAPI-tf model values
    def compute_okapi_helper(self, q_no, term_inv_list):
        query_okapi = self.term_okapi(1, len(self.query_map[q_no]))
        okapitf = 0.0
        for (docid, doclen, tf) in term_inv_list['rows']:
            okapitf = tf / (tf + 0.5 + (1.5 * doclen / self.avg_doc_len[self.index_type]))
            if docid in self.doc_ranked_list:
                self.doc_ranked_list[docid] += okapitf * query_okapi
            else:
                self.doc_ranked_list[docid] = okapitf * query_okapi

    # Compute TF-IDF
    def compute_tf_idf(self):
        for q_no in sorted(self.query_map.keys()):
            terms = self.get_query_terms(q_no)
            for term in terms:
                term_inv_list = self.get_term_inv_list(term)
                if term_inv_list != 0:
                    self.compute_tf_idf_helper(q_no, term_inv_list)
                else:
                    continue
            self.sort_doc_ranked_list()
            self.write_output("tfidf", q_no)
            self.clear_doc_ranked_list()

    #COmpute TD IDF model values
    def compute_tf_idf_helper(self, q_no, term_inv_list):
        query_okapi = self.term_okapi(1, len(self.query_map[q_no]))
        okapitf = 0.0
        idf = math.log10(3204 / term_inv_list['df'])
        for (docid, doclen, tf) in term_inv_list['rows']:
            okapitf = tf / (tf + 0.5 + (1.5 * doclen / self.avg_doc_len[self.index_type]))
            if docid in self.doc_ranked_list:
                self.doc_ranked_list[docid] += okapitf * query_okapi * idf
            else:
                self.doc_ranked_list[docid] = okapitf * query_okapi * idf

    # Compute Laplace Model Values
    def compute_laplace(self):
        results = []
        for q_no in sorted(self.query_map.keys()):
            terms = self.get_query_terms(q_no)
            self.compute_and_smoothen(q_no, terms, self.laplace_smoothing)
            self.sort_doc_ranked_list()
            self.write_output("laplace", q_no)
            self.clear_doc_ranked_list()

    # Laplace smoothing function
    def laplace_smoothing(self, val, tf, ctf, doclen):
        p = (tf + 1) / (doclen + self.unique_words[self.index_type])
        return p * val

    # Generic function call for Smoothing models
    def compute_and_smoothen(self, q_no, terms, f_name):
        results = self.get_optimized_term_inv_list(terms)
        docid_map = results[0]
        df_map = results[1]
        ctf_map = results[2]

        all_docid = {}
        t_results = {}
        for term in terms:
            if term in self.term_inv_list.keys():
                temp = docid_map[term]

                for docid in temp.keys():
                    all_docid[docid] = temp[docid][0]

        for c_id in all_docid.keys():
            t_results[c_id] = 1
            for term in terms:
                if term in self.term_inv_list.keys():
                    doc_map = docid_map[term]
                    ctf = ctf_map[term]
                    if c_id in doc_map:
                        t_results[c_id] = f_name(t_results[c_id],
                                                   doc_map[c_id][1],
                                                   ctf, doc_map[c_id][0])
                    else:
                        t_results[c_id] = f_name(t_results[c_id],
                                                  0, ctf, all_docid[c_id])
        self.doc_ranked_list = t_results

    def jelinek_smoothing(self, val, tf, ctf, doclen):
        lmda = 0.2
        p = (lmda * tf / doclen) + ((1 - lmda) * (ctf / self.num_terms[self.index_type]))
        return p * val

    def compute_jelinek(self):
        results = []
        for q_no in sorted(self.query_map.keys()):
            terms = self.get_query_terms(q_no)
            self.compute_and_smoothen(q_no, terms, self.jelinek_smoothing)
            self.sort_doc_ranked_list()
            self.write_output("jelinek", q_no)
            self.clear_doc_ranked_list()

    def compute_BM25(self):
        for q_no in sorted(self.query_map.keys()):
            terms = self.get_query_terms(q_no)
            for term in terms:
                term_inv_list = self.get_term_inv_list(term)
                if term_inv_list != 0:
                    self.compute_BM25_helper(q_no, self.get_term_inv_list(term))
                else:
                    continue
            self.sort_doc_ranked_list()
            self.write_output("bm25", q_no)
            self.clear_doc_ranked_list()

    def compute_BM25_helper(self, q_no, term_inv_list):
        i = 0
        count = 1
        ctf = term_inv_list['ctf']
        df = term_inv_list['df']
        for (docid, doclen, tf) in term_inv_list['rows']:
            B = 0.75
            k1 = 1.2
            k = k1 * ((1 - B) + B * doclen / self.avg_doc_len[self.index_type])
            bm_score = 0.0
            a = 0.0
            b = (2.2 * tf) / (k + tf)
            c = 1
            if(count <= df):
                a = math.log((0.5 / 0.5) / ((df + 0.5) / (3204 - df + 0.5)))
                count += 1
                if docid in self.doc_ranked_list:
                    bm_score = self.doc_ranked_list[docid] + (a * b * c)
                else:
                    bm_score = a * b * c
                self.doc_ranked_list[docid] = bm_score
            if count == df + 1:
                i += 1
                count = 1

    # Get list of all terms belonging to a specific query
    def get_query_terms(self, q_no):
        return self.query_map[q_no]

    # Sort the ranked list in descending order of SCORE
    def sort_doc_ranked_list(self):
        temp = sorted(self.doc_ranked_list.iteritems(), key=lambda x: x[1], reverse=True)
        self.doc_ranked_list = temp[0:1001]

    # Remove previously stored in-memory data
    def clear_doc_ranked_list(self):
        self.doc_ranked_list = dict()

start = time.time()
f_name = sys.argv[1]
index_type = int(sys.argv[2])
if len(sys.argv) == 3:
    lp = Project3(index_type, None)
elif len(sys.argv) == 4:
    # If query is passed from cmdline then use that query instead f
    # processing the stored Queries
    lp = Project3(index_type, sys.argv[3])

func_mapping = {'okapi': lp.compute_okapi,
                    'tfidf': lp.compute_tf_idf,
                    'laplace': lp.compute_laplace,
                    'jelinek': lp.compute_jelinek,
                    'bm25': lp.compute_BM25
                    }
func_mapping[f_name]()
end = time.time()
print end - start
