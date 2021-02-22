from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np

class BERTDataset(Dataset):
    def __init__(self, corpus_path, author_community, vocab, seq_len, doc_len, hops, encoding="utf-8", corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.doc_len = doc_len
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.author_community = author_community
        self.neighbor_size = 4
        self.num_negauthor = 3
        self.hops = hops
        '''
        for author in self.author_community:
            neighbors = self.author_community[author]['neighbors']
            relations = self.author_community[author]['relations']
            if len(neighbors) > self.neighbor_size:
                idxs = np.random.choice(len(neighbors), self.neighbor_size, replace=False) 
                neighbors = np.array(neighbors)[idxs]
                relations = np.array(relations)[idxs]

            else:
                idxs = np.random.choice(len(neighbors), self.neighbor_size, replace=True) 
                neighbors = np.array(neighbors)[idxs]
                relations = np.array(relations)[idxs]

            self.author_community[author]['neighbors'] = neighbors
            self.author_community[author]['relations'] = relations

        '''
        with open(corpus_path, "r", encoding=encoding) as f:
            '''
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1
            '''

            self.lines = {}
            for i, line in enumerate(tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)):
                tokens = line.replace("\n", "").split("\t")
                author = tokens[0]
                documents = tokens[1:-1]
                self.lines[author] = []
                for d in documents:
                    self.lines[author].append(d.split())  
                #self.lines = [line.replace("\n", "").replace("\t", " ").split()
                #              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
            self.corpus_lines = len(self.lines)
            self.all_author = list(self.lines.keys())
        '''
        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
        '''

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        author = self.vocab.itoa[item]
        bert_input, bert_label = self.author_papers(author)
        #bert_input = item
        #bert_label=1

        neighbor_papers, relations, relations_type, hop_count = self.author_neighbor_papers(author, self.neighbor_size, self.hops)

        negauthor_input = self.negauthor_papers(author, self.num_negauthor)

        #community_label = 0


        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "neighbor_input": neighbor_papers,
                  "relation": relations,
                  "relations_type": relations_type,
                  "hop_count": hop_count,
                  "negauthor_input":negauthor_input,
                  "community_label":0}

        return {key: torch.tensor(value) for key, value in output.items()}

    def author_papers(self, author):
        documents = self.lines[author].copy()

        #doc_len = 10
        
        semantic_inputs = []
        semantic_labels = []
        document_mask = []

        if len(documents)>=self.doc_len:
            documents = documents[:self.doc_len]
            document_mask = [1]*self.doc_len
        else:
            padding = [documents[-1] for _ in range(self.doc_len - len(documents))]
            documents.extend(padding)
            document_mask = [1]*len(documents)+[0]*(self.doc_len - len(documents))

        for i,d in enumerate(documents):
            bert_input, bert_label = self.random_word(d)
            #bert_input = bert_input[:self.seq_len]
            #bert_label = bert_label[:self.seq_len]

            if len(bert_input) >= self.seq_len:
                #idxs = np.random.choice(len(bert_input), self.seq_len, replace=False) 
                #bert_input = list(np.array(bert_input)[idxs])
                #bert_label = list(np.array(bert_label)[idxs])

                bert_input = bert_input[:self.seq_len]
                bert_label = bert_label[:self.seq_len]

            else:
                padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
                bert_input.extend(padding), bert_label.extend(padding)
            
            bert_input = [self.vocab.sos_index] + bert_input[:-1]
            bert_label = [self.vocab.pad_index] + bert_label[:-1]

            semantic_inputs.append(bert_input)
            semantic_labels.append(bert_label)

        return semantic_inputs, semantic_labels#, document_mask

    def author_neighbor_papers(self, author, neighbor_size, hops):
        neighbors, relations = self.author_neighbor(author, neighbor_size)

        neighbors_lasthop = neighbors
        relations_lasthop = relations
        hop_count = [0]*neighbor_size
        
        for i in range(hops-1):
            nid = random.randrange(len(neighbors_lasthop))
            i_neighbor_lasthop = neighbors_lasthop[nid]
            i_relation_lasthop = relations_lasthop[nid]

            neighbors_thishop, relations_thishop = self.author_neighbor(i_neighbor_lasthop, neighbor_size)
            relations_thishop = relations_thishop + (i_relation_lasthop+1)*3 #3 is the relation numer, and relation idx starts with 0

            neighbors.extend(neighbors_thishop)
            relations.extend(relations_thishop)

            neighbors_lasthop = neighbors_thishop
            relations_lasthop = relations_thishop

            hop_count.extend([i+1]*neighbor_size)

        relations_type = self.random_relation(relations)
        hop_count = self.random_relation(hop_count)

        neighbor_papers = []
        for n_author in neighbors:
            neighbor_input, _ = self.author_papers(n_author)
            #neighbor_input = self.vocab.atoi.get(n_author)
            neighbor_papers.append(neighbor_input)

        return neighbor_papers, relations, relations_type, hop_count

    def author_neighbor(self, author, neighbor_size):
        neighbors = self.author_community[author]['neighbors']
        relations = self.author_community[author]['relations']

        if len(neighbors) > neighbor_size:
            idxs = np.random.choice(len(neighbors), neighbor_size, replace=False) 
            neighbors = np.array(neighbors)[idxs]
            relations = np.array(relations)[idxs]

        else:
            idxs = np.random.choice(len(neighbors), neighbor_size, replace=True) 
            neighbors = np.array(neighbors)[idxs]
            relations = np.array(relations)[idxs]

        return list(neighbors), list(relations)

    def random_relation(self, relations):
        relations_label = []
        for i, relation in enumerate(relations):
            prob = random.random()
            if prob < 0.15:
                relations_label.append(relations[i]+1)
                
            else:
                relations_label.append(0)

        return relations_label

    def negauthor_papers(self, author, num_negauthor):
        negauthor_input = []
        i=0
        while i < num_negauthor:
            #jtem = random.randrange(len(self.lines))
            jauthor = random.choice(self.all_author)
            if jauthor != author:
                author_input, _  = self.author_papers(jauthor)
                negauthor_input.append(author_input)
                #negauthor_input.append(self.vocab.atoi.get(jauthor))
                i=i+1
        return negauthor_input
        

    def random_word(self, sentence):
        tokens = sentence
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            
            if prob < 0.15:
                prob /= 0.15
                
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                
                #tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                #output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                #output_label.append(self.vocab.atoi.get(author)+1)
                
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
            
            #tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
            #output_label.append(item+1)
        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)
        return t1, t2, 1

        # output_text, label(isNotNext:0, isNext:1)
        #if random.random() > 0.5:
        #    return t1, t2, 1
        #else:
        #    return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        '''
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2
        '''

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]
        '''
        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
        '''
