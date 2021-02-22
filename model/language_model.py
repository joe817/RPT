import torch.nn as nn
import torch
from .hierarchical_transformer import Document_Transformer, Researcher_Transformer


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, document_transformer, researcher_transformer, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        #self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.document_transformer = document_transformer
        self.researcher_transformer = researcher_transformer
        self.ht = HierarchicalTransformer(self.document_transformer, self.researcher_transformer,self.researcher_transformer.hidden)
        self.hmlm = HierarchicalMaskedLanguageModel(self.researcher_transformer.hidden, vocab_size)
        #self.author_embeddings = AutherEmbedding(self.bert.hidden)

        self.rt = RelationType(self.researcher_transformer.hidden)
        self.hc = HopCount(self.researcher_transformer.hidden)
        #self.softmax = nn.LogSoftmax(dim=-1)
        self.logsigmoid = nn.LogSigmoid()

        #self.relations = self.bert.relations()
        self.relation_emb =  nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(121,self.researcher_transformer.hidden)))
        #self.relation_bias = nn.Parameter(torch.zeros(3,self.bert.hidden))

        #self.author_emb = nn.Embedding(40281, 100)

        #self.relation_emb.requires_grad=True

        self.criterion = nn.NLLLoss(ignore_index=0)
        #self.criterion_n = nn.NLLLoss(ignore_index=0)
        #self.criterion_c = nn.NLLLoss(reduction = 'sum')

    def forward(self, data):
        #x, author = self.bert(x)
        #return self.mask_lm(x), author

        author_embeddings, masked_token_ouput, masked_doc_ouput = self.ht(data["bert_input"])
        #(batch_size * 10 * 20) * 64, (batch_size * 10) * 64
        [batch_size, doc_num, seq_len] = list(data["bert_input"].shape)
        masked_doc_ouput = masked_doc_ouput.repeat(1,seq_len, 1).reshape(batch_size*doc_num*seq_len, -1)
        masked_token_ouput = masked_token_ouput.reshape(batch_size*doc_num*seq_len, -1)
        #mask_output = self.mask_lm(torch.cat([masked_doc_ouput, masked_token_ouput], dim=1))
        mask_output = self.hmlm(masked_doc_ouput + masked_token_ouput)
        HMLM_loss = self.criterion(mask_output, data["bert_label"].reshape(batch_size*doc_num*seq_len))

        [batch_size, neighbor_num, _, _] = list(data["neighbor_input"].shape)
        neighbor_embeddings ,_ ,_ = self.ht(data["neighbor_input"])
        relations_emb = self.relation_emb[data["relation"]]
        community_embedding = (neighbor_embeddings * relations_emb).mean(1)
        author_embeddings_1 = author_embeddings.repeat(1,neighbor_num).reshape(batch_size, neighbor_num,-1)
        neighbor_output_rt = self.rt(author_embeddings_1 * neighbor_embeddings)
        neighbor_output_hc = self.hc(author_embeddings_1 * neighbor_embeddings)
        GRP_loss = self.criterion(neighbor_output_rt.transpose(1, 2), data["relations_type"]) + self.criterion(
            neighbor_output_hc.transpose(1, 2), data["hop_count"])

        [batch_size, negative_num, _, _] = list(data["negauthor_input"].shape)
        negauthor_embeddings ,_ ,_ = self.ht(data["negauthor_input"])
        score_ture = (author_embeddings * community_embedding).sum(-1)
        community_embedding_1 = community_embedding.repeat(1,negative_num).reshape(batch_size, negative_num,-1)
        score_false = (negauthor_embeddings * community_embedding_1).sum(-1)
        GCL_loss = - self.logsigmoid(torch.cat((score_ture.unsqueeze(1), -1*score_false),1)).mean(-1).mean(-1)

        loss = GCL_loss + 0.1*GRP_loss +  0.1* HMLM_loss
        
       
        #loss = community_loss
        '''
        author_embeddings = self.author_emb(data["bert_input"])
        neighbor_embeddings = self.author_emb(data["neighbor_input"])
        negauthor_embeddings = self.author_emb(data["negauthor_input"])

        #[dim1, dim2, _] = list(data["neighbor_input"].shape)
        relations_emb = self.relation_emb[data["relation"]]
        community_embedding = (neighbor_embeddings*relations_emb).mean(1)
        #community_embedding = neighbor_embeddings.unsqueeze(2).matmul(relations_emb).squeeze(2).sum(1) 
        #community_embedding = neighbor_embeddings.mean(1)

        [dim1, dim2] = list(data["negauthor_input"].shape)
        score_ture = (author_embeddings * community_embedding).sum(-1)
        community_embedding_1 = community_embedding.repeat(1,dim2).reshape(dim1, dim2,-1)
        score_false = (negauthor_embeddings * community_embedding_1).sum(-1)
        
        #scores = self.logsigmoid(torch.cat((score_ture.unsqueeze(1), score_false),1))

        #loss = self.criterion_c(scores, data["community_label"])
        loss = - self.logsigmoid(torch.cat((score_ture.unsqueeze(1), -1*score_false),1)).mean(-1).sum(-1)
        '''
        return loss


class HierarchicalTransformer(nn.Module):
    def __init__(self, document_transformer, researcher_transformer, hidden):
        super().__init__()
        self.document_transformer = document_transformer
        self.researcher_transformer = researcher_transformer
        self.hidden = hidden

    def forward(self, x):
        inputs = x.reshape(-1, list(x.shape)[-1])
        doc_num = list(x.shape)[-2]
        masked_token_ouput, doc_embeddings = self.document_transformer(inputs)
        doc_embeddings = doc_embeddings.reshape(-1, doc_num, self.hidden)
        masked_doc_ouput, author_embeddings = self.researcher_transformer(doc_embeddings)
        #print (x.shape, doc_embeddings.shape,author_embeddings.shape)
        if len (list(x.shape)) ==4:
            author_embeddings = author_embeddings.reshape(list(x.shape)[0],-1,self.hidden)

        return author_embeddings, masked_token_ouput, masked_doc_ouput
        


class RelationType(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 121)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        return self.softmax(self.linear(x))

class HopCount(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 5)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        return self.softmax(self.linear(x))


class HierarchicalMaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


