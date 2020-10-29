

class Instance():
    def __init__(self, sentence, srl, char_list, gold_predicates, pre_starts, pre_ends, arg_starts, arg_ends, arg_labels, srl_rels):
        self.sentence_list = sentence
        self.sentence_id_list = []
        self.srl = srl

        self.char_list = char_list
        self.char_id_list = []
        self.sentence_len = -1
        self.gold_predicates = gold_predicates
        self.num_gold_predicates = -1

        #label
        self.pre_starts = pre_starts
        self.pre_ends = pre_ends
        self.arg_starts = arg_starts
        self.arg_ends = arg_ends
        self.arg_labels = arg_labels
        self.arg_labels_list = []
        self.srl_len = []

        self.srl_rels = srl_rels

    def default(self):
        self.sentence_len = len(self.sentence_list)
        self.num_gold_predicates = len(self.gold_predicates)
        self.srl_len = len(self.pre_starts)

    # def __str__(self):
    #     return 'sent:' + ''.join(self.sentence_list) + ' srl:' +


