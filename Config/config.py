from configparser import ConfigParser


class Config(object):

    def __init__(self, config_file):
        config = ConfigParser()
        config.read(config_file)
        for section in config.sections():
            for k, v in config.items(section):
                print(k, ":", v)
        self._config = config
        self.config_file = config_file
        config.write(open(config_file, 'w+'))

    def add_args(self, section, key, value):
        if self._config.has_section(section):
            print('This is a section already.')
        else:
            print('Now, we will add a new section.')
            self._config.add_section(section)
        if self._config.has_option(section, key):
            self._config.set(section, key, value)
            print('Add parameter successfully.')
        self._config.write(open(self.config_file, 'w'))

    # Dataset
    @property
    def usual_train_file(self):
        return self._config.get('Dataset', 'usual_train_file')

    @property
    def virus_train_file(self):
        return self._config.get('Dataset', 'virus_train_file')

    @property
    def usual_dev_file(self):
        return self._config.get('Dataset', 'usual_dev_file')

    @property
    def virus_dev_file(self):
        return self._config.get('Dataset', 'virus_dev_file')

    @property
    def usual_eval_file(self):
        return self._config.get('Dataset', 'usual_eval_file')

    @property
    def virus_eval_file(self):
        return self._config.get('Dataset', 'virus_eval_file')

    @property
    def usual_vat_file(self):
        return self._config.get('Dataset', 'usual_vat_file')

    @property
    def virus_vat_file(self):
        return self._config.get('Dataset', 'virus_vat_file')

    @property
    def embedding_file(self):
        return self._config.get('Dataset', 'embedding_file')

    # Save
    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def save_pkl_path(self):
        return self._config.get('Save', 'save_pkl_path')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_vocab_path(self):
        return self._config.get('Save', 'save_vocab_path')

    @property
    def usual_model_pkl(self):
        return self._config.get('Save', 'usual_model_pkl')

    @property
    def virus_model_pkl(self):
        return self._config.get('Save', 'virus_model_pkl')

    @property
    def usual_train_data_word_pkl(self):
        return self._config.get('Save', 'usual_train_data_word_pkl')

    @property
    def virus_train_data_word_pkl(self):
        return self._config.get('Save', 'virus_train_data_word_pkl')

    @property
    def usual_dev_data_word_pkl(self):
        return self._config.get('Save', 'usual_dev_data_word_pkl')

    @property
    def virus_dev_data_word_pkl(self):
        return self._config.get('Save', 'virus_dev_data_word_pkl')

    @property
    def usual_vat_data_word_pkl(self):
        return self._config.get('Save', 'usual_vat_data_word_pkl')

    @property
    def virus_vat_data_word_pkl(self):
        return self._config.get('Save', 'virus_vat_data_word_pkl')

    @property
    def embedding_pkl(self):
        return self._config.get('Save', 'embedding_pkl')

    @property
    def train_word_data_iter(self):
        return self._config.get('Save', 'train_word_data_iter')

    @property
    def dev_word_data_iter(self):
        return self._config.get('Save', 'dev_word_data_iter')

    @property
    def test_word_data_iter(self):
        return self._config.get('Save', 'test_word_data_iter')

    @property
    def fact_word_src_vocab(self):
        return self._config.get('Save', 'fact_word_src_vocab')

    @property
    def fact_word_tag_vocab(self):
        return self._config.get('Save', 'fact_word_tag_vocab')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def load_vocab_path(self):
        return self._config.get('Save', 'load_vocab_path')

    @property
    def output_dir(self):
        return self._config.get('Save', 'output_dir')

    @property
    def save_usual_path(self):
        return self._config.get('Save', 'save_usual_path')

    @property
    def save_virus_path(self):
        return self._config.get('Save', 'save_virus_path')

    # Train
    @property
    def use_cuda(self):
        return self._config.getboolean('Train', 'use_cuda')

    @property
    def epoch(self):
        return self._config.getint('Train', 'epoch')

    @property
    def tra_batch_size(self):
        return self._config.getint('Train', 'tra_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Train', 'test_batch_size')

    @property
    def use_lr_decay(self):
        return self._config.getboolean('Train', 'use_lr_decay')

    @property
    def clip_max_norm_use(self):
        return self._config.getboolean('Train', 'clip_max_norm_use')

    @property
    def test_interval(self):
        return self._config.getint('Train', 'test_interval')

    @property
    def early_stop(self):
        return self._config.getint('Train', 'early_stop')

    @property
    def update_every(self):
        return self._config.getint('Train', 'update_every')

    @property
    def loss(self):
        return self._config.get('Train', 'loss')

    @property
    def shuffle(self):
        return self._config.get('Train', 'shuffle')

    @property
    def scheduler_bert(self):
        return self._config.getboolean('Train', 'scheduler_bert')

    # Data_loader
    @property
    def data_cut(self):
        return self._config.getboolean('Data_loader', 'data_cut')

    @property
    def data_cut_k(self):
        return self._config.getint('Data_loader', 'data_cut_k')

    @property
    def stop_word(self):
        return self._config.getboolean('Data_loader', 'stop_word')

    @property
    def read_sen(self):
        return self._config.getboolean('Data_loader', 'read_sen')

    @property
    def read_char(self):
        return self._config.getboolean('Data_loader', 'read_char')

    # Model

    @property
    def max_length(self):
        return self._config.getint('Model', 'max_length')

    @property
    def bert_size(self):
        return self._config.getint('Model', 'bert_size')

    @property
    def dropout(self):
        return self._config.getfloat('Model', 'dropout')

    @property
    def class_num(self):
        return self._config.getint('Model', 'class_num')

    @property
    def dis_class_num(self):
        return self._config.getint('Model', 'dis_class_num')

    @property
    def bert_lr(self):
        return self._config.getfloat('Model', 'bert_lr')

    @property
    def dis_lr(self):
        return self._config.getfloat('Model', 'dis_lr')

    @property
    def min_lr(self):
        return self._config.getfloat('Model', 'min_lr')

    @property
    def weight_decay(self):
        return self._config.getfloat('Model', 'weight_decay')

    @property
    def lr_rate_decay(self):
        return self._config.getfloat('Model', 'lr_rate_decay')

    @property
    def margin(self):
        return self._config.getfloat('Model', 'margin')

    @property
    def p(self):
        return self._config.getint('Model', 'p')

    @property
    def patience(self):
        return self._config.getint('Model', 'patience')

    @property
    def epsilon(self):
        return self._config.getfloat('Model', 'epsilon')

    @property
    def factor(self):
        return self._config.getfloat('Model', 'factor')

    @property
    def pre_embedding(self):
        return self._config.getboolean('Model', 'pre_embedding')

    @property
    def correct_bias(self):
        return self._config.getboolean('Model', 'correct_bias')

    @property
    def decay(self):
        return self._config.getfloat('Model', 'decay')

    @property
    def decay_steps(self):
        return self._config.getfloat('Model', 'decay_steps')

    @property
    def clip(self):
        return self._config.getfloat('Model', 'clip')

    @property
    def beta_1(self):
        return self._config.getfloat('Model', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Model', 'beta_2')

    @property
    def label_smoothing(self):
        return self._config.getfloat('Model', 'label_smoothing')

    @property
    def vat_xi(self):
        return self._config.getfloat('Model', 'vat_xi')

    @property
    def vat_eps(self):
        return self._config.getfloat('Model', 'vat_eps')

    @property
    def vat_iter(self):
        return self._config.getint('Model', 'vat_iter')

    @property
    def vat_alpha(self):
        return self._config.getfloat('Model', 'vat_alpha')
