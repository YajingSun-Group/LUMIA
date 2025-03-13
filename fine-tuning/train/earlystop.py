import torch

class EarlyStopping(object):
    """Early stop performing
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    taskname : str or None
        Filename for storing the model checkpoint

    """
    
    def __init__(self, pretrained_model='Null_early_stop.pth', 
                 mode='higher', patience=10, filename=None,
                 former_task_name="None",
                 logger=None):

        
        former_filename = filename
        
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        
        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.former_filename = former_filename
        self.best_score = None
        self.early_stop = False
        
        self.logger = logger
    
    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)
    
    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)
    
    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            self.logger(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
        # print(self.filename)
    
    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])
    
    def load_former_model(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.former_filename)['model_state_dict'])
        # model.load_state_dict(torch.load(self.former_filename, map_location=torch.device('cpu'))['model_state_dict'])


