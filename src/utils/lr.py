'''A wrapper class for scheduled optimizer '''

class CustomStepLr():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.n_steps = 0


    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        if(self.n_steps <= 1200000):
            lr = float('1e-4')
        elif(self.n_steps <= 1200000 + 300000):
            lr = float('3e-5')
        else:
            lr = float('1e-5')

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != '_optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)