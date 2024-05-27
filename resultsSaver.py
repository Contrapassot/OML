import json

class sharpnessResultsSaver:
    def __init__(self, effect = None, effect_name = None, list_of_sharpness = None, list_of_values = None, errors = None, model_class_name='MLP_1', optimizer='SGD', n_iterations=5, save = True):
        
        if save:
            self.effect = effect
            self.effect_name = effect_name
            self.list_of_sharpness = list_of_sharpness
            self.list_of_values = list_of_values
            self.errors = errors
            self.model_class_name = model_class_name
            self.optimizer = optimizer
            self.n_iterations = n_iterations
            
            self.file_name = "results/" + model_class_name + "_" + effect_name + "_" + optimizer 
            
            try:
                assert effect is not None
                assert effect_name is not None
                assert list_of_sharpness is not None
                assert list_of_values is not None
                assert errors is not None
                assert len(self.list_of_values) == len(self.list_of_sharpness)
                assert len(self.list_of_values) == len(self.errors)
                if save:
                    self.saveResultToJSON()
                    self.results = self.getResultFromFile()
            except:
                if self.check_if_file_exists():
                    self.results = self.getResultFromFile()
                else:
                    self.results = None

            

    def saveResultToJSON(self):
        result = {
            "effect": self.effect,
            "effect_name": self.effect_name,
            "list_of_sharpness": self.list_of_sharpness,
            "list_of_values": self.list_of_values,
            "errors": self.errors,
            "model_class_name": self.model_class_name,
            "optimizer": self.optimizer,
            "n_iterations": self.n_iterations
        }
        # save to json file
        
        with open(f'{self.file_name}.json', 'w') as f:
            json.dump(result, f, indent=4)

    def getResultFromFile(self):
        with open(f'{self.file_name}.json', 'r') as f:
            result = json.load(f)
        
        return result
    
    def printResult(self):
        print(self.file_name)
    
    def check_if_file_exists(self):
        try:
            with open(f'{self.file_name}.json', 'r') as f:
                return True
        except:
            return False