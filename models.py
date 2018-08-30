class model:

    def train(self, train_data):
        #do something with self.train_data 
        self.model = None

    def predict(self,test_data):
        self.test_data = test_data
        return None


from random import random
class random_model(model):

    def train(self,train_data):
        self.model = None

    def predict(self, test_data):
        result = []
        for i in range(0,len(test_data)):
            result.append(random())
        return result

class friends_model(model):

    def train(self,train_data):
        self.friends = train_data

    def predict(self, test_data):
        forecast = test_data.apply(lambda x: x in self.friends.index) #return 1 if vertex in train data set
        return forecast
        