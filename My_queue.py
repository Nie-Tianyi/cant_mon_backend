import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import random
import numpy as np
input_data: list=[]

scanteen = {
    "s1": {
        "service": np.mean([54, 55, 50, 66, 60, 101, 59]),
        "wait": 70,
        "arrive": [0, 2, 5, 15, 16, 32, 39, 44, 61]
    },
    "s3": {
        "service": np.mean([14, 12, 14, 20, 25, 28, 39]),
        "wait": 43,
        "arrive": []
    },
    "s4": {
        "service": np.mean([10, 65, 33, 56, 40]),
        "wait": 87
    },
    "s6": {
        "service": np.mean([66, 40, 50, 54, 68]),
        "wait": 100
    },
}

va: dict = {
    "vac": {
        "service": np.mean([7, 8, 16, 16, 22, 12, 30, 24, 10, 24, 32, 20, 25,8]),
        "wait": 25
    },
    "vae": {
        "service": 38,
        "wait": 69
    },
    "vaf": {
        "service": 79,
        "wait": 145
    }
}

def generate_random_number_list(start_time = random.randint(4, 23), mean= random.randint(5, 44), std_dev= random.randint(1, 100), length = 11):
    random_numbers = []
    previous_number = start_time

    for _ in range(length):
        new_number = previous_number + np.random.normal(mean, std_dev)
        random_numbers.append(new_number)
        previous_number = new_number

    return random_numbers

def data_generation(single_wait: int, seri: int, length: int = 10):
    threshold, randomv = 0.74, random.uniform(0, 1)
    new_seri = [seri]
    for _ in range(length):
        neg = random.choice([1, -1])
        if randomv > threshold:
            ser_val = random.randint(10, 24)
        else:
            ser_val = random.randint(1, 3)
        new_seri.append(seri+ser_val*neg)
    return waiting_time_generation(single_wait, length=length+1), new_seri

def waiting_time_generation(mean: int, std_var: int = 30, length=10):
    std = std_var/6
    border = (mean-std_var/2, mean+std_var/2)
    values = np.random.normal(mean, std, length)
    values = np.clip(values, border[0], border[1]).tolist()
    return values

def data_enlargement(waiting: list, seri: list, arrive: list, epoch: int = 4):
    assert len(waiting) == len(seri), "Length of arrive and service should be the same."
    mean, std = 0, 0.15
    for _ in range(epoch):
        neg = random.choice([1, -1])
        waiting += (waiting + neg*np.random.normal(mean, std, len(waiting))).tolist()
        seri += (seri + neg*np.random.normal(mean, std, len(seri))).tolist()
        arrive += (arrive + neg*np.random.normal(mean, std, len(arrive))).tolist()
    return waiting, seri, arrive

def _single_run(canteen):
    for k, v in canteen.items():
        waiting, seri = data_generation(v["wait"], v["service"])
        arrive = generate_random_number_list()
        v["wait"], v["service"], v["arrive"] = data_enlargement(waiting, seri, arrive)
    return canteen

scanteen, va = _single_run(scanteen), _single_run(va)
canteen = {**scanteen, **va}

import torch.nn as nn
import torch

def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class my_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(my_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out) + out
        out = self.fc3(out)
        return out

    def get_call__(self, path = r"waiting_time_LSTM.pth") -> torch.Any:
        return self.load_pretrained(path=path)

    def load_pretrained(self, path):
        return torch.load(path)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def get_call__(self, path = r"waiting_time_MLP.pth") -> torch.Any:
        return self.load_pretrained(path=path)

    def load_pretrained(self, path):
        return torch.load(path)

from collections import Counter
from scipy import stats

class ExponentionalWaitime:
    """ArithmeticError
    class only to simulate for the exponential wait time, since it is possibly fitted for canteen data
    """
    def __init__(self, arrive, service, lambda_=1):
        self.lambda_ = lambda_
        self.arrive = arrive
        self.service = service
        self.cva=1
        self.cvp=0
        self.a = 1
        self.p = np.mean(arrive)
    
    def get_next(self):
        return np.random.exponential(1/self.lambda_)

    def waiting_time(self):
        """
        assume every windows could only serve one customer at a time
        """
        previous, waiting_list, length = 0, [], len(self.arrive)
        for customer in range(length):
            waiting = 0
            if self.arrive[customer] <= previous:
                waiting = previous - self.arrive[customer]
                waiting_list.append((waiting, customer))
                previous = previous + self.service[customer]
                continue
            previous = self.service[customer] + self.arrive[customer]
        return waiting_list
    
    def stationary(self):
        """"""
        ia = [abs(self.arrive[i] - self.arrive[i-1]) for i in range(len(self.arrive)-1, 0, -1)]
        test_stat, p_value = stats.kstest(ia, 'expon')
        plt.hist(ia, bins=20, density=True)
        return p_value < 0.05, ia

    def cva_compute(self):
        stationary, ia = self.stationary()
        if stationary:
            self.a = np.mean(ia)
            return 1
        return np.std(ia)/self.a
    
    def cvp_compute(self):
        return np.std(self.service) / np.mean(self.service)
    
    def utilization_mui(self):
        return self.p/self.a
    
    def return_time(self):
        miu = self.utilization_mui()
        cva, cvp = self.cva_compute(), self.cvp_compute()
        return self.p*(miu/(1-miu))*((cva**2 + cvp**2)/2)
    
    def __call__(self):
        return abs(self.return_time())


import time
class Team_Model:
    def __init__(self, model_type: str, models: torch.nn = None):
        self.modelt = model_type
        self.time = time.time()
        self.people = 0
        self.arrivelist = generate_random_number_list(0, length=32)
        self.wait, self.service = canteen["s1"]["wait"][:32].copy(), canteen["s1"]["service"][:32].copy()
        if not models:
            self.calling, self.methods = self.get_model()
        else:
            self.calling, self.methods = models, 1

    def get_model(self):
        if self.modelt == "MLP":
            return MLPModel(input_size=32, hidden_size=64, output_size=32).get_call__(), 1
        elif self.modelt == "LSTM":
            return my_LSTM(input_size=32, hidden_size=64, output_size=32).get_call__(), 1
        elif self.modelt == "math model":
            return ExponentionalWaitime(self.arrivelist, self.service), 0
    
    def model_value(self):
        if not self.methods:
            return self.calling()
        with torch.no_grad():
            return self.calling(torch.tensor(self.wait).to(torch.float32))

    def update(self):
        temp = time.time()
        distinct = temp - self.time
        if self.methods == 0:
            self.arrivelist = self.arrivelist + generate_random_number_list(self.arrivelist[-1], length=2)
            self.service = self.service + [self.service[-1]+random.randint(10, 24)*(-1)**i for i in range(2)]
            self.arrivelist, self.service = self.arrivelist[2:], self.service[2:]
        self.wait = self.wait + [distinct]
        self.wait = self.wait[1:] 
        return 

    def run_estimate(self, people: int):
        if not self.people-people: return self.model_value()
        self.update()
        return self.model_value()


if __name__ == "__main__":
    test = Team_Model("MLP")
    val = test.run_estimate(2)
    print(val[0].item(), val[-1].item())
    val = test.run_estimate(5)
    print(val[0].item(), val[-1].item())
    val = test.run_estimate(6)
    print(val[0].item(), val[-1].item())

