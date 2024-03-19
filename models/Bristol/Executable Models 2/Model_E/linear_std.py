"""
Version: 1.2
    Add unstd function
    
Version: 1.1
    Make sure every input follow correct type

Version: 1.0
    Linear_std finished construction
"""

class linear_std:
    # y= kx+b
    def __init__(self, k=0, b=0):
        self.k = float(k)
        self.b = float(b)

    def std(self, x):
        return self.k * x + self.b

    def unstd(self, y):
        return (y - self.b) / self.k

    def save(self,file_name):
        #save to file
        with open(file_name, 'w') as f:
            f.write('linear_std_data_file\n')
            f.write(str(self.k) + '\n')
            f.write(str(self.b) + '\n')
    
    def load(self,file_name):
        #load from file
        with open(file_name, 'r') as f:
            if f.readline()!='linear_std_data_file\n':
                throw('file is not linear_std_data_file')
            self.k = float(f.readline())
            self.b = float(f.readline())

    # def save(self,file_name):
    #     #save with npy
    #     np.save(file_name, np.array([self.k, self.b]))
    
    # def load(self,file_name):
    #     #load with npy
    #     data = np.load(file_name)
    #     self.k = float(data[0])
    #     self.b = float(data[1])


def get_std_range(min_x, max_x,min_range, max_range):
    #make sure every input is float
    min_x=float(min_x)
    max_x=float(max_x)
    min_range=float(min_range)
    max_range=float(max_range)

    k = (max_range - min_range) / (max_x - min_x)
    b = min_range - k * min_x
    return linear_std(k, b)


if __name__ == '__main__':

    rangeTest = get_std_range(10, 15, 0, 100)
    rangeTest.b,rangeTest.k
    rangeTest.std(11),rangeTest.std(14)

    b_std=get_std_range(10, 15, 0, 100)
    b_std.save('b_std.stdd')

    a_std = linear_std()
    a_std.load('b_std.stdd')

    (a_std.k, a_std.b),(b_std.k, b_std.b)


