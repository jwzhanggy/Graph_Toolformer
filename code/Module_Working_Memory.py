from collections import OrderedDict

class Module_Working_Memory(OrderedDict):

    def __init__(self, maxlen):
        self.maxlen = maxlen
        OrderedDict.__init__(self)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popitem(last=False)

    def length(self):
        return len(self)

    def max_length(self):
        return self.maxlen

    def change_max_length(self, new_maxlen):
        self.maxlen = new_maxlen

if __name__ == "__main__":
    m = Module_Working_Memory(maxlen=2)
    print(m.length(), m.max_length())
    m[1] = 'a'
    m['ad'] = 2
    m['da'] = '2'
    print(m)

    m.change_max_length(new_maxlen=3)
    m[1] = 'a'
    print(m)

    print(1 in m)

    print("as" in m)
