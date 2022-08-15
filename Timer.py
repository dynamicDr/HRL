import time
class Timer():
    def __init__(self):
        self.dict = {}
        self.current_timer_name = None
        self.start_time = 0
        pass

    def start_timer(self, name):
        if name not in self.dict.keys():
            self.dict[name] = 0
        if self.current_timer_name is not None:
            self.end_timer()
        self.start_time = time.time()
        self.current_timer_name = name

    def end_timer(self):
        end_time = time.time()
        time_elapsed = end_time - self.start_time
        self.dict[self.current_timer_name] += time_elapsed
        self.current_timer_name = None

    def print_result(self):
        sorted_dict = sorted(self.dict.items(), key=lambda x: x[1], reverse=True)
        print(sorted_dict)


if __name__ == '__main__':
    timer = Timer()
    timer.start_timer("a")
    a = 0
    for i in range(10000):
        a += 1
    timer.start_timer("b")
    b = 0
    for i in range(100000):
        b += 1
    timer.start_timer("c")
    c = 0
    for i in range(1000000):
        c += 1
    timer.end_timer()
    print(timer)
