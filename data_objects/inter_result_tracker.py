import pickle
import copy


class InterResultTracker:
    def __init__(self, save_loc, args={}, content={}):
        self._args = args
        self.save_loc = save_loc
        self._content = content

    def add(self, key, value):
        content = self._content.setdefault(key, [value])
        if content != [value]:
            self._content[key].append(value)
        self.save_to_file()

    def get_deepcopy(self):
        return copy.deepcopy(self._content)

    def save_to_file(self, save_loc=None):
        with open(save_loc if save_loc else self.save_loc, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)

    def load_from_file(self, save_loc=None):
        with open(save_loc if save_loc else self.save_loc, 'rb') as f:
            self.__dict__.clear()
            self.__dict__.update(pickle.load(f))


if __name__ == '__main__':
    tracker = InterResultTracker('testfile3', args={'test':'testvalue'})
    print(tracker.get_deepcopy())
    tracker.add('accuracy', 10)
    print(tracker.get_deepcopy())
    tracker.add('accuracy', 20)
    print(tracker.get_deepcopy())
    tracker.add('acc2', 30)
    print(tracker.get_deepcopy())
    tracker.save_to_file()
    print('tracker2:')
    tracker2 = InterResultTracker('testfile3')
    tracker2.load_from_file()
    print(tracker2.get_deepcopy())
    print(tracker2._args)
