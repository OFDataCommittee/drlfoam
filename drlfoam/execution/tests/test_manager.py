
from os import remove
from os.path import join, isfile
from queue import Queue
from pathlib import Path
from time import sleep
from ..manager import Runner, TaskManager


class TestRunner():
    def test_run_no_args(self):
        queue = Queue()
        path = join("/tmp", "runner_testfile_no_args.txt")
        def create_file():
            if isfile(path):
                remove(path)
            Path(path).touch()
        queue.put((create_file, [], {}))
        runner = Runner(queue, "Runner 0")
        runner.run()
        runner.join()
        assert isfile(path)
        remove(path)

    def test_run_args(self):
        queue = Queue()
        def create_file(path):
            if isfile(path):
                remove(path)
            Path(path).touch()
        path = join("/tmp", "runner_testfile_args.txt")
        queue.put((create_file, [path], {}))
        runner = Runner(queue, "Runner 0")
        runner.run()
        runner.join()
        assert isfile(path)
        remove(path)

    def test_run_kwargs(self):
        queue = Queue()
        def create_file(path):
            if isfile(path):
                remove(path)
            Path(path).touch()
        path = join("/tmp", "runner_testfile_kwargs.txt")
        queue.put((create_file, [], {"path": path}))
        runner = Runner(queue, "Runner 0")
        runner.run()
        runner.join()
        assert isfile(path)
        remove(path)

    def test_run_args_kwargs(self):
        queue = Queue()
        def create_file(path_1, path_2):
            for p in (path_1, path_2):
                if isfile(p):
                    remove(p)
                Path(p).touch()
        path_1 = join("/tmp", "runner_testfile_1.txt")
        path_2 = join("/tmp", "runner_testfile_2.txt")
        queue.put((create_file, [path_1], {"path_2": path_2}))
        runner = Runner(queue, "Runner 0")
        runner.run()
        runner.join()
        for p in (path_1, path_2):
            assert isfile(p)
            remove(p)


class TestTaskManager():
    def test_add_and_run_no_args(self):
        manager = TaskManager(1)
        path = join("/tmp", "runner_testfile_no_args.txt")
        def create_file():
            if isfile(path):
                remove(path)
            Path(path).touch()
        manager.add(create_file)
        assert manager.qsize() == 1
        manager.run()
        assert isfile(path)
        remove(path)


    def test_add_and_run_args(self):
        manager = TaskManager(1)
        def create_file(path_1, path_2):
            for p in (path_1, path_2):
                if isfile(p):
                    remove(p)
                Path(p).touch()
        path_1 = join("/tmp", "runner_testfile_1.txt")
        path_2 = join("/tmp", "runner_testfile_2.txt")
        manager.add(create_file, path_1, path_2)
        manager.run()
        for p in (path_1, path_2):
            assert isfile(p)
            remove(p)

    def test_add_and_run_args_kwargs(self):
        manager = TaskManager(1)
        def create_file(path_1, path_2):
            for p in (path_1, path_2):
                if isfile(p):
                    remove(p)
                Path(p).touch()
        path_1 = join("/tmp", "runner_testfile_1.txt")
        path_2 = join("/tmp", "runner_testfile_2.txt")
        manager.add(create_file, path_1, path_2=path_2)
        manager.run()
        for p in (path_1, path_2):
            assert isfile(p)
            remove(p)