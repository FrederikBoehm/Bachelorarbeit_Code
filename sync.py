import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import json
import os

class BasicHandler(FileSystemEventHandler):

    def __init__(self, host, syncignore):
        super().__init__()
        self.host = host
        self.syncignore = syncignore
    
    def on_created(self, event):
        output = ''
        path = self._get_Path(event.src_path)

        if not event.src_path(tuple(self.syncignore)):
            if event.is_directory:
                logging.log(level = logging.INFO, msg = 'Creating directory ' + path)
                output = subprocess.run(['ssh', f'ubuntu@{self.host}', f'mkdir -p {path}'], stdout=subprocess.PIPE)
            else:
                logging.log(level = logging.INFO, msg = f'Copying new file {event.src_path} to {path}')
                output = subprocess.run(['scp', event.src_path, f'ubuntu@{self.host}:{path}'], stdout=subprocess.PIPE)
            
            logging.log(level = logging.INFO, msg=output)

    def on_deleted(self, event):
        output = ''
        path = self._get_Path(event.src_path)

        if not event.src_path(tuple(self.syncignore)):
            if event.is_directory:
                logging.log(level=logging.INFO, msg = 'Deleting directory ' + path)
                output = subprocess.run(['ssh', f'ubuntu@{self.host}', f'rmdir -p {path}'], stdout=subprocess.PIPE)
            else:
                logging.log(level = logging.INFO, msg = 'Deleting file ' + path)
                output = subprocess.run(['ssh', f'ubuntu@{self.host}', f'rm -f {path}'], stdout=subprocess.PIPE)

            logging.log(level = logging.INFO, msg=output)
        

    def on_modified(self, event):
        output = ''
        path = self._get_Path(event.src_path)

        if not event.src_path(tuple(self.syncignore)):
            if not event.is_directory:
                logging.log(level=logging.INFO, msg=f'Updating file {path} from {event.src_path}')
                output = subprocess.run(['scp', event.src_path, f'ubuntu@{self.host}:{path}'], stdout=subprocess.PIPE)

            logging.log(level = logging.INFO, msg=output)

    def _get_Path(self, event_path):
        event_path = event_path.replace('\\', '/')
        event_path = event_path[1:]
        path = '/home/ubuntu/Bachelorarbeit' + event_path
        return path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.log(level = logging.INFO, msg=f'Starting file watcher with remote {sys.argv[1]}')
    syncignore_file = open('./syncignore.json', 'r')
    syncignore = list(map(lambda p : p.replace('/', '\\'), json.load(syncignore_file)["ignorepaths"]))
    logging.log(level = logging.INFO, msg=syncignore)
    path = '.'
    event_handler = BasicHandler(sys.argv[1], syncignore)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()