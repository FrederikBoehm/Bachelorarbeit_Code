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
        if 'amazonaws' in host:
            self.workspace_dir = '/home/ubuntu/Bachelorarbeit'
            user_dir = os.environ['userprofile']
            self.ssh_dir = f'{user_dir}/.ssh/id_rsa'
            self.user = 'ubuntu'
        else:
            self.workspace_dir = '/mnt/md0/user/boehmfr68270/Bachelorarbeit'
            user_dir = os.environ['userprofile']
            self.ssh_dir = f'{user_dir}/.ssh/ohm'
            self.user = 'boehmfr68270'
    
    def on_created(self, event):
        output = ''
        path = self._get_Path(event.src_path)

        if not event.src_path.startswith(tuple(self.syncignore)):
            if event.is_directory:
                logging.log(level = logging.INFO, msg = 'Creating directory ' + path)
                output = subprocess.run(['ssh', '-i', f'{self.ssh_dir}', f'{self.user}@{self.host}', f'mkdir -p {path}'], stdout=subprocess.PIPE)
            else:
                logging.log(level = logging.INFO, msg = f'Copying new file {event.src_path} to {path}')
                output = subprocess.run(['scp', '-i', f'{self.ssh_dir}', event.src_path, f'{self.user}@{self.host}:{path}'], stdout=subprocess.PIPE)
            
            logging.log(level = logging.INFO, msg=output)

    def on_deleted(self, event):
        output = ''
        path = self._get_Path(event.src_path)

        if not event.src_path.startswith(tuple(self.syncignore)):
            if event.is_directory:
                logging.log(level=logging.INFO, msg = 'Deleting directory ' + path)
                output = subprocess.run(['ssh', '-i', f'{self.ssh_dir}', f'{self.user}@{self.host}', f'rmdir -p {path}'], stdout=subprocess.PIPE)
            else:
                logging.log(level = logging.INFO, msg = 'Deleting file ' + path)
                output = subprocess.run(['ssh', '-i', f'{self.ssh_dir}', f'{self.user}@{self.host}', f'rm -f {path}'], stdout=subprocess.PIPE)

            logging.log(level = logging.INFO, msg=output)
        

    def on_modified(self, event):
        output = ''
        path = self._get_Path(event.src_path)

        if not event.src_path.startswith(tuple(self.syncignore)):
            if not event.is_directory:
                logging.log(level=logging.INFO, msg=f'Updating file {path} from {event.src_path}')
                output = subprocess.run(['scp', '-i', f'{self.ssh_dir}', event.src_path, f'{self.user}@{self.host}:{path}'], stdout=subprocess.PIPE)

            logging.log(level = logging.INFO, msg=output)

    def _get_Path(self, event_path):
        splitted_path = event_path.split('\\')
        file_name = splitted_path[len(splitted_path) - 1]
        path = self.workspace_dir + '/' + file_name
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