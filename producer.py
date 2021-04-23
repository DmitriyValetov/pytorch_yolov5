import os
from pathlib import Path 

from pprint import pprint
import time

from rsmq import RedisSMQ
import json


def get_test_pdfs_paths():
    """
        To be integrated
    """
    root = Path(os.getcwd()).absolute()
    test_pdfs_dir_path = root.parent / 'test_pdfs'
    test_pdfs_paths = [str(test_pdfs_dir_path/f) for f in os.listdir(test_pdfs_dir_path)]
    return test_pdfs_paths


def fill_queue_with_pdfs(queue, pdfs_paths):
    for pdf_path in pdfs_paths:
        # Send a message with a 2 second delay
        message_id = queue.sendMessage().message({
            'pdf_path': pdf_path
        }).execute()

        pprint({'queue_status': queue.getQueueAttributes().execute()})



# Create controller.
# In this case we are specifying the host and default queue name
queue = RedisSMQ(host="192.168.99.101", qname="my-queue")

# Delete Queue if it already exists, ignoring exceptions
queue.deleteQueue().exceptions(False).execute()

# Create Queue with default visibility timeout of 20 and delay of 0
# demonstrating here both ways of setting parameters
queue.createQueue(delay=0).vt(20).execute()


fill_queue_with_pdfs(queue, get_test_pdfs_paths())

# No action
queue.quit()