import requests
import os

data_url = "http://spider.cd.cse/~gpclend/unclassified_data_sets/"

def get_file(file_name, outdir, max_attempts = 3):
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    if os.path.exists(outdir + "/" + file_name):
        return True
    
    try:
        r = requests.get(data_url + file_name)
        open(outdir + "/" + file_name, "wb").write(r.content)
        return True
    
    except e:
        return False
        
    return False
        