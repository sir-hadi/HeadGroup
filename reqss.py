import requests

cam_id = 8
data = {1:22,2:32,4:11}

url = 'http://127.0.0.1:5050/sl_vision_post/'

for key, value in data.items():
    parms = {'id_cam' : cam_id, 'id_dot': key, 'people_count': value}
    print(parms)
    x = requests.post(url, params=parms)
    print('status code :',x.status_code)
    del parms