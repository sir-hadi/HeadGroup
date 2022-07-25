import time
import random
import paho.mqtt.client as paho

def on_connect(client, userdata, flags, rc):
  print('CONNECTED, received with code %d.' % (rc))

broker = "broker.hivemq.com"
port = 1883
topic_header = "/headGroup/"
client_id = f'head-group-{random.randint(0, 10)}'

client = paho.Client(client_id)
client.on_connect = on_connect
client.connect(broker, port)

# Connect to MQTT Broker
while( not client.is_connected() ):
  print('.', end='')
  time.sleep(1)