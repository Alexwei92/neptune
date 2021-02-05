import setup_path
import airsim
import time

client = airsim.CarClient()
client.confirmConnection()

while True:
    data_car1 = client.getDistanceSensorData(vehicle_name="Car1")
    print(data_car1)
    time.sleep(0.5)