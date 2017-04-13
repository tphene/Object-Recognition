import time
import read_data as rd

#out = rd.read_data(11,15, "003.backpack", 128)
out = rd.read_data(11,15, "012.binoculars", 128)
print len(out)
for i in out:
	i.show()
	time.sleep(1)