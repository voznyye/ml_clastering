req:
	pip install -r requirements.txt

runc:
	python clustering.py

runv:
	python visualization.py

rune:
	python elbow.py
	
runs:
	python silhouette.py
