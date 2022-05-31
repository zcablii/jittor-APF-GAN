make $1
while [ $? -ne 0 ]
do
	# ps -ef | grep 23335 | grep -v "grep"
    echo "process has been restarted!"
    # python debug.py
    make $1
done
echo "process finished!"
